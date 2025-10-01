import os
import pathlib
import pickle
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, InMemoryDataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class PhyloGraphDataset(InMemoryDataset):
    """Dataset that loads phylogenetic trees stored as pickled dictionaries.

    We expect a single raw pickle file ``phylo.pkl`` placed inside ``<root>/raw``.
    The file must contain a Python list of dictionaries with the keys
    ``tree_id``, ``X``, ``E`` and ``L`` as described in the data overview. During
    processing we shuffle the complete list once, persist deterministic indices
    for the train / val / test splits, and materialise the tensors for the
    requested split.
    """

    def __init__(
        self,
        root: str,
        split: str,
        transform=None,
        pre_transform=None,
        pre_filter=None,
        *,
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
    ) -> None:
        self.split = split
        self.split_ratios = tuple(split_ratios)
        self.split_seed = int(split_seed)
        self.num_node_types = 3  # 0 = root, 1 = clone, 2 = mutation
        self.num_edge_types = 3  # 0 = no edge, 1 = clone edge, 2 = mutation edge
        super().__init__(root, transform, pre_transform, pre_filter)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self) -> List[str]:
        # A single pickle file contains the complete dataset; all splits read
        # from it and slice their portion using cached indices.
        return ["phylo.pkl"]

    @property
    def processed_file_names(self) -> List[str]:
        # Processed tensors for each split are cached separately so that we can
        # reload them without touching the raw pickles again.
        return [f"{self.split}.pt"]

    def download(self) -> None:
        # The user is responsible for placing the pickle files locally, so we
        # only provide a friendly error if the expected file is missing.
        expected = self.raw_paths[0]
        if not os.path.exists(expected):
            raise FileNotFoundError(
                f"Expected raw pickle for split '{self.split}' at '{expected}'."
                " Please place the file manually before instantiating the dataset."
            )

    def process(self) -> None:
        # Load the batch of phylogenetic trees for the requested split.
        with open(self.raw_paths[0], "rb") as handle:
            raw_graphs = pickle.load(handle)

        split_indices = self._load_or_create_split_indices(len(raw_graphs))
        chosen_indices = split_indices[self.split]

        data_list: List[Data] = []

        for idx in chosen_indices:
            tree = raw_graphs[idx]
            node_types_np = np.asarray(tree["X"], dtype=np.int64)
            edge_types_np = np.asarray(tree["E"], dtype=np.int64)

            node_types = torch.from_numpy(node_types_np).long()
            # We keep the features as one-hot vectors so that training code can
            # immediately distinguish root, clone and mutation nodes.
            x = F.one_hot(node_types, num_classes=self.num_node_types).float()

            # The y target is intentionally empty because we only care about
            # unconditional generation for this dataset.
            y = torch.zeros((1, 0), dtype=torch.float)

            # Edge types use the convention: 0 = no edge, 1 = clone edge,
            # 2 = mutation association edge. We skip zero entries and keep the
            # provided directions as-is.
            edge_type_tensor = torch.from_numpy(edge_types_np).long()
            edge_positions = torch.nonzero(edge_type_tensor, as_tuple=False)

            if edge_positions.numel() == 0:
                # We still need a valid tensor even for isolated graphs, so we
                # create empty placeholders with the correct shapes.
                edge_index = torch.empty((2, 0), dtype=torch.long)
                edge_attr = torch.empty((0, self.num_edge_types), dtype=torch.float)
            else:
                edge_index = edge_positions.t().contiguous()
                edge_types = edge_type_tensor[edge_positions[:, 0], edge_positions[:, 1]]

                edge_attr = torch.zeros(
                    (edge_positions.size(0), self.num_edge_types), dtype=torch.float
                )
                edge_attr[torch.arange(edge_positions.size(0)), edge_types] = 1.0

            num_nodes = torch.tensor([node_types.size(0)], dtype=torch.long)

            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                y=y,
                n_nodes=num_nodes,
            )

            # Attach metadata so downstream analysis can trace back to the raw tree.
            data.tree_id = tree.get("tree_id", "")
            data.node_labels = list(tree.get("L", []))

            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            data_list.append(data)

        torch.save(self.collate(data_list), self.processed_paths[0])

    def _load_or_create_split_indices(self, dataset_size: int) -> Dict[str, List[int]]:
        """Create or reload deterministic split indices stored on disk."""

        split_plan_path = os.path.join(self.processed_dir, "split_plan.pt")

        if os.path.exists(split_plan_path):
            plan: Dict[str, List[int]] = torch.load(split_plan_path)
            return plan

        ratios = torch.tensor(self.split_ratios, dtype=torch.float)
        ratios = ratios / ratios.sum()

        generator = torch.Generator()
        generator.manual_seed(self.split_seed)

        shuffled_indices = torch.randperm(dataset_size, generator=generator).tolist()

        train_len = int(round(dataset_size * ratios[0].item()))
        train_len = min(train_len, dataset_size)
        val_len = int(round(dataset_size * ratios[1].item()))
        val_len = min(val_len, dataset_size - train_len)
        test_len = dataset_size - train_len - val_len

        plan = {
            "train": shuffled_indices[:train_len],
            "val": shuffled_indices[train_len:train_len + val_len],
            "test": shuffled_indices[train_len + val_len:train_len + val_len + test_len],
        }

        torch.save(plan, split_plan_path)
        return plan


class PhyloGraphDataModule(AbstractDataModule):
    """Lightning-compatible data module for the phylogenetic graphs."""

    def __init__(self, cfg, n_graphs: int = None) -> None:
        self.cfg = cfg
        self.datadir = cfg.dataset.datadir
        base_path = pathlib.Path(os.path.realpath(__file__)).parents[2]
        root_path = os.path.join(base_path, self.datadir)

        split_ratios = getattr(self.cfg.dataset, "split_ratios", (0.8, 0.1, 0.1))
        split_seed = getattr(self.cfg.dataset, "split_seed", 0)

        datasets = {
            "train": PhyloGraphDataset(
                root=root_path,
                split="train",
                split_ratios=split_ratios,
                split_seed=split_seed,
            ),
            "val": PhyloGraphDataset(
                root=root_path,
                split="val",
                split_ratios=split_ratios,
                split_seed=split_seed,
            ),
            "test": PhyloGraphDataset(
                root=root_path,
                split="test",
                split_ratios=split_ratios,
                split_seed=split_seed,
            ),
        }

        super().__init__(cfg, datasets)
        self.inner = self.train_dataset

    def __getitem__(self, index):
        return self.inner[index]


class PhyloDatasetInfos(AbstractDatasetInfos):
    """Holds dataset-level statistics used by the diffusion model."""

    def __init__(self, datamodule: PhyloGraphDataModule, dataset_config) -> None:
        self.datamodule = datamodule
        self.name = "phylo_graphs"
        self.n_nodes = self.datamodule.node_counts()
        # Estimate the empirical distribution of node categories from the data.
        self.node_types = self.datamodule.node_types()
        self.edge_types = self.datamodule.edge_counts()
        super().complete_infos(self.n_nodes, self.node_types)


