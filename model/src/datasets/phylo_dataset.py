import json
import os
import pathlib
import pickle
from collections import OrderedDict
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch_geometric.data import Data, Dataset

from src.datasets.abstract_dataset import AbstractDataModule, AbstractDatasetInfos


class PhyloGraphDataset(Dataset):
    """Dataset that lazily loads phylogenetic trees from sharded tensors.

    Accepts a direct path to a pickle file containing phylogenetic trees.
    The file must contain a Python list of dictionaries with the keys
    ``tree_id``, ``X``, ``E`` and ``L``. During processing we shuffle the
    complete list once, persist deterministic indices for the train / val / test
    splits, and write compact shards of ``64`` graphs each to a cache directory.
    """

    SHARD_SIZE: int = 64

    def __init__(
        self,
        data_path: str,
        split: str,
        transform=None,
        *,
        split_ratios: Sequence[float] = (0.8, 0.1, 0.1),
        split_seed: int = 0,
        shard_size: int = SHARD_SIZE,
        max_cached_shards: int = 1,
    ) -> None:
        self.split = split
        self.split_ratios = tuple(split_ratios)
        self.split_seed = int(split_seed)

        # Store direct path to the raw pickle file
        self._data_path = pathlib.Path(data_path).resolve()
        if not self._data_path.exists():
            raise FileNotFoundError(f"Dataset file not found: {self._data_path}")

        # Create cache directory next to the data file, named after the pkl file
        self._cache_dir = self._data_path.parent / self._data_path.stem

        self.num_node_types = 3  # 0 = root, 1 = clone, 2 = mutation
        self.num_edge_types = 3  # 0 = no edge, 1 = clone edge, 2 = mutation edge
        self.shard_size = max(1, int(shard_size))
        self.max_cached_shards = max(1, int(max_cached_shards))
        self._shard_cache: "OrderedDict[str, List[Dict[str, Any]]]" = OrderedDict()

        # Initialize attributes that parent Dataset expects
        # We don't call super().__init__() since we manage paths ourselves
        self.transform = transform
        self.pre_transform = None
        self.pre_filter = None
        self._indices = None  # Required by PyTorch Geometric Dataset

        # Ensure cache directory exists
        self._cache_dir.mkdir(parents=True, exist_ok=True)

        # Process data if needed and load index
        self._ensure_processed()
        self._load_index()

    @property
    def raw_paths(self) -> List[str]:
        return [str(self._data_path)]

    @property
    def processed_dir(self) -> str:
        return str(self._cache_dir)

    @property
    def raw_file_names(self) -> List[str]:
        return [self._data_path.name]

    @property
    def processed_file_names(self) -> List[str]:
        # We always materialise at least the index for this split. Shards live
        # alongside the index inside ``processed/<split>`` and are enumerated at
        # runtime, so we only declare the index here for the Dataset API.
        return [f"{self.split}_index.json"]

    @property
    def processed_paths(self) -> List[str]:
        return [os.path.join(self.processed_dir, f) for f in self.processed_file_names]

    def _ensure_processed(self) -> None:
        """Check if processed data exists, otherwise trigger processing."""
        index_path = self.processed_paths[0]
        if not os.path.exists(index_path):
            self.process()

    def process(self) -> None:
        # Load the batch of phylogenetic trees for the requested split.
        with open(self.raw_paths[0], "rb") as handle:
            raw_graphs = pickle.load(handle)

        split_indices = self._load_or_create_split_indices(len(raw_graphs))
        chosen_indices = split_indices[self.split]

        shard_dir = os.path.join(self.processed_dir, self.split)
        os.makedirs(shard_dir, exist_ok=True)

        # Remove any stale shards from previous runs so the layout matches the
        # freshly generated index exactly.
        for filename in os.listdir(shard_dir):
            if filename.endswith(".pt"):
                os.remove(os.path.join(shard_dir, filename))

        legacy_path = os.path.join(self.processed_dir, f"{self.split}.pt")
        if os.path.exists(legacy_path):
            os.remove(legacy_path)

        index: List[Dict[str, object]] = []
        current_shard: List[Dict[str, Any]] = []
        shard_id = 0

        for raw_idx in chosen_indices:
            tree = raw_graphs[raw_idx]
            node_types_np = np.asarray(tree["X"], dtype=np.uint8)
            edge_types_np = np.asarray(tree["E"], dtype=np.uint8)

            node_types = torch.from_numpy(node_types_np).to(torch.uint8)

            edge_type_tensor = torch.from_numpy(edge_types_np).to(torch.uint8)
            edge_positions = torch.nonzero(edge_type_tensor, as_tuple=False)

            if edge_positions.numel() == 0:
                edge_index = torch.empty((2, 0), dtype=torch.int32)
                edge_types = torch.empty((0,), dtype=torch.uint8)
            else:
                edge_index = edge_positions.t().contiguous().to(torch.int32)
                edge_types = edge_type_tensor[edge_positions[:, 0], edge_positions[:, 1]]

            sample: Dict[str, Any] = {
                "node_types": node_types,
                "edge_index": edge_index,
                "edge_types": edge_types,
                "n_nodes": int(node_types.size(0)),
                "tree_id": tree.get("tree_id", ""),
                "node_labels": list(tree.get("L", [])),
            }

            current_shard.append(sample)
            shard_offset = len(current_shard) - 1
            index.append(
                {
                    "shard": shard_id,
                    "offset": shard_offset,
                    "tree_id": sample["tree_id"],
                    "raw_index": int(raw_idx),
                }
            )

            if len(current_shard) == self.shard_size:
                shard_path = os.path.join(shard_dir, f"shard_{shard_id:05d}.pt")
                torch.save(current_shard, shard_path)
                current_shard = []
                shard_id += 1

        if current_shard:
            shard_path = os.path.join(shard_dir, f"shard_{shard_id:05d}.pt")
            torch.save(current_shard, shard_path)

        index_path = self.processed_paths[0]
        with open(index_path, "w", encoding="utf-8") as handle:
            json.dump(
                {
                    "shard_size": self.shard_size,
                    "num_samples": len(index),
                    "entries": index,
                },
                handle,
            )

    def _load_or_create_split_indices(self, dataset_size: int) -> Dict[str, List[int]]:
        """Create or reload deterministic split indices stored on disk."""

        split_plan_path = os.path.join(self.processed_dir, "split_plan.pt")

        if os.path.exists(split_plan_path):
            plan: Dict[str, List[int]] = torch.load(split_plan_path, weights_only=False)
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

    def _load_index(self) -> None:
        """Read the cached shard index."""
        index_path = self.processed_paths[0]
        with open(index_path, "r", encoding="utf-8") as handle:
            payload = json.load(handle)

        entries = payload.get("entries", [])
        expected_count = payload.get("num_samples", len(entries))
        if len(entries) != expected_count:
            raise ValueError(
                "Shard index is inconsistent: num_samples does not match the number of entries."
            )

        self._entries = entries
        self._num_samples = expected_count
        self._stored_shard_size: int = int(payload.get("shard_size", self.shard_size))
        if self._stored_shard_size != self.shard_size:
            # Prefer the on-disk shard size to keep index lookups consistent.
            self.shard_size = self._stored_shard_size

        shard_dir = os.path.join(self.processed_dir, self.split)
        shard_ids = {int(entry["shard"]) for entry in self._entries}
        self._shard_paths: Dict[int, str] = {}
        for shard_id in sorted(shard_ids):
            shard_path = os.path.join(shard_dir, f"shard_{shard_id:05d}.pt")
            if not os.path.exists(shard_path):
                raise FileNotFoundError(
                    f"Expected shard '{shard_path}' for split '{self.split}' is missing."
                )
            self._shard_paths[shard_id] = shard_path

    def len(self) -> int:
        return self._num_samples

    def get(self, idx: int) -> Data:
        """Load a single graph on demand and materialise the PyG ``Data``."""

        entry = self._entries[idx]
        shard_id = int(entry["shard"])
        offset = int(entry["offset"])

        shard = self._load_shard(shard_id)
        sample = shard[offset]

        data = self._materialise_data(sample)
        if self.transform is not None:
            data = self.transform(data)
        return data

    # -- helpers -----------------------------------------------------------------
    def _load_shard(self, shard_id: int) -> List[Dict[str, Any]]:
        """Load a shard from disk, keeping a tiny LRU cache in memory."""

        cache_key = f"shard_{shard_id}"
        if cache_key in self._shard_cache:
            shard = self._shard_cache.pop(cache_key)
            # Move to end so LRU ordering is preserved.
            self._shard_cache[cache_key] = shard
            return shard

        shard_path = self._shard_paths[shard_id]
        shard: List[Dict[str, Any]] = torch.load(shard_path, map_location="cpu", weights_only=False)
        self._shard_cache[cache_key] = shard

        if len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return shard

    def _materialise_data(self, sample: Dict[str, torch.Tensor]) -> Data:
        """Reconstruct the PyG ``Data`` instance from compact shard tensors."""

        node_types = sample["node_types"].to(torch.long)
        n_nodes = int(sample["n_nodes"])

        x = F.one_hot(node_types, num_classes=self.num_node_types).float()

        edge_index = sample["edge_index"].to(torch.long)
        edge_types = sample["edge_types"].to(torch.long)
        if edge_index.numel() == 0:
            edge_attr = torch.empty((0, self.num_edge_types), dtype=torch.float)
        else:
            edge_attr = F.one_hot(edge_types, num_classes=self.num_edge_types).float()

        data = Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y=torch.zeros((1, 0), dtype=torch.float),
            n_nodes=torch.tensor([n_nodes], dtype=torch.long),
        )
        data.tree_id = sample.get("tree_id", "")
        data.node_labels = sample.get("node_labels", [])
        return data


class PhyloGraphDataModule(AbstractDataModule):
    """Lightning-compatible data module for the phylogenetic graphs."""

    def __init__(self, cfg, n_graphs: Optional[int] = None) -> None:
        self.cfg = cfg

        # Get direct path to the pickle file
        data_path = cfg.dataset.data_path

        split_ratios = getattr(self.cfg.dataset, "split_ratios", (0.8, 0.1, 0.1))
        split_seed = getattr(self.cfg.dataset, "split_seed", 0)
        shard_size = getattr(self.cfg.dataset, "shard_size", PhyloGraphDataset.SHARD_SIZE)
        max_cached_shards = getattr(self.cfg.dataset, "max_cached_shards", 1)

        # We instantiate the lazily loaded datasets per split. Each dataset only
        # pulls a tiny JSON index into memory during construction; actual graph
        # tensors are fetched on demand when the dataloaders request them.
        datasets = {
            split: PhyloGraphDataset(
                data_path=data_path,
                split=split,
                split_ratios=split_ratios,
                split_seed=split_seed,
                shard_size=shard_size,
                max_cached_shards=max_cached_shards,
            )
            for split in ("train", "val", "test")
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



