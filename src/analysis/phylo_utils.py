import numpy as np
import torch
import wandb
from torch import nn


class PhyloSamplingMetrics(nn.Module):
    """Simple sampling diagnostics for phylogenetic graphs.

    We summarize generated graphs with a few coarse statistics and compare them
    against the validation/test splits. This keeps the interface compatible
    with the existing training loop while avoiding heavy spectral metrics.
    """

    def __init__(self, datamodule):
        super().__init__()
        self.reference_stats = {
            'val': self._stats_from_loader(datamodule.val_dataloader()),
            'test': self._stats_from_loader(datamodule.test_dataloader()),
        }

    def reset(self):
        # Nothing to reset for now, but the hook keeps parity with other
        # sampling metric implementations.
        return None

    def forward(self, generated_graphs, name, current_epoch, val_counter, local_rank, test=False):
        target_split = 'test' if test else 'val'
        reference_stats = self.reference_stats[target_split]
        generated_stats = self._stats_from_generated(generated_graphs)

        if local_rank == 0:
            print(f"[PhyloSamplingMetrics] Reference ({target_split}) stats: {reference_stats}")
            print(f"[PhyloSamplingMetrics] Generated stats: {generated_stats}")
            delta = {key: generated_stats[key] - reference_stats.get(key, 0.0)
                     for key in generated_stats}
            print(f"[PhyloSamplingMetrics] Delta: {delta}")

        if wandb.run:
            log_payload = {}
            for key, value in generated_stats.items():
                ref_value = reference_stats.get(key, 0.0)
                log_payload[f'sampling/{key}_gen'] = value
                log_payload[f'sampling/{key}_ref'] = ref_value
                log_payload[f'sampling/{key}_delta'] = value - ref_value
            wandb.log(log_payload, commit=False)

    def _stats_from_loader(self, loader):
        nodes = []
        edges = []
        clone_fraction = []
        mutation_fraction = []
        densities = []

        for batch in loader:
            data_list = batch.to_data_list()
            for data in data_list:
                node_types = torch.argmax(data.x, dim=-1)
                n = int(node_types.size(0))
                nodes.append(n)

                clones = int((node_types == 1).sum().item())
                mutations = int((node_types == 2).sum().item())
                clone_fraction.append(self._safe_ratio(clones, n))
                mutation_fraction.append(self._safe_ratio(mutations, n))

                edge_types = torch.argmax(data.edge_attr, dim=-1)
                num_edges = int((edge_types > 0).sum().item() / 2)
                edges.append(num_edges)
                densities.append(self._safe_ratio(num_edges, n * (n - 1) / 2))

        return self._aggregate(nodes, edges, clone_fraction, mutation_fraction, densities)

    def _stats_from_generated(self, generated_graphs):
        nodes = []
        edges = []
        clone_fraction = []
        mutation_fraction = []
        densities = []

        for node_types, edge_types in generated_graphs:
            node_types = torch.as_tensor(node_types)
            valid_mask = node_types >= 0
            n = int(valid_mask.sum().item())

            if n == 0:
                continue

            node_types = node_types[valid_mask]
            nodes.append(n)

            clones = int((node_types == 1).sum().item())
            mutations = int((node_types == 2).sum().item())
            clone_fraction.append(self._safe_ratio(clones, n))
            mutation_fraction.append(self._safe_ratio(mutations, n))

            adjacency = torch.as_tensor(edge_types)
            adjacency = adjacency[:n, :n]
            num_edges = int(torch.triu(adjacency > 0, diagonal=1).sum().item())
            edges.append(num_edges)
            densities.append(self._safe_ratio(num_edges, n * (n - 1) / 2))

        return self._aggregate(nodes, edges, clone_fraction, mutation_fraction, densities)

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _aggregate(nodes, edges, clone_fraction, mutation_fraction, densities):
        def safe_mean(values):
            return float(np.mean(values)) if values else 0.0

        def safe_std(values):
            return float(np.std(values)) if values else 0.0

        return {
            'num_graphs': float(len(nodes)),
            'mean_nodes': safe_mean(nodes),
            'std_nodes': safe_std(nodes),
            'mean_edges': safe_mean(edges),
            'std_edges': safe_std(edges),
            'mean_density': safe_mean(densities),
            'mean_clone_fraction': safe_mean(clone_fraction),
            'mean_mutation_fraction': safe_mean(mutation_fraction),
        }

