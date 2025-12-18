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
        validity_flags = []

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

                # Convert the sparse representation back to a dense adjacency so
                # that we can run the validity rules on the exact edge layout.
                adjacency = self._dense_adjacency(data.edge_index, edge_types, n)
                node_types_cpu = node_types.to(torch.int8).detach().cpu()
                is_valid = self._evaluate_validity(node_types_cpu, adjacency)
                validity_flags.append(1.0 if is_valid else 0.0)

        return self._aggregate(nodes, edges, clone_fraction, mutation_fraction, densities, validity_flags)

    def _stats_from_generated(self, generated_graphs):
        nodes = []
        edges = []
        clone_fraction = []
        mutation_fraction = []
        densities = []
        validity_flags = []

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

            node_types_cpu = node_types.to(torch.int8)
            adjacency_cpu = adjacency.to(torch.int8)
            is_valid = self._evaluate_validity(node_types_cpu, adjacency_cpu)
            validity_flags.append(1.0 if is_valid else 0.0)

        return self._aggregate(nodes, edges, clone_fraction, mutation_fraction, densities, validity_flags)

    @staticmethod
    def _safe_ratio(numerator, denominator):
        if denominator <= 0:
            return 0.0
        return float(numerator) / float(denominator)

    @staticmethod
    def _aggregate(nodes, edges, clone_fraction, mutation_fraction, densities, validity_flags):
        def safe_mean(values):
            return float(np.mean(values)) if values else 0.0

        def safe_std(values):
            return float(np.std(values)) if values else 0.0

        # The pass ratio expresses how many graphs respected every validity
        # rule. We multiply by ``100`` later to surface an intuitive
        # percentage.
        pass_ratio = safe_mean(validity_flags)

        return {
            'num_graphs': float(len(nodes)),
            'mean_nodes': safe_mean(nodes),
            'std_nodes': safe_std(nodes),
            'mean_edges': safe_mean(edges),
            'std_edges': safe_std(edges),
            'mean_density': safe_mean(densities),
            'mean_clone_fraction': safe_mean(clone_fraction),
            'mean_mutation_fraction': safe_mean(mutation_fraction),
            'validity_pass_pct': float(pass_ratio * 100.0),
        }

    @staticmethod
    def _dense_adjacency(edge_index, edge_types, num_nodes):
        """Rebuild a dense adjacency matrix from the PyG edge format."""
        adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.int8)
        if edge_index is None or edge_index.numel() == 0:
            return adjacency

        edge_index_cpu = edge_index.detach().cpu()
        edge_types_cpu = edge_types.detach().cpu()

        for idx in range(edge_index_cpu.size(1)):
            src = int(edge_index_cpu[0, idx].item())
            dst = int(edge_index_cpu[1, idx].item())
            edge_type = int(edge_types_cpu[idx].item())
            if edge_type == 0:
                continue
            adjacency[src, dst] = edge_type
            adjacency[dst, src] = edge_type

        return adjacency

    @staticmethod
    def _evaluate_validity(node_types, adjacency):
        """Run the mutation-graph validity rules and return a boolean flag."""
        # The validity checks operate on CPU NumPy arrays for readability. The
        # graphs are small, so the conversion cost is negligible while keeping
        # the implementation close to our reference helper.
        node_np = node_types.detach().cpu().numpy().astype(int)
        adj_np = adjacency.detach().cpu().numpy().astype(int)

        if node_np.size == 0:
            return False

        has_cycle = PhyloSamplingMetrics._has_clone_cycle(adj_np, node_np)
        root_ok = PhyloSamplingMetrics._check_root_degree(adj_np, node_np)
        clone_ok, mutation_ok = PhyloSamplingMetrics._validate_edge_types(adj_np, node_np)
        return (not has_cycle) and root_ok and clone_ok and mutation_ok

    @staticmethod
    def _clone_neighbors(adjacency, node_index):
        """Return indices connected to ``node_index`` via clone edges."""
        mask = adjacency[node_index] == 1
        return np.flatnonzero(mask)

    @staticmethod
    def _has_clone_cycle(adjacency, node_types):
        """Detect cycles inside the clone-only subgraph."""
        clone_like = np.flatnonzero((node_types == 0) | (node_types == 1))
        if clone_like.size == 0:
            return False

        visited = set()
        for start in clone_like:
            if start in visited:
                continue
            stack = [(int(start), -1)]
            while stack:
                node, parent = stack.pop()
                if node in visited:
                    if parent != -1:
                        return True
                    continue
                visited.add(node)
                neighbours = PhyloSamplingMetrics._clone_neighbors(adjacency, node)
                for neighbour in neighbours:
                    if neighbour == parent:
                        continue
                    if neighbour in visited:
                        return True
                    stack.append((int(neighbour), node))
        return False

    @staticmethod
    def _validate_edge_types(adjacency, node_types):
        """Check that clone/mutation edges connect the expected node types."""
        clone_ok = True
        mutation_ok = True
        node_count = adjacency.shape[0]

        for i in range(node_count):
            for j in range(i + 1, node_count):
                edge_type = adjacency[i, j]
                if edge_type == 0:
                    continue

                node_i_type = node_types[i]
                node_j_type = node_types[j]

                if edge_type == 1:
                    if not ((node_i_type in (0, 1)) and (node_j_type in (0, 1))):
                        clone_ok = False
                elif edge_type == 2:
                    valid_pair = ((node_i_type == 1 and node_j_type == 2) or
                                  (node_i_type == 2 and node_j_type == 1))
                    if not valid_pair:
                        mutation_ok = False

        return clone_ok, mutation_ok

    @staticmethod
    def _check_root_degree(adjacency, node_types):
        """Confirm the root node has exactly one clone neighbour."""
        root_indices = np.flatnonzero(node_types == 0)
        if root_indices.size != 1:
            return False

        root_idx = int(root_indices[0])
        neighbours = PhyloSamplingMetrics._clone_neighbors(adjacency, root_idx)
        clone_neighbours = [idx for idx in neighbours if node_types[idx] == 1]
        return len(clone_neighbours) == 1 and len(neighbours) == 1

