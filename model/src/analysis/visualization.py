import os
import imageio
import networkx as nx
import numpy as np
import wandb
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors

class NonMolecularVisualization:
    def to_networkx(self, node_list, adjacency_matrix):
        """
        Convert graphs to networkx graphs
        node_list: the nodes of a batch of nodes (bs x n)
        adjacency_matrix: the adjacency_matrix of the molecule (bs x n x n)
        """
        graph = nx.Graph()

        for i in range(len(node_list)):
            if node_list[i] == -1:
                continue
            graph.add_node(i, number=i, symbol=node_list[i], color_val=node_list[i])

        rows, cols = np.where(adjacency_matrix >= 1)
        edges = zip(rows.tolist(), cols.tolist())
        for edge in edges:
            edge_type = adjacency_matrix[edge[0]][edge[1]]
            if edge_type == 1:
                # Clone-to-clone edges: we keep them long by assigning a small
                # spring weight (weaker attraction) and indicate that they should
                # be drawn with a solid stroke later on.
                graph.add_edge(
                    edge[0],
                    edge[1],
                    color=float(edge_type),
                    edge_type=1,
                    spring_weight=0.5,
                )
            elif edge_type == 2:
                # Clone-to-mutation edges: these should appear shorter and dotted
                # so we strengthen the spring and mark the style explicitly.
                graph.add_edge(
                    edge[0],
                    edge[1],
                    color=float(edge_type),
                    edge_type=2,
                    spring_weight=2.0,
                )
            else:
                graph.add_edge(
                    edge[0],
                    edge[1],
                    color=float(edge_type),
                    edge_type=int(edge_type),
                    spring_weight=1.0,
                )

        return graph

    def visualize_non_molecule(self, graph, pos, path, iterations=100, node_size=100, largest_component=False):
        if largest_component:
            CGs = [graph.subgraph(c) for c in nx.connected_components(graph)]
            CGs = sorted(CGs, key=lambda x: x.number_of_nodes(), reverse=True)
            graph = CGs[0]

        # Plot the graph structure with colors
        if pos is None:
            # We always provide the spring layout with our custom spring weights
            # so clone edges (weight 0.5) end up longer than mutation edges (weight 2.0).
            pos = nx.spring_layout(graph, iterations=iterations, weight='spring_weight')

        node_colors = self._phylo_node_colors(graph)
        clone_edges, mutation_edges, other_edges = self._phylo_edge_groups(graph)

        plt.figure()

        if graph.number_of_nodes() > 0:
            nx.draw_networkx_nodes(
                graph,
                pos,
                node_color=node_colors,
                node_size=node_size,
                linewidths=0.5,
            )

        if clone_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=clone_edges,
                width=2.5,
                edge_color='#4a4a4a',
                style='solid',
            )

        if mutation_edges:
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=mutation_edges,
                width=1.8,
                edge_color='#4a4a4a',
                style='dotted',
            )

        if other_edges:
            # Draw any other edge types (unexpected but possible in early frames)
            # with a light dashed style so they remain visible without causing confusion.
            nx.draw_networkx_edges(
                graph,
                pos,
                edgelist=other_edges,
                width=1.5,
                edge_color='#b0b0b0',
                style='dashdot',
            )

        plt.tight_layout()
        plt.savefig(path)
        plt.close("all")

    def _phylo_node_colors(self, graph):
        """Assign colours for every node using our phylogenetic conventions.

        Requirements from the dataset:
        * ``symbol`` == 0 denotes the unique root node (colored grey).
        * ``symbol`` == 1 denotes clone nodes (assigned stable RGB colours).
        * ``symbol`` == 2 denotes mutation nodes (lighter shade of their parent clone).

        Any unexpected value is handled gracefully by falling back to a neutral
        grey so the visualization never crashes nor reverts to the legacy mode.
        """

        node_symbols = nx.get_node_attributes(graph, 'symbol')

        root_rgb = mcolors.to_rgb('#7f7f7f')
        default_clone_rgb = mcolors.to_rgb('#1f77b4')
        fallback_rgb = mcolors.to_rgb('#9e9e9e')

        clone_nodes = [node for node, symbol in node_symbols.items() if symbol == 1]

        cmap = plt.get_cmap('tab10')
        clone_colors = {}
        for idx, node in enumerate(sorted(clone_nodes)):
            base_rgba = cmap(idx % cmap.N)
            # Store an RGB triple for our palette and explain the mapping.
            clone_colors[node] = base_rgba[:3]

        colors = []
        for node in graph.nodes():
            symbol = node_symbols.get(node)
            if symbol == 0:
                colors.append(root_rgb)
            elif symbol == 1:
                colors.append(clone_colors.get(node, default_clone_rgb))
            elif symbol == 2:
                mutation_color = self._mutation_color_from_neighbors(
                    graph,
                    node,
                    clone_colors,
                    node_symbols,
                    default_clone_rgb,
                )
                colors.append(mutation_color)
            else:
                # Unknown label: keep the node visible with a neutral colour.
                colors.append(fallback_rgb)

        return colors

    def _phylo_edge_groups(self, graph):
        """Split the edges into clone, mutation, and fallback buckets."""

        edge_types = nx.get_edge_attributes(graph, 'edge_type')

        clone_edges = []
        mutation_edges = []
        other_edges = []

        for edge in graph.edges():
            e_type = edge_types.get(edge)
            if e_type == 1:
                clone_edges.append(edge)
            elif e_type == 2:
                mutation_edges.append(edge)
            else:
                other_edges.append(edge)

        return clone_edges, mutation_edges, other_edges

    def _mutation_color_from_neighbors(self, graph, node, clone_colors, node_symbols, default_clone_rgb):
        """Generate a lighter clone color for a mutation node.

        We inspect neighbouring clone nodes (type 1). If several clones connect
        to the mutation, we simply pick the first one; the surrounding structure
        still conveys the uncertainty. When no clone neighbour exists we fall
        back to a neutral grey so the node remains visible.
        """

        clone_neighbours = [nbr for nbr in graph.neighbors(node) if node_symbols.get(nbr) == 1]

        if len(clone_neighbours) == 1:
            base_clone = sorted(clone_neighbours)[0]
            base_rgb = clone_colors.get(base_clone, default_clone_rgb)
        elif len(clone_neighbours) > 1:
            # Ambiguous attachment: we treat it as an unresolved relationship and
            # fall back to the default clone colour so the viewer is not misled.
            base_rgb = default_clone_rgb
        else:
            # No clone neighbour yet: lighten the default clone colour so the mutation is still readable.
            base_rgb = default_clone_rgb

        return self._lighten_rgb(base_rgb, amount=0.5)

    def _lighten_rgb(self, rgb, amount=0.5):
        """Lighten a colour by mixing it with white.

        The ``amount`` parameter controls how far we move towards white. The
        value must lie in ``[0, 1]`` where ``0`` yields the original colour and
        ``1`` yields pure white.
        """

        base_r, base_g, base_b = mcolors.to_rgb(rgb)
        amount = float(np.clip(amount, 0.0, 1.0))

        def _towards_white(component):
            return component + (1.0 - component) * amount

        return (_towards_white(base_r), _towards_white(base_g), _towards_white(base_b))

    def visualize(self, path: str, graphs: list, num_graphs_to_visualize: int, log='graph'):
        # Skip if nothing to visualize
        if num_graphs_to_visualize <= 0:
            return

        # define path to save figures
        if not os.path.exists(path):
            os.makedirs(path)

        # visualize the final molecules
        for i in range(num_graphs_to_visualize):
            file_path = os.path.join(path, 'graph_{}.png'.format(i))
            graph = self.to_networkx(graphs[i][0].numpy(), graphs[i][1].numpy())
            self.visualize_non_molecule(graph=graph, pos=None, path=file_path)
            im = plt.imread(file_path)
            # Disabled: too noisy for W&B dashboard
            # if wandb.run and log is not None:
            #     wandb.log({log: [wandb.Image(im, caption=file_path)]})

    def visualize_chain(self, path, nodes_list, adjacency_matrix):
        # convert graphs to networkx
        graphs = [self.to_networkx(nodes_list[i], adjacency_matrix[i]) for i in range(nodes_list.shape[0])]
        # find the coordinates of atoms in the final molecule
        final_graph = graphs[-1]
        final_pos = nx.spring_layout(final_graph, seed=0)

        # draw gif
        save_paths = []
        num_frams = nodes_list.shape[0]

        for frame in range(num_frams):
            file_name = os.path.join(path, 'fram_{}.png'.format(frame))
            self.visualize_non_molecule(graph=graphs[frame], pos=final_pos, path=file_name)
            save_paths.append(file_name)

        imgs = [imageio.imread(fn) for fn in save_paths]
        gif_path = os.path.join(os.path.dirname(path), '{}.gif'.format(path.split('/')[-1]))
        imgs.extend([imgs[-1]] * 10)
        imageio.mimsave(gif_path, imgs, subrectangles=True, duration=20)
        # Disabled: too noisy for W&B dashboard
        # if wandb.run:
        #     wandb.log({'chain': [wandb.Video(gif_path, caption=gif_path, format="gif")]})
