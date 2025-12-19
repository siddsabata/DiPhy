"""
PhyloGraph: Convert Newick tree + SNV events to graph representation

This module transforms raw output from SISTEM (Newick tree + SNV event list) 
into a graph representation. 

Citation: 
SISTEM: simulation of tumor evolution, metastasis, and DNA-seq data under genotype-driven selection
Samson Weiner and Mukul S. Bansal
Under Review

https://sistem.readthedocs.io/en/latest/index.html
"""

import pandas as pd
import numpy as np
from collections import defaultdict
import re
from typing import Dict, List, Tuple, Set, Optional, Any
from bigtree import newick_to_tree


class Phylogeny:
    
    def __init__(self, newick_file: str, snv_file: str) -> None:
        """
        Initialize Phylogeny with SISTEM files
        
        Args:
            newick_file: Path to Newick tree file
            snv_file: Path to SNV events TSV file
        """
        self.newick_file: str = newick_file
        self.snv_file: str = snv_file
        self.clone_mutations: Dict[str, List[int]] = defaultdict(list)  # clone_id -> list of mutation indices
        self.tree_structure: Dict[str, List[str]] = {}  # parent -> [children]
        self.all_clones: Set[str] = set()
        
    def _parse_snv(self) -> Dict[str, List[int]]:
        """
        Parse SNV events file and map clones to mutations 
        
        Returns:
            dict: Maps clone_id to list of mutation row indices
        """
        df = pd.read_csv(self.snv_file, sep='\t')
        
        # Group mutations by clone
        for idx, row in df.iterrows():
            if pd.isna(row['Cell/Clone']):  
                continue
                
            clone_id = row['Cell/Clone']
            self.clone_mutations[clone_id].append(idx + 1)  
            self.all_clones.add(clone_id)
            
        return self.clone_mutations
    
    def _parse_newick(self) -> Dict[str, List[str]]:
        """
        Parse Newick tree to map parent-child relationships 
        
        Returns:
            dict: Maps parent_id to list of child_ids
        """
        with open(self.newick_file, 'r') as f:
            newick_str = f.read().strip()
        
        root = newick_to_tree(newick_str)
        
        tree_structure = self._extract_parent_child_relationships(root)
        
        # Convert root name from "P0;" to just "P0" for consistency
        if root.name in tree_structure:
            tree_structure['P0'] = tree_structure.pop(root.name)
        
        # Add all nodes to the set
        for parent, children in tree_structure.items():
            self.all_clones.add(parent)
            for child in children:
                self.all_clones.add(child)
        
        self.tree_structure = tree_structure
            
        return tree_structure
    
    def _remove_mutationless_clones(self) -> Dict[str, List[str]]:
        """
        Remove clones that have no mutations and move their children to their parent
        
        Returns:
            Dict[str, List[str]]: Updated tree structure with mutationless clones removed
        """
        # Identify clones without mutations
        mutationless_clones = []
        for clone in self.all_clones:
            if clone not in self.clone_mutations and clone != 'P0':  # P0 is root, keep it
                mutationless_clones.append(clone)
        
        # Remove mutationless clones and restructure tree
        updated_tree = self.tree_structure.copy()
        
        for mutationless_clone in mutationless_clones:
            # Find parent of mutationless clone
            parent = None
            for p, children in updated_tree.items():
                if mutationless_clone in children:
                    parent = p
                    break
            
            if parent is None:
                continue
                
            # Get children of mutationless clone
            children_of_mutationless = updated_tree.get(mutationless_clone, [])
            
            # Remove mutationless clone from parent's children
            updated_tree[parent] = [c for c in updated_tree[parent] if c != mutationless_clone]
            
            # Add mutationless clone's children to its parent
            updated_tree[parent].extend(children_of_mutationless)
            
            # Remove mutationless clone from tree structure
            if mutationless_clone in updated_tree:
                del updated_tree[mutationless_clone]
        
        self.tree_structure = updated_tree
            
        return updated_tree
    
    def _create_graph(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Create X, E, L vectors for graph representation

        Node Types:
        - 0: Root node (normal cell)
        - 1: Clone nodes (cells with mutations)  
        - 2: Mutation nodes (individual SNV events)

        Edge Types:
        - 0: No edge
        - 1: Clone edge (undirected parent->child relationship)
        - 2: Mutation edge (undirected clone<->mutation association)
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (X, E, L) where:
                X: (n,) array of node types
                E: (n,n) adjacency matrix of edge types
                L: (n,) array of node labels
        """
        node_labels = []
        node_types = []
        
        node_labels.append('root')
        node_types.append(0)  # 0 = root
        
        # Add clone nodes (excluding root P0)
        clone_nodes = []
        for parent, children in self.tree_structure.items():
            if parent != 'P0':  # P0 becomes 'root'
                clone_nodes.append(parent)
            clone_nodes.extend(children)
        
        # Remove duplicates and sort for consistency 
        clone_nodes = sorted(list(set(clone_nodes)))
        
        for clone in clone_nodes:
            node_labels.append(clone)
            node_types.append(1)  # 1 = clone
        
        # Add mutation nodes
        mutation_labels = []
        for clone, mutation_indices in self.clone_mutations.items():
            for mut_idx in mutation_indices:
                mutation_label = f'mut_{mut_idx}'
                mutation_labels.append(mutation_label)
                node_labels.append(mutation_label)
                node_types.append(2)  # 2 = mutation
        
        n = len(node_labels)
        
        E = np.zeros((n, n), dtype=int)
        
        # Create label to index mapping
        label_to_idx = {label: idx for idx, label in enumerate(node_labels)}
        
        # Add clone edges (directed parent -> child)
        for parent, children in self.tree_structure.items():
            if parent == 'P0':
                parent_idx = 0  # Root index
            else:
                parent_idx = label_to_idx[parent]

            for child in children:
                child_idx = label_to_idx[child]
                E[parent_idx, child_idx] = 1  # 1 = clone edge
                E[child_idx, parent_idx] = 1  # 1 = clone edge (symmetric)
        
        # Add mutation edges (undirected clone <-> mutation)
        for clone, mutation_indices in self.clone_mutations.items():
            clone_idx = label_to_idx[clone]
            
            for mut_idx in mutation_indices:
                mutation_label = f'mut_{mut_idx}'
                mutation_idx = label_to_idx[mutation_label]
                
                # Undirected edge (both directions)
                E[clone_idx, mutation_idx] = 2  # 2 = mutation edge
                E[mutation_idx, clone_idx] = 2  # 2 = mutation edge
        
        X = np.array(node_types)
        L = np.array(node_labels)
        
        return X, E, L
    
    def transform(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Complete transformation
        
        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: (X, E, L) graph representation
        """
        
        # Step 1: Parse input files
        self._parse_snv()
        self._parse_newick()
        
        # Step 2: Remove mutationless clones
        self._remove_mutationless_clones()
        
        # Step 3: Create graph representation
        X, E, L = self._create_graph()
        
        return X, E, L

    def _extract_parent_child_relationships(self, node: Any, relationships: Optional[Dict[str, List[str]]] = None) -> Dict[str, List[str]]:
        """
        Helper method to recursively extract parent-child relationships
        """
        if relationships is None:
            relationships = {}
        
        children = []
        seen_children = set()
        for child in node.children:
            if child.name != "diploid":  # Skip diploid nodes
                if child.name in seen_children:
                    raise ValueError(f"Duplicate clone '{child.name}' in tree")
                children.append(child.name)
                seen_children.add(child.name)
                self._extract_parent_child_relationships(child, relationships)
        
        if children:
            relationships[node.name] = children
        
        return relationships


def main() -> None:

    # File paths
    newick_file = '/Users/siddharthsabata/dev/research/sistem-transform/dummy_data/clone_tree.nwk'
    snv_file = '/Users/siddharthsabata/dev/research/sistem-transform/dummy_data/SNV_events.tsv'
    
    # Create phylogeny and run pipeline
    phylo = Phylogeny(newick_file, snv_file)
    X, E, L = phylo.transform()
    
    # Print results
    print("\n" + "="*50)
    print("FINAL RESULTS")
    print("="*50)
    
    print(f"\nX (node types): {X}")
    print(f"Shape: {X.shape}")
    
    print(f"\nL (node labels): {L}")
    print(f"Shape: {L.shape}")
    
    print(f"\nE (adjacency matrix):")
    print(f"Shape: {E.shape}")
    print(E)

if __name__ == "__main__":
    main()
