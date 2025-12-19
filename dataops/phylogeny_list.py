"""
Create pkl files containing lists of tree dictionaries:
[{tree_id, X, E, L}, {tree_id, X, E, L}, ...]

Usage:
    # For 500k trees in 5 batches of 100k each:
    pl = PhylogenyList(batch_size=100000, output_dir="model_data", 
                              base_dir="/path/to/data", pattern="out_*")
    pl.process_directories()
"""
import io
import sys
import os
import pickle
import glob
from typing import Dict, List
from phylogeny import Phylogeny


class PhylogenyList:
    
    def __init__(self, batch_size=100000, output_dir="batches", base_dir=".", pattern="out_*"):
        self.batch_size = batch_size
        self.output_dir = output_dir
        self.base_dir = base_dir
        self.pattern = pattern
        
        os.makedirs(output_dir, exist_ok=True)
        
    def process_directories(self):
        """
        Process all directories matching pattern and create pkl files
        """
        directories = glob.glob(os.path.join(self.base_dir, self.pattern))
        directories = [d for d in directories if os.path.isdir(d)]
        
        print(f"Found {len(directories)} directories to process")
        
        current_batch = []
        batch_num = 1
        processed_count = 0
        failed_count = 0
        
        for i, directory in enumerate(directories):
           
            tree_dict = self._process_directory(directory)
            
            if tree_dict:
                current_batch.append(tree_dict)
                processed_count += 1
                
                # Save batch when full
                if len(current_batch) >= self.batch_size:
                    self._save_batch(current_batch, batch_num)
                    current_batch = []
                    batch_num += 1
            else:
                failed_count += 1
        
        # Save remaining trees
        if current_batch:
            self._save_batch(current_batch, batch_num)
        
        print(f"Processing complete!")
        print(f"  Processed: {processed_count} directories")
        print(f"  Failed: {failed_count} directories") 
        print(f"  Created: {batch_num if current_batch else batch_num-1} batch files")
    
    def _process_directory(self, directory):
        try:
            newick_files = glob.glob(os.path.join(directory, "clone_tree.nwk"))
            snv_files = glob.glob(os.path.join(directory, "SNV_events.tsv"))

            if not newick_files or not snv_files:
                print(f"Missing files in {directory}: newick={len(newick_files)}, snv={len(snv_files)}")
                return None

            tree_id = os.path.basename(directory).replace("out_", "")
            return self._build_tree_dict(tree_id, newick_files[0], snv_files[0])

        except Exception as e:
            print(f"Error processing {directory}: {e}")
            return None
    
    def _save_batch(self, batch, batch_num):
        filename = f"batch_{batch_num:03d}.pkl"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, 'wb') as f:
            pickle.dump(batch, f)
        print(f"Saved {filepath} with {len(batch)} trees")

    def _build_tree_dict(self, tree_id: str, newick_file: str, snv_file: str) -> Dict[str, List]:
        """
        Transform a single tree into the portable dictionary representation that
        downstream consumers expect.  We keep this helper small so it can serve
        both the old batch mode and the new experiment traversal code.
        """
        old_stdout = sys.stdout
        sys.stdout = io.StringIO()

        try:
            phylo = Phylogeny(newick_file, snv_file)
            X, E, L = phylo.transform()
        finally:
            sys.stdout = old_stdout

        return {
            'tree_id': tree_id,
            'X': X.tolist(),
            'E': E.tolist(),
            'L': L.tolist()
        }

    def collect_experiment_replicates(self, base_dir: str) -> List[Dict[str, List]]:
        """
        Walk one or more experiment folders and collect every replicate tree.

        The expected on-disk shape mirrors the example in ``data_struct.txt``:

            exp_###/
              resamples/
                tumor_XXX/
                  rep_YYY/
                    clone_tree.nwk
                    SNV_events.tsv

        If ``base_dir`` already points to ``exp_###`` we use it as-is.  When it
        points to a parent directory we scan all immediate children that look
        like experiments.  Missing files are skipped with a short message so the
        user can inspect the underlying run.
        """
        base_dir = os.path.abspath(base_dir)

        if os.path.isdir(os.path.join(base_dir, "resamples")):
            experiment_dirs = [base_dir]
        else:
            experiment_dirs = [
                os.path.join(base_dir, name)
                for name in sorted(os.listdir(base_dir))
                if name.startswith("exp_") and os.path.isdir(os.path.join(base_dir, name))
            ]

        if not experiment_dirs:
            raise FileNotFoundError(
                f"Did not find any experiment directories under {base_dir}. Expected folders named exp_* with a resamples subfolder."
            )

        collected = []

        for experiment_dir in experiment_dirs:
            experiment_name = os.path.basename(experiment_dir)
            resamples_dir = os.path.join(experiment_dir, "resamples")

            if not os.path.isdir(resamples_dir):
                print(f"Skipping {experiment_dir} because resamples/ is missing")
                continue

            for tumor_name in sorted(os.listdir(resamples_dir)):
                tumor_dir = os.path.join(resamples_dir, tumor_name)
                if not os.path.isdir(tumor_dir):
                    continue

                for replicate_name in sorted(os.listdir(tumor_dir)):
                    replicate_dir = os.path.join(tumor_dir, replicate_name)
                    if not os.path.isdir(replicate_dir):
                        continue

                    newick_file = os.path.join(replicate_dir, "clone_tree.nwk")
                    snv_file = os.path.join(replicate_dir, "SNV_events.tsv")

                    if not (os.path.isfile(newick_file) and os.path.isfile(snv_file)):
                        print(
                            f"Skipping {replicate_dir} because clone_tree.nwk or SNV_events.tsv is missing"
                        )
                        continue

                    # Include enough context in the tree_id so downstream users
                    # can trace the origin of each entry without consulting the
                    # pickle again.
                    tree_id = f"{experiment_name}/{tumor_name}/{replicate_name}"
                    try:
                        tree_dict = self._build_tree_dict(tree_id, newick_file, snv_file)
                        collected.append(tree_dict)
                    except Exception as exc:
                        print(f"Error processing {replicate_dir}: {exc}")

        return collected

    def write_experiment_pickle(self, base_dir: str, output_path: str) -> None:
        """
        Gather all replicate trees under ``base_dir`` and write them to
        ``output_path`` as a single pickle containing a list of dictionaries.

        The caller is responsible for supplying an absolute or relative output
        path; directories are created as needed so the command can drop the file
        directly inside a chosen experiment folder.
        """
        trees = self.collect_experiment_replicates(base_dir)

        if not trees:
            raise RuntimeError(
                f"No trees collected from {base_dir}. Please check the directory structure and rerun."
            )

        os.makedirs(os.path.dirname(os.path.abspath(output_path)) or ".", exist_ok=True)

        with open(output_path, "wb") as handle:
            pickle.dump(trees, handle)

        print(f"Wrote {len(trees)} trees to {output_path}")

if __name__ == "__main__":
    # Keep the original batch behaviour available when the script is invoked
    # without extra arguments.  This avoids breaking older workflows while
    # giving us a lightweight entry point for experiment aggregation.
    if len(sys.argv) == 1:
        pl = PhylogenyList(batch_size=2000, output_dir="dataset",
                           base_dir="./dataset", pattern="out_*")
        pl.process_directories()
        sys.exit(0)

    # Minimal CLI: ``python phylogeny_list.py <base_dir> [output.pkl]``
    # When the output path is omitted we drop a combined pickle next to the
    # experiments so the command stays quick to run.
    if len(sys.argv) not in (2, 3):
        print("Usage: python phylogeny_list.py <base_dir> [output.pkl]")
        sys.exit(1)

    _, base_dir_arg, *maybe_output = sys.argv

    if maybe_output:
        output_path_arg = maybe_output[0]
    else:
        output_path_arg = os.path.join(os.path.abspath(base_dir_arg), "combined_experiments.pkl")

    aggregator = PhylogenyList()
    aggregator.write_experiment_pickle(base_dir_arg, output_path_arg)
