import os, sys
import fcntl
from contextlib import contextmanager
from typing import Tuple, List, Optional, Callable, Union
import time
import numpy as np
import argparse

import ROOT

from .tree_splitter import TreeSplitter


"""
quick fit utility for batch fitting with multi-dimensional binning support, 
using TreeSplitter for handling TTree splitting and snapshot creation.
version : 1.0.0
Date    : 2026-03-25
Author  : wangzheng
"""

class QUICK_FIT():
    """
    Batch fitting with multi-dimensional binning support

    The fit function should take the following parameters:
    - input_data: ROOT.TTree or ROOT.TH1 object containing the data to fit
    - output_file: str, path to save the fit results
    - log_file: str, path to save the log output
    - range_use: str or tuple, range description for the fit
    - binned_fit: bool, whether to perform binned fit
    - **kwargs: additional keyword arguments for flexibility
    
    Output: result, nsig, nsig_err
    """

    def __init__(self,
                 fit_function: Callable, 
                 bin_var_config: Optional[List[Tuple]] = None,
                 tree_path: str = "",
                 output_dir: str = "",
                 binned_fit: bool = False):
        """
        Args:
            fit_function: Callable fit function
            bin_var_config: List of tuples for binning.
            tree_path: Path to input ROOT file
            output_dir: Output directory for fit results
            binned_fit: If True, perform binned fit; if False, unbinned fit
        """
        self.fit_function = fit_function
        self.binned_fit = binned_fit
        self.tree_path = tree_path
        self.output_dir = output_dir
        
        if bin_var_config is not None:
            # Initialize TreeSplitter
            self.splitter = TreeSplitter(tree_path, bin_var_config)
            
            # Expose properties expected by existing methods
            self.n_dimensions = self.splitter.n_dimensions
            self.total_bins = self.splitter.total_bins
            self.bins_per_dim = self.splitter.bins_per_dim
            print(f"Initialized {self.n_dimensions}D binning with {self.total_bins} total bins using TreeSplitter")
        else:
             self.n_dimensions = 0
             self.total_bins = 1
             self.bins_per_dim = []
             self.splitter = None
             print("Initialized without binning config (single bin mode)")

    def batch_fit(self, bins_to_fit: Optional[List[int]] = None,
                  vec_br_to_keep: Optional[List[str]] = None,
                  additional_cut: str = ""):
        """
        Perform batch fitting - always passes TTree to fit function
        
        Args:
            bins_to_fit: List of flat bin indices to fit
            vec_br_to_keep: List of vector branch to handle properly when splitting tree
            additional_cut: Additional filter condition
        """
        start_time = time.time()

        tree_path = self.tree_path
        output_dir = self.output_dir
        fit_function = self.fit_function

        print(f"Fit mode: {'Binned' if self.binned_fit else 'Unbinned'}")
        os.makedirs(output_dir, exist_ok=True)

        # Initialize results array
        results_columns = 2 * self.n_dimensions + 3 
        results = np.zeros((self.total_bins, results_columns))

        # Initialize results file
        results_file = os.path.join(output_dir, "nsig_results.csv")
        if not os.path.isfile(results_file):
            self._initialize_results_file(results_file)

        if bins_to_fit is None:
            bins_to_fit = list(range(self.total_bins))
        bins_to_fit = [i for i in bins_to_fit if 0 <= i < self.total_bins]

        print(f"Will process {len(bins_to_fit)} bins out of {self.total_bins} total bins")

        successful_fits = 0
        failed_bins = []
        
        # Process each bin
        for flat_bin_idx in bins_to_fit:
            try:
                bin_indices = self.splitter._get_bin_indices(flat_bin_idx)
                ranges = self.splitter._get_bin_ranges(bin_indices)
                print("-----------------------------------------------------------------------")
                print(f"Processing bin: {flat_bin_idx}")
                print(f"Bin indices: {bin_indices}")
                
                pad_width = len(str(self.total_bins - 1))
                bin_output = f"{output_dir}bin_{flat_bin_idx:0{pad_width}d}"
                bin_log_file = f"{output_dir}bin_{flat_bin_idx:0{pad_width}d}.log"
                temp_file_path = f"{output_dir}temp_bin_{flat_bin_idx}.root"
                
                if not os.path.exists(temp_file_path):
                     # Use TreeSplitter to create snapshot
                    success = self.splitter.create_bin_snapshot(
                        flat_bin_idx,
                        temp_file_path,
                        vec_br_to_keep=vec_br_to_keep,
                        additional_cut=additional_cut
                    )
                    if not success:
                        print(f"Snapshot creation failed or empty for bin {flat_bin_idx}")
                        failed_bins.append(flat_bin_idx)
                        continue
                else:
                     print(f"Using existing file: {temp_file_path}")

                # Perform fit
                f_temp = ROOT.TFile.Open(temp_file_path)
                tree_temp = f_temp.Get("event")
                
                if not tree_temp or tree_temp.GetEntries() == 0:
                     print(f"Warning: Empty tree for bin {flat_bin_idx}")
                     f_temp.Close() 
                     failed_bins.append(flat_bin_idx)
                     continue

                try:
                    range_lines = []
                    for dim_idx, (bin_min, bin_max) in enumerate(ranges):
                        var_name = self.splitter.var_names[dim_idx]
                        range_lines.append(f"{var_name}: [{bin_min:.3f}, {bin_max:.3f}]")
                    range_use_str = ";".join(range_lines)

                    result, nsig, nsig_err = fit_function(
                        tree_temp,         
                        bin_output,
                        bin_log_file,
                        range_use_str,
                        binned_fit=self.binned_fit,
                    )
                    
                    # Store results
                    result_row = []
                    for dim_idx, (bin_min, bin_max) in enumerate(ranges):
                        result_row.extend([(bin_min + bin_max) / 2, (bin_max - bin_min) / 2])
                    
                    result_row.extend([nsig, nsig_err, nsig_err])
                    self._save_single_bin_result(results_file, flat_bin_idx, np.array(result_row))
                    
                    print(f"Signal yield: {nsig:.2f} ± {nsig_err:.2f}")
                    print("Output saved to:", bin_output)

                    
                    fit_status = result.status()
                    if fit_status == 0:
                        print("Fit converged successfully!")
                        successful_fits += 1
                    else:
                        status_codes = {
                            0: "successful fit",
                            1: "covariance was made positive definite",
                            2: "Hesse is invalid",
                            3: "EDM is above max",
                            4: "Reached call limit",
                            5: "other failure"
                        }
                        failed_bins.append(flat_bin_idx)
                        print(f"Fit had issues: {status_codes.get(fit_status, 'unknown error')}")
                        
                except Exception as e:
                    print(f"Fit failed for bin {flat_bin_idx}: {e}")
                    import traceback
                    traceback.print_exc()
                    failed_bins.append(flat_bin_idx)
                finally:
                    f_temp.Close()

            except Exception as e:
                print(f"Error processing bin {flat_bin_idx}: {e}")
                failed_bins.append(flat_bin_idx)

        # Save results (full array save skipped as we save incrementally)
        #self._save_results(results_file, results)

        end_time = time.time()
        print("-----------------------------------------------------------------------")
        print(f"Batch fitting complete! Successfully fit {successful_fits}/{len(bins_to_fit)} bins.")
        if failed_bins:
            print(f"Failed bins: {failed_bins}")
        print(f"Total time: {end_time - start_time:.1f} seconds")

    def _save_results(self, results_file: str, results: np.ndarray):
        """Create results CSV with header if it doesn't exist yet."""
        if not os.path.isfile(results_file):
            with open(results_file, "w") as f:
                header_parts = []
                for dim_idx in range(self.n_dimensions):
                    var_name = self.bin_var_configs[dim_idx][0]
                    header_parts.extend([f"{var_name}_center", f"{var_name}_width"])
                header_parts.extend(["nsig", "nsig_err_lo", "nsig_err_hi"])
                f.write(",".join(header_parts) + "\n")
                for i in range(self.total_bins):
                    f.write(",".join(["0.0000"] * results.shape[1]) + "\n")
            print(f"Initialized results file: {results_file}")

    def _save_single_bin_result(self, results_file: str, bin_idx: int, result_row: np.ndarray):
        """Write one bin's result into the CSV (thread-safe)."""
        lock_file = results_file + ".lock"
        with open(lock_file, 'w') as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                if not os.path.isfile(results_file):
                    self._initialize_results_file(results_file)
                with open(results_file, "r") as f:
                    lines = f.readlines()
                line_idx = bin_idx + 1  # +1 for header
                if line_idx < len(lines):
                    lines[line_idx] = ",".join([f"{val:.4f}" for val in result_row]) + "\n"
                    with open(results_file, "w") as f:
                        f.writelines(lines)
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    
    def _initialize_results_file(self, results_file: str):
        """Create CSV file with header and zero-filled rows."""
        header_parts = []
        for dim_idx in range(self.n_dimensions):
            var_name = self.splitter.var_names[dim_idx]
            header_parts.extend([f"{var_name}_center", f"{var_name}_width"])
        header_parts.extend(["nsig", "nsig_err_lo", "nsig_err_hi"])
        n_cols = 2 * self.n_dimensions + 3
        with open(results_file, "w") as f:
            f.write(",".join(header_parts) + "\n")
            for i in range(self.total_bins):
                f.write(",".join(["0.0000"] * n_cols) + "\n")
        print(f"Initialized results file: {results_file}")
        

    def parse_arguments(self):
        """Parse command line arguments"""
        parser = argparse.ArgumentParser(description='Batch fitting tool')
        parser.add_argument('--input', '-i', type=str, help='Input ROOT file')
        parser.add_argument('--output_dir', '-od', type=str, default='./', help='Output directory')
        parser.add_argument('--batch', action='store_true', help='Run batch fitting')
        parser.add_argument('--bins', type=str, help='Bins to fit')
        parser.add_argument('--vec_branches', '-vecBr', type=str,
                        help='Vector branch names to keep when splitting tree (comma-separated, e.g. "br1,br2" )')
        parser.add_argument('--binned', action='store_true', 
                          help='Perform binned fit (default: False)')
        
        args = parser.parse_args()

        self.tree_path = args.input
        if self.splitter:
            self.splitter.tree_path = args.input
            # Force re-detection of types because tree_path was likely valid now

        self.output_dir = args.output_dir
        self.binned_fit = args.binned
        
        vec_branches= None
        if args.vec_branches:
            vec_branches= [name.strip() for name in args.vec_branches.split(",")]

        if args.batch:
            bins_to_fit = None
            if args.bins:
                if ":" in args.bins:
                    start, end = map(int, args.bins.split(":"))
                    bins_to_fit = list(range(start, end + 1))
                else:
                    bins_to_fit = [int(x) for x in args.bins.split(",")]
            self.batch_fit(bins_to_fit,vec_branches)
        elif args.input:
            file = ROOT.TFile(args.input, "READ")
            tree = file.Get("event")
            self.fit_function(tree, args.output_dir, None, None, 
                            binned_fit=self.binned_fit)
                    
"""
v1.0.0
- Initial version, splitted from fit_tools.py v3.0.0
date : 2026-03-25
"""