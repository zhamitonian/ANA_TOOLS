import os, sys
from contextlib import contextmanager
from typing import Tuple, List, Optional, Callable, Dict, Any
import time
import numpy as np
import argparse
from abc import ABC, abstractmethod
from DRAW import style_draw
from PHY_CALCULATOR import PhysicsCalculator

import ROOT
from ROOT import RooFit as rf
from ROOT import RooStats

class FIT_UTILS():
    """
    Utility class for performing fits with ROOT and RooFit.
    Contains methods for creating combined datasets, reirecting output,
    
    fit_config : List of tuples (name, min, max) for datasets, default the first one is the var used to fit
    """
    def __init__(self, log_file = None, var_config = List[Tuple[str,float,float]] ):
        self.log_file = log_file
        self.var_config = var_config 

    @contextmanager
    def redirect_output(self):
        """
        Redirect both stdout and stderr to a log file at the file descriptor level.
        This captures output from both Python and C++/ROOT.
        """
        log_file = self.log_file

        if log_file is None:
            yield
            return
            
        # Save original file descriptors
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        
        # Save a copy of the original file descriptors
        saved_stdout = os.dup(original_stdout_fd)
        saved_stderr = os.dup(original_stderr_fd)
        
        try:
            # Open log file
            log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
            
            # Replace standard file descriptors with log file
            os.dup2(log_fd, original_stdout_fd)
            os.dup2(log_fd, original_stderr_fd)
            
            yield
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout, original_stdout_fd)
            os.dup2(saved_stderr, original_stderr_fd)
            
            # Close all duplicated descriptors
            os.close(saved_stdout)
            os.close(saved_stderr)
        os.close(log_fd)


    def handle_dataset(self, tree:ROOT.TTree, workspace:ROOT.RooWorkspace, branches_name:Optional[List[str]] = None, save_rootFile:bool = False) -> ROOT.RooDataSet:
        """
        Create corresponding RooRealVariable objects in the workspace , and create the RooDataSet
        and if branches_name are provided, create a combined dataset from the specified branches.
        
        Args:
            tree: ROOT TTree object
            workspace: RooWorkspace object
            branches_name: List of branch names to combine, e.g. ["phi1_M", "phi2_M"]
            save_rootFile: bool, whether to save the combined dataset to a ROOT file , 
                default is False, if True, the combined dataset will be saved to "combined.root",
                when not save ROOT file, the quick strategy is used to create the dataset.
        
        Returns:
            ROOT.RooDataSet: Combined dataset
        """
        var_configs = self.var_config
        
        # Get variables from the workspace
        arg_set = ROOT.RooArgSet()
        for config in var_configs:
            # Create RooRealVariable in the workspace with name same to the branch name
            workspace.factory(f"{config[0]}[{config[1]},{config[2]}]") 
            arg_set.add(workspace.var(config[0]))

        if branches_name is None:
            dataset = ROOT.RooDataSet("dataset", "dataset", tree, arg_set)
            return dataset

        if not save_rootFile:
            # Create empty dataset with the configured variables
            dataset = ROOT.RooDataSet("dataset", "Combined dataset",arg_set)
            fit_var = workspace.var(var_configs[0][0])  # The first variable is the one used for fitting
            other_vars = [workspace.var(config[0]) for config in var_configs[1:]]

            print(f"Combining branches: {branches_name}")
            
            # Iterate through each event in the tree
            total_entries = 0
            for i, event in enumerate(tree):
                if i % 10000 == 0:
                    print(f"Processing event {i}...")
                    
                # Get other variables' value from the event
                other_values = [getattr(event, var_config[0]) for var_config in var_configs[1:]]
                
                # For each specified branch, add its value as an entry to the dataset
                for branch_name in branches_name:
                    try:
                        # Get current branch value
                        branch_value = getattr(event, branch_name)
                        
                        # Check if value is within reasonable range
                        if var_configs[0][1] <= branch_value <= var_configs[0][2]:
                            # Set variable values
                            fit_var.setVal(branch_value)

                            for var, value in zip(other_vars, other_values):
                                var.setVal(value)
                            
                            # Add to dataset
                            dataset.add(arg_set)
                            total_entries += 1
                            
                    except AttributeError:
                        print(f"Warning: Branch {branch_name} not found in event {i}")
                        continue
        else:
            df = ROOT.RDataFrame(tree)
            temp_files = [f"temp_{i}.root" for i in range(len(branches_name))]
            for i,branch_name in enumerate(branches_name):
                filter_condition = f"{branch_name} >= {var_configs[0][1]} && {branch_name} <= {var_configs[0][2]}"
                df.Filter(filter_condition).Redefine(f'{var_configs[0][0]}',f'{branch_name}').Snapshot("event", temp_files[i])

            # Combine all filtered data using a TChain
            df = ROOT.RDataFrame("event", temp_files)
            df.Snapshot('event', 'combined.root')

            file = ROOT.TFile("combined.root", "READ")
            merged_tree = file.Get("event")

            dataset = ROOT.RooDataSet("dataset", "dataset", merged_tree, arg_set)
            total_entries = merged_tree.GetEntries()

            # Clean up temporary files
            for temp_file in temp_files:
                os.remove(temp_file)

        print(f"Combined dataset created with {total_entries} entries")
        print(f"Original tree had {tree.GetEntries()} events")
        print(f"Combined {len(branches_name)} branches per event")
        
        return dataset

class QUICK_FIT():
    """
    contains methods for performing batch fits on datasets

    the fit function should take the following parameters:
    #- should not have positional arguments, only keyword arguments
    - tree: ROOT.TTree object containing the data to fit
    - output_file: str, path to save the fit results
    - log_file: str, path to save the log output
    - range_use: tuple, (min, max) range for the fit
    - **kwargs: additional keyword arguments for flexibility
    output to be: result, nsig, nsig_err
    """

    def __init__(self,
                 fit_function: Callable, 
                 bin_var_config: Tuple[str, float, float, int], 
                 tree_path: str = "",
                 output_dir: str = ""):

        self.fit_function = fit_function
        self.bin_var_config = bin_var_config
        self.output_dir = output_dir
        self.tree_path = tree_path

    def batch_fit(self, bins_to_fit:Optional[List[int]] = None , 
                  branches_name:Optional[list[str]] = None,
                  additional_cut: str = ""):

        start_time = time.time()

        tree_path = self.tree_path
        output_dir = self.output_dir
        fit_function = self.fit_function

        os.makedirs(output_dir, exist_ok=True)

        bin_var, min_val, max_val, nbin = self.bin_var_config
        bin_step = (max_val - min_val) / nbin
        #Nbin_tot = int(round((max_val - min_val) / nbin))

        results = np.zeros((nbin, 5)) #[bin_center, bin_width/2, nsig, msig_err, nsig_err]

        # Initialize results file if it doesn't exist
        results_file = output_dir + "nsig_results.txt"
        if not os.path.isfile(results_file):
            with open(results_file, "w") as init_file:
                for i in range(nbin):
                    init_file.write("\n")
        else:
            try:
                loaded_results = np.loadtxt(results_file)
                if loaded_results.shape[0] >= nbin:
                    results = loaded_results[:nbin]
                    print(f"Loaded existing results from {results_file}")
            except Exception as e:
                print(f"Could not load existing results: {e}")

        # Use provided bins or default to all bins
        if bins_to_fit is None:
            bins_to_fit = list(range(nbin))

        # Check for valid bin numbers
        bins_to_fit = [i for i in bins_to_fit if 0 <= i < nbin]
        if not bins_to_fit:
            print("No valid bins specified. Exiting.")
            return
    
        print(f"Will process {len(bins_to_fit)} bins: {bins_to_fit}")

        #process each bin sequentially
        successful_fits = 0

        for bin_i in bins_to_fit:
            try:
                m_min = min_val + bin_i * bin_step
                m_max = min_val + (bin_i + 1) * bin_step
                range_cut = f"{bin_var} >= {m_min:.3f} && {bin_var} <= {m_max:.3f}"
                range_use = f"({m_min:.3f},{m_max:.3f})"
                
                print(f"Processing bin {bin_i}: {range_cut}")
                
                # Create output file paths
                bin_output = f"{output_dir}bin_{bin_i}" # output_dir/bin_i 
                bin_log_file = f"{output_dir}bin_{bin_i}.log"
                print(f"Output file: {bin_output}, Log file: {bin_log_file}")
                
                rf = ROOT.RDataFrame("event", tree_path)
                rf = rf.Filter(range_cut + additional_cut)
                #rf = rf.Filter(range_cut )

                temp_file_path = f"{output_dir}temp_bin_{bin_i}.root"
                print(temp_file_path)
                rf.Snapshot("event", temp_file_path)
                temp_file = ROOT.TFile.Open(temp_file_path)
                filtered_tree = temp_file.Get("event")
                print("-----------------------")

                # perform fit
                try:
                    # Call fit_function with optional branches_name parameter
                    result, nsig, nsig_err = fit_function(
                        filtered_tree, 
                        bin_output, 
                        bin_log_file, 
                        range_use,
                        branches_name = branches_name,
                    )
                    
                    results[bin_i] = [m_min + bin_step/2, bin_step/2, nsig, nsig_err, nsig_err]
                    
                    print(f"Bin {bin_i} fit complete: signal yield = {nsig:.2f} ± {nsig_err:.2f}")

                    # Convert status codes to a dictionary for better readability
                    status_codes = {
                        0: "successful fit",
                        1: "covariance was made positive definite",
                        2: "Hesse is invalid",
                        3: "EDM is above max",
                        4: "Reached call limit",
                        5: "other failure"
                    }
                    
                    fit_status = result.status()
                    if fit_status == 0:
                        print("Fit converged successfully!")
                        successful_fits += 1
                    else:
                        print(f"Fit had issues: {status_codes.get(fit_status, 'unknown error')}")
                    
                except Exception as e:
                    print(f"Error processing bin {bin_i}: {str(e)}")
                
                # Clean up memory
                filtered_tree.Delete()
                temp_file.Close()
                os.remove(temp_file_path) 
                
                    
            except Exception as e:
                print(f"Error setting up bin {bin_i}: {str(e)}")
        
        # Save all results to file
        with open(results_file, "w") as f:
            for i in range(nbin):
                f.write(f"{results[i][0]:.6f}  {results[i][1]:.6f}  {results[i][2]:.6f} {results[i][3]:.6f} {results[i][4]:.6f}\n")
        
        end_time = time.time()
        
        print(f"Batch fitting complete! Successfully fit {successful_fits}/{len(bins_to_fit)} bins.")
        print(f"Total time: {end_time - start_time:.1f} seconds")


    def parse_arguments(self):
        """
        """
        parser = argparse.ArgumentParser(description='general argument paeser for convenient fit')
        parser.add_argument('--input', type=str, help='Input ROOT file')
        parser.add_argument('--output_dir', type=str, default='./', help='Output dir')
        parser.add_argument('--batch', action='store_true', help='Run batch fitting')
        parser.add_argument('--bins', type=str, help='Bins to fit (comma-separated list or range "start:end")')
        parser.add_argument('--branches_name', type=str, 
                        help='Comma-separated list of branch names to combine (e.g., "phi1_M,phi2_M")')
        
        args = parser.parse_args()

        self.tree_path = args.input
        self.output_dir = args.output_dir

        # Parse branch names if provided
        branches_name = None
        if args.branches_name:
            branches_name = [name.strip() for name in args.branches_name.split(",")]
        print(f"Branch names to combine: {branches_name}")

        if args.batch:
            # Parse bins from command line arguments if provided
            bins_to_fit = None
            
            if args.bins:
                if ":" in args.bins:
                    start, end = map(int, args.bins.split(":"))
                    bins_to_fit = list(range(start, end + 1))
                else:
                    bins_to_fit = [int(x) for x in args.bins.split(",")]
                    
            self.batch_fit( bins_to_fit, branches_name)
            
        elif args.input:
            file = ROOT.TFile(args.input, "READ")
            tree = file.Get("event")  
            self.fit_function(tree, args.output_dir, None, None, branches_name = branches_name)
        else:
            print("Please provide an input file or use --batch mode")

        '''
        if args.batch:
            tools = PhysicsCalculator(output_dir=args.output_dir + "record.root")
            tools.set_bins(bin_num=[12], bin_boundaries=[0, 2*math.pi])
            h_nsig = tools.getNsigHist(args.output_dir + "nsig_results_splot.txt")
            tools.saveHist(h_nsig, "nsig")
            h_nsig.GetXaxis().SetTitle("#phi")
        style_draw([h_nsig], args.output_dir + "nsig.png")
        '''