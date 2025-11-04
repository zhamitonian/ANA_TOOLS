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


    def handle_dataset(self, tree:ROOT.TTree, workspace:ROOT.RooWorkspace, 
                       branches_name:Optional[List[str]] = None, 
                       save_rootFile:bool = False) -> ROOT.RooDataSet:
        """
        Create corresponding RooRealVariable objects in the workspace, and create the RooDataSet.
        If branches_name are provided, create a combined dataset from the specified branches.
        
        Args:
            tree: ROOT TTree object
            workspace: RooWorkspace object
            branches_name: List of branch names to combine (can be RVec types)
            save_rootFile: bool, whether to save the combined dataset to a ROOT file
        
        Returns:
            ROOT.RooDataSet: Combined dataset with flattened entries if needed
        """
        var_configs = self.var_config
        
        # Create RooRealVariables in workspace
        arg_set = ROOT.RooArgSet()
        for config in var_configs:
            workspace.factory(f"{config[0]}[{config[1]},{config[2]}]") 
            arg_set.add(workspace.var(config[0]))

        if branches_name is None:
            dataset = ROOT.RooDataSet("dataset", "dataset", tree, arg_set)
            return dataset

        # Create empty dataset
        dataset = ROOT.RooDataSet("dataset", "Combined dataset", arg_set)
        fit_var = workspace.var(var_configs[0][0])
        other_vars = [workspace.var(config[0]) for config in var_configs[1:]]

        print(f"Combining branches: {branches_name}")
        
        total_entries = 0
        var_min, var_max = var_configs[0][1], var_configs[0][2]

        # save file  not supported for vector branches yet
        if save_rootFile:
            df = ROOT.RDataFrame(tree)
            temp_files = [f"temp_{i}.root" for i in range(len(branches_name))]
            for i,branch_name in enumerate(branches_name):
                filter_condition = f"{branch_name} >= {var_min} && {branch_name} <= {var_max}"
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
            return dataset
        
        # Helper function to add entry to dataset
        def add_entry(value):
            nonlocal total_entries
            if var_min <= value <= var_max:
                fit_var.setVal(value)
                for var, other_val in zip(other_vars, other_values):
                    var.setVal(other_val)
                dataset.add(arg_set)
                total_entries += 1
        
        # Iterate through tree
        for i, event in enumerate(tree):
            if i % 10000 == 0:
                print(f"Processing event {i}...")
            
            # Get other variables' values (common for both vector and scalar)
            other_values = [getattr(event, var_config[0]) for var_config in var_configs[1:]]
            
            # Check if branches are vectors (RVec or std::vector)
            first_branch = getattr(event, branches_name[0])
            is_vector = hasattr(first_branch, '__len__') and not isinstance(first_branch, str)
            
            if is_vector:
                # Handle vector branches: flatten by iterating over all branches and their elements
                for branch_name in branches_name:
                    try:
                        branch_vec = getattr(event, branch_name)
                        # Iterate over each element in the vector
                        for value in branch_vec:
                            add_entry(value)
                    except AttributeError:
                        print(f"Warning: Branch {branch_name} not found in event {i}")
                        continue
            else:
                # Handle scalar branches
                for branch_name in branches_name:
                    try:
                        branch_value = getattr(event, branch_name)
                        add_entry(branch_value)
                    except AttributeError:
                        print(f"Warning: Branch {branch_name} not found in event {i}")
                        continue

        print(f"Combined dataset created with {total_entries} entries")
        print(f"Original tree had {tree.GetEntries()} events")
        
        return dataset

class QUICK_FIT():
    """
    Contains methods for performing batch fits on datasets with multi-dimensional binning support.

    The fit function should take the following parameters:
    - tree: ROOT.TTree object containing the data to fit
    - output_file: str, path to save the fit results
    - log_file: str, path to save the log output
    - range_use: str or tuple, range description for the fit
    - **kwargs: additional keyword arguments for flexibility
    
    Output: result, nsig, nsig_err
    """

    def __init__(self,
                 fit_function: Callable, 
                 bin_var_config: Optional[List[Tuple[str, float, float, int]]] = None,
                 tree_path: str = "",
                 output_dir: str = ""):
        """
        Args:
            fit_function: Callable fit function
            bin_var_config: List of tuples (var_name, min, max, nbins) for multi-dimensional binning
                           If single tuple provided, will be converted to list for backward compatibility
            tree_path: Path to input ROOT file
            output_dir: Output directory for fit results
        """
        self.fit_function = fit_function
        
        # Support backward compatibility: convert single tuple to list
        if bin_var_config is not None:
            if isinstance(bin_var_config, tuple) and len(bin_var_config) == 4:
                self.bin_var_configs = [bin_var_config]
            else:
                self.bin_var_configs = bin_var_config
        else:
            self.bin_var_configs = []
            
        self.tree_path = tree_path
        self.output_dir = output_dir
        
        # Calculate total number of bins for multi-dimensional case
        self.n_dimensions = len(self.bin_var_configs)
        self.total_bins = 1
        self.bins_per_dim = []
        
        for config in self.bin_var_configs:
            n_bins = config[3]
            self.bins_per_dim.append(n_bins)
            self.total_bins *= n_bins
            
        print(f"Initialized {self.n_dimensions}D binning with {self.total_bins} total bins")
        if self.n_dimensions > 1:
            print(f"Bins per dimension: {self.bins_per_dim}")

    def _get_bin_indices(self, flat_index: int) -> List[int]:
        """
        Convert flat bin index to multi-dimensional bin indices
        
        Args:
            flat_index: Flat index in range [0, total_bins)
            
        Returns:
            List of indices for each dimension
        """
        indices = []
        remaining = flat_index
        
        for i in range(self.n_dimensions - 1, -1, -1):
            n_bins = self.bins_per_dim[i]
            indices.insert(0, remaining % n_bins)
            remaining //= n_bins
            
        return indices

    def _get_flat_index(self, indices: List[int]) -> int:
        """
        Convert multi-dimensional bin indices to flat index
        
        Args:
            indices: List of indices for each dimension
            
        Returns:
            Flat index
        """
        flat_index = 0
        multiplier = 1
        
        for i in range(self.n_dimensions - 1, -1, -1):
            flat_index += indices[i] * multiplier
            multiplier *= self.bins_per_dim[i]
            
        return flat_index

    def _get_bin_ranges(self, bin_indices: List[int]) -> List[Tuple[float, float]]:
        """
        Get the range for each dimension given bin indices
        
        Args:
            bin_indices: List of bin indices for each dimension
            
        Returns:
            List of (min, max) tuples for each dimension
        """
        ranges = []
        
        for dim_idx, bin_idx in enumerate(bin_indices):
            var_name, min_val, max_val, n_bins = self.bin_var_configs[dim_idx]
            bin_step = (max_val - min_val) / n_bins
            
            bin_min = min_val + bin_idx * bin_step
            bin_max = min_val + (bin_idx + 1) * bin_step
            
            ranges.append((bin_min, bin_max))
            
        return ranges

    def _format_bin_description(self, bin_indices: List[int], ranges: List[Tuple[float, float]]) -> str:
        """
        Create a readable description of the bin
        
        Args:
            bin_indices: List of bin indices
            ranges: List of (min, max) tuples
            
        Returns:
            String description
        """
        descriptions = []
        
        for dim_idx, (bin_min, bin_max) in enumerate(ranges):
            var_name = self.bin_var_configs[dim_idx][0]
            descriptions.append(f"{var_name}[{bin_min:.3f},{bin_max:.3f}]")
            
        return "_".join(descriptions)

    def batch_fit(self, bins_to_fit: Optional[List[int]] = None,
                  branches_name: Optional[List[str]] = None,
                  additional_cut: str = ""):
        """
        Perform batch fitting with multi-dimensional binning support
        
        Args:
            bins_to_fit: List of flat bin indices to fit. If None, fit all bins
            branches_name: List of branch names to combine for fitting
            additional_cut: Additional filter condition to apply
        """
        start_time = time.time()

        tree_path = self.tree_path
        output_dir = self.output_dir
        fit_function = self.fit_function

        os.makedirs(output_dir, exist_ok=True)

        # Initialize results array: [bin_center_dim1, bin_center_dim2, ..., nsig, nsig_err, nsig_err]
        results_columns = 2 * self.n_dimensions + 3  # centers + widths + nsig + 2*err
        results = np.zeros((self.total_bins, results_columns))

        # Initialize results file
        results_file = output_dir + "nsig_results.txt"
        if not os.path.isfile(results_file):
            with open(results_file, "w") as init_file:
                for i in range(self.total_bins):
                    init_file.write("\n")
        else:
            try:
                loaded_results = np.loadtxt(results_file)
                if loaded_results.shape[0] >= self.total_bins:
                    results = loaded_results[:self.total_bins]
                    print(f"Loaded existing results from {results_file}")
            except Exception as e:
                print(f"Could not load existing results: {e}")

        # Determine which bins to fit
        if bins_to_fit is None:
            bins_to_fit = list(range(self.total_bins))

        bins_to_fit = [i for i in bins_to_fit if 0 <= i < self.total_bins]
        if not bins_to_fit:
            print("No valid bins specified. Exiting.")
            return

        print(f"Will process {len(bins_to_fit)} bins out of {self.total_bins} total bins")

        # Detect variable types for all dimensions
        test_df = ROOT.RDataFrame("event", tree_path)
        bin_var_types = []
        
        for dim_idx, config in enumerate(self.bin_var_configs):
            var_name = config[0]
            try:
                column_type = test_df.GetColumnType(var_name)
                is_vector = "vector" in column_type.lower() or "rvec" in column_type.lower()
                bin_var_types.append(is_vector)
                
                type_str = "vector" if is_vector else "scalar"
                print(f"Dimension {dim_idx}: '{var_name}' is {type_str} type")
                
            except Exception as e:
                print(f"Warning: Could not determine type of '{var_name}': {e}")
                bin_var_types.append(False)

        successful_fits = 0
        failed_bins = []

        # Process each bin
        for flat_bin_idx in bins_to_fit:
            try:
                # Convert flat index to multi-dimensional indices
                bin_indices = self._get_bin_indices(flat_bin_idx)
                ranges = self._get_bin_ranges(bin_indices)
                
                # Create bin description
                bin_desc = self._format_bin_description(bin_indices, ranges)
                range_use = "(" + ",".join([f"[{r[0]:.3f},{r[1]:.3f}]" for r in ranges]) + ")"
                
                print("-----------------------")
                print(f"Processing bin {flat_bin_idx}: {bin_desc}")
                print(f"Bin indices: {bin_indices}")
                
                bin_output = f"{output_dir}bin_{flat_bin_idx}"
                bin_log_file = f"{output_dir}bin_{flat_bin_idx}.log"
                print(f"Log file: {bin_log_file}")
                
                # Create RDataFrame and apply cuts
                rf = ROOT.RDataFrame("event", tree_path)
                
                # apply extra cut, if any
                if additional_cut:
                    rf = rf.Filter(additional_cut, "Additional cut")

                # Apply cuts for all dimensions
                bin_conditions = []
                
                for dim_idx, (bin_min, bin_max) in enumerate(ranges):
                    var_name = self.bin_var_configs[dim_idx][0]
                    is_vector = bin_var_types[dim_idx]
                    
                    condition = f"({var_name} >= {bin_min:.3f}) && ({var_name} <= {bin_max:.3f})"
                    bin_conditions.append(condition)
                    
                    if is_vector:
                        # For vector type: keep events with at least one element in range
                        rf = rf.Filter(f"Any({condition})", f"Dim {dim_idx} has elements in range")
                    else:
                        # For scalar type: filter events directly
                        rf = rf.Filter(condition, f"Dim {dim_idx} in range")

                # Apply element filtering for vector branches
                # All vector bin variables should use the same indexing logic
                has_vector = any(bin_var_types)
                
                if has_vector:
                    # Create combined condition for vector indexing
                    # Use AND logic: element must be in range for ALL vector dimensions
                    vector_conditions = [cond for dim_idx, cond in enumerate(bin_conditions) 
                                        if bin_var_types[dim_idx]]
                    combined_condition = " && ".join(vector_conditions)
                    
                    print(f"Combined vector filter: {combined_condition}")
                    
                    # apply selection to provided branches as well
                    if branches_name:
                        for branch_name in branches_name:
                            try:
                                branch_type = rf.GetColumnType(branch_name)
                                if "vector" in branch_type.lower() or "rvec" in branch_type.lower():
                                    rf = rf.Redefine(branch_name, f"{branch_name}[{combined_condition}]")
                                    print(f"Filtered vector branch: {branch_name}")
                            except Exception as e:
                                print(f"Warning: Could not filter branch {branch_name}: {e}")
                    
                    """
                    # not neccessary to filter vector bin variables again
                    # Apply to all vector bin variables
                    for dim_idx, config in enumerate(self.bin_var_configs):
                        if bin_var_types[dim_idx]:
                            var_name = config[0]
                            rf = rf.Redefine(var_name, f"{var_name}[{combined_condition}]")
                            print(f"Filtered vector bin variable: {var_name}")
                    """

                # Save filtered tree
                temp_file_path = f"{output_dir}temp_bin_{flat_bin_idx}.root"
                rf.Snapshot("event", temp_file_path)
                
                # Open the filtered tree
                temp_file = ROOT.TFile.Open(temp_file_path)
                filtered_tree = temp_file.Get("event")

                # check if there are entries to fit
                n_entries = filtered_tree.GetEntries()
                print(f"Filtered tree has {n_entries} entries")
                
                if n_entries == 0:
                    print(f"Warning: No entries in bin {flat_bin_idx}, skipping fit")
                    failed_bins.append(flat_bin_idx)
                    temp_file.Close()
                    os.remove(temp_file_path)
                    continue

                # Perform fit
                try:
                    result, nsig, nsig_err = fit_function(
                        filtered_tree,
                        bin_output,
                        bin_log_file,
                        range_use,
                        branches_name=branches_name,
                    )
                    
                    # Store results: [bin_centers, bin_widths, nsig, nsig_err, nsig_err]
                    result_row = []
                    
                    for dim_idx, (bin_min, bin_max) in enumerate(ranges):
                        bin_center = (bin_min + bin_max) / 2
                        bin_width = (bin_max - bin_min) / 2
                        result_row.extend([bin_center, bin_width])
                    
                    result_row.extend([nsig, nsig_err, nsig_err])
                    results[flat_bin_idx] = result_row
                    
                    print(f"Signal yield: {nsig:.2f} ± {nsig_err:.2f}")

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
                    print(f"Error during fit for bin {flat_bin_idx}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                    failed_bins.append(flat_bin_idx)
                
                # Clean up
                filtered_tree.Delete()
                temp_file.Close()
                os.remove(temp_file_path)
                
            except Exception as e:
                print(f"Error setting up bin {flat_bin_idx}: {str(e)}")
                import traceback
                traceback.print_exc()
                failed_bins.append(flat_bin_idx)

        # Save results
        with open(results_file, "w") as f:
            # Write header
            header_parts = []
            for dim_idx in range(self.n_dimensions):
                var_name = self.bin_var_configs[dim_idx][0]
                header_parts.extend([f"{var_name}_center", f"{var_name}_width"])
            header_parts.extend(["nsig", "nsig_err", "nsig_err"])
            f.write("# " + "  ".join(header_parts) + "\n")
            
            # Write data
            for i in range(self.total_bins):
                row_str = "  ".join([f"{val:.6f}" for val in results[i]])
                f.write(row_str + "\n")

        end_time = time.time()
        
        print("-----------------------")
        print(f"Batch fitting complete! Successfully fit {successful_fits}/{len(bins_to_fit)} bins.")
        if failed_bins:
            print(f"Failed bins: {failed_bins}")
        print(f"Total time: {end_time - start_time:.1f} seconds")

    def parse_arguments(self):
        """Parse command line arguments for multi-dimensional binning"""
        parser = argparse.ArgumentParser(description='General argument parser for convenient fit')
        parser.add_argument('--input', '-i', type=str, help='Input ROOT file')
        parser.add_argument('--output_dir', '-od', type=str, default='./', help='Output directory')
        parser.add_argument('--batch', action='store_true', help='Run batch fitting')
        parser.add_argument('--bins', type=str, 
                          help='Bins to fit. For 1D: "0,1,2" or "0:10". For 2D+: flat indices "0,1,2" or "0:100"')
        parser.add_argument('--branches_name', '-BrN', type=str,
                        help='Comma-separated list of branch names to combine (e.g., "phi1_M,phi2_M")')
        
        args = parser.parse_args()

        self.tree_path = args.input
        self.output_dir = args.output_dir

        branches_name = None
        if args.branches_name:
            branches_name = [name.strip() for name in args.branches_name.split(",")]
        print(f"Branch names to combine: {branches_name}")

        if args.batch:
            bins_to_fit = None
            
            if args.bins:
                if ":" in args.bins:
                    start, end = map(int, args.bins.split(":"))
                    bins_to_fit = list(range(start, end + 1))
                else:
                    bins_to_fit = [int(x) for x in args.bins.split(",")]
                    
            self.batch_fit(bins_to_fit, branches_name)
            
        elif args.input:
            file = ROOT.TFile(args.input, "READ")
            tree = file.Get("event")
            self.fit_function(tree, args.output_dir, None, None, branches_name=branches_name)
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


    ### below part not finished yet
    def perform_joint_fit(self, 
                     workspace1: ROOT.RooWorkspace, 
                     workspace2: ROOT.RooWorkspace,
                     joint_params: List[str] = ["nsig"],
                     output_file: str = "", 
                     log_file: str = "") -> Tuple[ROOT.RooFitResult, float, float]:
        """
        Perform joint fitting using RooSimultaneous with shared parameters
    
        Args:
            workspace1: First workspace containing model and dataset
            workspace2: Second workspace containing model and dataset
            joint_params: List of parameter names to share between models
            output_file: Path to save fit results
            log_file: Path to save log output
    
        Returns:
            Tuple of (fit_result, nsig, nsig_error)
        """
    
        # Create a new workspace for joint fitting
        joint_workspace = ROOT.RooWorkspace("joint_ws", "Joint fitting workspace")
    
        print(f"=== Starting Joint Fit ===")
        print(f"Shared parameters: {joint_params}")
    
        # Step 1: Import everything from both workspaces with renaming
        print("Importing models and datasets...")
    
        # Import from workspace1 with suffix "_var1"
        dataset1 = workspace1.data("dataset")
        model1 = workspace1.pdf("model")
        joint_workspace.import_(dataset1, ROOT.RooFit.RenameAllNodes("_var1"))
        joint_workspace.import_(model1, ROOT.RooFit.RenameAllNodes("_var1"))
    
        # Import from workspace2 with suffix "_var2"  
        dataset2 = workspace2.data("dataset")
        model2 = workspace2.pdf("model")
        joint_workspace.import_(dataset2, ROOT.RooFit.RenameAllNodes("_var2"))
        joint_workspace.import_(model2, ROOT.RooFit.RenameAllNodes("_var2"))
    
        print(f"Dataset 1 entries: {dataset1.numEntries()}")
        print(f"Dataset 2 entries: {dataset2.numEntries()}")
    
        # Step 2: Create shared parameters and rebuild models
        self._create_shared_parameters(joint_workspace, workspace1, workspace2, joint_params)
    
        # Step 3: Create category variable for RooSimultaneous
        joint_workspace.factory("category[var1,var2]")
        category = joint_workspace.cat("category")
    
        # Step 4: Create RooSimultaneous model
        sim_model = ROOT.RooSimultaneous("sim_model", "Simultaneous model", category)
    
        # Add the modified models to simultaneous PDF
        model1_modified = joint_workspace.pdf("model_var1_shared")
        model2_modified = joint_workspace.pdf("model_var2_shared")
    
        if not model1_modified:
            model1_modified = joint_workspace.pdf("model_var1")
        if not model2_modified:
            model2_modified = joint_workspace.pdf("model_var2")
    
        sim_model.addPdf(model1_modified, "var1")
        sim_model.addPdf(model2_modified, "var2")
        joint_workspace.import_(sim_model)
    
        # Step 5: Create combined dataset with category information
        combined_dataset = ROOT.RooDataSet("joint dataset", "joint dataset",
                                           Index()) 
        self._create_simultaneous_dataset(
            joint_workspace, "dataset_var1", "dataset_var2", category
        )
        joint_workspace.import_(combined_dataset)
    
        # Step 6: Perform simultaneous fit
        fit_utils = FIT_UTILS(log_file=log_file, var_config=[])
    
        with fit_utils.redirect_output():
            print("Performing simultaneous fit...")
        
            sim_model = joint_workspace.pdf("sim_model")
            combined_data = joint_workspace.data("combined_dataset")
        
            joint_result = sim_model.fitTo(combined_data,
                                     ROOT.RooFit.Save(),
                                     ROOT.RooFit.Extended(True),
                                     ROOT.RooFit.PrintLevel(0),
                                     ROOT.RooFit.NumCPU(4))
        
            # Extract shared signal yield
            nsig_joint, nsig_err_joint = self._extract_shared_signal_yield(joint_workspace, joint_params)
        
            print(f"Joint fit signal yield: {nsig_joint:.2f} ± {nsig_err_joint:.2f}")
    
        # Save results if output file specified
        if output_file:
            joint_workspace.writeToFile(output_file + "_joint_workspace.root")
        
            result_file = ROOT.TFile(output_file + "_joint_fitresult.root", "RECREATE")
            joint_result.Write()
            result_file.Close()
        
            print(f"Joint fit results saved to {output_file}_joint_*.root")
    
        print(f"=== Joint Fit Complete ===")
    
        return joint_result, nsig_joint, nsig_err_joint

    def _create_shared_parameters(self, joint_workspace: ROOT.RooWorkspace, 
                            workspace1: ROOT.RooWorkspace, workspace2: ROOT.RooWorkspace,
                            joint_params: List[str]):
        """
        Create shared parameters and rebuild models to use them
        """
        print(f"Creating shared parameters: {joint_params}")
    
        for param_name in joint_params:
            param1 = workspace1.var(param_name)
            param2 = workspace2.var(param_name)
        
            if param1 and param2:
                # Create shared parameter with average initial value
                param_val = (param1.getVal() + param2.getVal()) / 2
                param_min = min(param1.getMin(), param2.getMin())
                param_max = max(param1.getMax(), param2.getMax())
            
                shared_param_name = f"{param_name}_shared"
                joint_workspace.factory(f"{shared_param_name}[{param_val}, {param_min}, {param_max}]")
            
                print(f"Created shared parameter: {shared_param_name} = {param_val}")
            
                # Rebuild models to use shared parameter
                self._rebuild_model_with_shared_param(joint_workspace, "model_var1", param_name, shared_param_name)
                self._rebuild_model_with_shared_param(joint_workspace, "model_var2", param_name, shared_param_name)

    def _rebuild_model_with_shared_param(self, workspace: ROOT.RooWorkspace, 
                                   model_name: str, old_param_name: str, new_param_name: str):
        """
        Rebuild model to use shared parameter
        """
        model = workspace.pdf(model_name)
        if not model:
            print(f"Warning: Model {model_name} not found")
            return
    
        # Get the original model components
        if hasattr(model, 'coefList') and hasattr(model, 'pdfList'):
            # This is a RooAddPdf (SUM model)
            pdf_list = model.pdfList()
            coef_list = model.coefList()
        
            # Create new coefficient list with shared parameter
            new_coef_list = ROOT.RooArgList()
            coef_iter = coef_list.createIterator()
            coef = coef_iter.Next()
            while coef:
                if coef.GetName() == f"{old_param_name}_var1" or coef.GetName() == f"{old_param_name}_var2":
                    # Replace with shared parameter
                    shared_param = workspace.var(new_param_name)
                    if shared_param:
                        new_coef_list.add(shared_param)
                    else:
                        new_coef_list.add(coef)
                else:
                    new_coef_list.add(coef)
                coef = coef_iter.Next()
        
            # Create new model with shared parameters
            new_model = ROOT.RooAddPdf(f"{model_name}_shared", f"{model_name} with shared params", 
                                 pdf_list, new_coef_list)
            workspace.import_(new_model)
        
            print(f"Rebuilt model {model_name} with shared parameter {new_param_name}")

    def _create_simultaneous_dataset(self, workspace: ROOT.RooWorkspace, 
                               dataset1_name: str, dataset2_name: str,
                               category: ROOT.RooCategory) -> ROOT.RooDataSet:
        """
        Create combined dataset with category variable for RooSimultaneous
        """
        dataset1 = workspace.data(dataset1_name)
        dataset2 = workspace.data(dataset2_name)
    
        if not dataset1 or not dataset2:
            raise ValueError(f"Datasets {dataset1_name} or {dataset2_name} not found")
    
        # Get all variables from both datasets
        vars1 = dataset1.get()
        vars2 = dataset2.get()
    
        # Create combined variable set with category
        combined_vars = ROOT.RooArgSet(category)
    
        # Add variables from first dataset
        var_iter1 = vars1.createIterator()
        var = var_iter1.Next()
        while var:
            combined_vars.add(var)
            var = var_iter1.Next()
    
        # Add variables from second dataset (if not already present)
        var_iter2 = vars2.createIterator()
        var = var_iter2.Next()
        while var:
            if not combined_vars.find(var.GetName()):
                combined_vars.add(var)
            var = var_iter2.Next()
    
        combined_dataset = ROOT.RooDataSet("combined_dataset", "Combined dataset", combined_vars)
    
        print("Creating combined dataset with category information...")
    
        # Add entries from first dataset with category "var1"
        category.setLabel("var1")
        for i in range(dataset1.numEntries()):
            dataset1.get(i)
            combined_dataset.add(combined_vars)
            if i % 10000 == 0:
                print(f"Added entry {i}/{dataset1.numEntries()} from dataset1")
    
        # Add entries from second dataset with category "var2"
        category.setLabel("var2")
        for i in range(dataset2.numEntries()):
            dataset2.get(i)
            combined_dataset.add(combined_vars)
            if i % 10000 == 0:
                print(f"Added entry {i}/{dataset2.numEntries()} from dataset2")
    
        total_entries = dataset1.numEntries() + dataset2.numEntries()
        print(f"Combined dataset created with {combined_dataset.numEntries()} entries (expected: {total_entries})")
    
        return combined_dataset

    def _extract_shared_signal_yield(self, workspace: ROOT.RooWorkspace, 
                               joint_params: List[str]) -> Tuple[float, float]:
        """
        Extract shared signal yield from workspace
        """
        # Look for shared signal parameter
        for param_name in joint_params:
            if "nsig" in param_name or "signal" in param_name:
                shared_param = workspace.var(f"{param_name}_shared")
                if shared_param:
                    return shared_param.getVal(), shared_param.getError()
    
        # Fallback: look for any shared parameter
        for param_name in joint_params:
            shared_param = workspace.var(f"{param_name}_shared")
            if shared_param:
                return shared_param.getVal(), shared_param.getError()
    
        print("Warning: No shared signal yield parameter found")
        return 0.0, 0.0