import os, sys
from contextlib import contextmanager
from typing import Tuple, List, Optional, Union
import ROOT

"""
FIT_IO utility for handling fit datasets and redirecting output
version : 1.0.0
Date    : 2026-03-25
Author  : wangzheng
"""

class FIT_IO():
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
        
        # Flush Python buffers before redirecting
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Save original file descriptors
        original_stdout_fd = sys.stdout.fileno()
        original_stderr_fd = sys.stderr.fileno()
        
        # Save a copy of the original file descriptors
        saved_stdout = os.dup(original_stdout_fd)
        saved_stderr = os.dup(original_stderr_fd)
        
        # Open log file
        log_fd = os.open(log_file, os.O_WRONLY | os.O_CREAT | os.O_TRUNC)
        
        try:
            # Replace standard file descriptors with log file
            os.dup2(log_fd, original_stdout_fd)
            os.dup2(log_fd, original_stderr_fd)
            
            # Close log file descriptor (no longer needed, fds are duplicated)
            os.close(log_fd)
            
            yield
            
            # Final flush before restoring
            sys.stdout.flush()
            sys.stderr.flush()
        finally:
            # Restore original file descriptors
            os.dup2(saved_stdout, original_stdout_fd)
            os.dup2(saved_stderr, original_stderr_fd)
            
            # Close saved descriptors
            os.close(saved_stdout)
            os.close(saved_stderr)


    def handle_dataset(self, input_tree: ROOT.TTree, 
                   workspace: ROOT.RooWorkspace, 
                   target_brs: Optional[List[str]] = None,  # to handle vector branches
                   binned_fit: bool = False,
                   hist_bins: int = 100,
                   weight_branch : Optional[str] = None) -> Union[ROOT.RooDataSet, ROOT.RooDataHist]:
        """
        Create RooDataSet (unbinned) or RooDataHist (binned) from TTree
        
        Args:
            input_tree: ROOT TTree object
            workspace: RooWorkspace object
            target_brs: Create dataset from these branches; if None, use var_config[0]
            binned_fit: If True, create RooDataHist; if False, create RooDataSet
            hist_bins: Number of bins for histogram (only for binned mode)
            weight_branch: Optional branch name for event weights
        
        Returns:
            ROOT.RooDataSet (unbinned) or ROOT.RooDataHist (binned)
        """
        if input_tree is None:
            raise ValueError("input_tree must be provided")
        var_configs = self.var_config
        
        # Create RooRealVariables in workspace
        arg_set = ROOT.RooArgSet()
        for config in var_configs:
            workspace.factory(f"{config[0]}[{config[1]},{config[2]}]")
            arg_set.add(workspace.var(config[0]))
        if weight_branch is not None:
            workspace.factory(f"{weight_branch}[0, 1000000]")
            weight_var = workspace.var(weight_branch)
            arg_set.add(weight_var)
        
        fit_var = workspace.var(var_configs[0][0])
        var_min, var_max = var_configs[0][1], var_configs[0][2]

        # === UNBINNED MODE: Create RooDataSet ===
        if not binned_fit:
            print(f"Creating RooDataSet (unbinned) from TTree...")
            
            if target_brs is None:
                if weight_branch is not None:
                    dataset = ROOT.RooDataSet("dataset", "dataset", input_tree, arg_set, "", weight_branch)
                else:
                    dataset = ROOT.RooDataSet("dataset", "dataset", input_tree, arg_set)
                print(f"Created RooDataSet with {dataset.numEntries()} entries")
                return dataset

            # Create empty dataset for combining branches
            # Add event_idx and cand_idx to track original tree structure
            workspace.factory("event_idx[0, 1000000000]")  # Original event number
            workspace.factory("cand_idx[0, 100]")  # Candidate index within event
            arg_set.add(workspace.var("event_idx"))
            arg_set.add(workspace.var("cand_idx"))
            
            if weight_branch is not None:
                dataset = ROOT.RooDataSet("dataset", "Combined dataset", arg_set, ROOT.RooFit.WeightVar(weight_var))
            else:
                dataset = ROOT.RooDataSet("dataset", "Combined dataset", arg_set)
            other_vars = [workspace.var(config[0]) for config in var_configs[1:]]
            event_idx_var = workspace.var("event_idx")
            cand_idx_var = workspace.var("cand_idx")

            print(f"Combining branches: {target_brs} into RooDataSet...")
            total_entries = 0

            # Helper function to add entry to dataset
            def add_entry(value, evt_idx, cnd_idx, weight=None):
                nonlocal total_entries
                if var_min <= value <= var_max:
                    fit_var.setVal(value)
                    for var, other_val in zip(other_vars, other_values):
                        var.setVal(other_val)
                    event_idx_var.setVal(evt_idx)
                    cand_idx_var.setVal(cnd_idx)
                    if weight_branch is not None and weight is not None:
                        weight_var.setVal(weight)
                        dataset.add(arg_set, float(weight))
                    else:
                        dataset.add(arg_set)
                    total_entries += 1
            
            # Iterate through tree
            for i, event in enumerate(input_tree):
                if i % 10000 == 0:
                    print(f"Processing event {i}...")

                # Get other variables' values
                other_values = [getattr(event, var_config[0]) for var_config in var_configs[1:]]

                # Prepare weight(s) if needed
                if weight_branch is not None:
                    weight_val = getattr(event, weight_branch, None)
                else:
                    weight_val = None

                # Check if branches are vectors
                first_branch = getattr(event, target_brs[0])
                is_vector = hasattr(first_branch, '__len__') and not isinstance(first_branch, str)

                if is_vector:
                    # Handle vector branches
                    for branch_name in target_brs:
                        try:
                            branch_vec = getattr(event, branch_name)
                            # If weight is also a vector, match by index; else use scalar for all
                            if weight_branch is not None and weight_val is not None and hasattr(weight_val, '__len__') and not isinstance(weight_val, str):
                                for cand_idx, value in enumerate(branch_vec):
                                    w = weight_val[cand_idx] if cand_idx < len(weight_val) else 1.0
                                    add_entry(value, i, cand_idx, w)
                            else:
                                for cand_idx, value in enumerate(branch_vec):
                                    add_entry(value, i, cand_idx, weight_val)
                        except AttributeError:
                            print(f"Warning: Branch {branch_name} not found in event {i}")
                            continue
                else:
                    # Handle scalar branches
                    for branch_name in target_brs:
                        try:
                            branch_value = getattr(event, branch_name)
                            add_entry(branch_value, i, 0, weight_val)  # Scalar branch uses cand_idx=0
                        except AttributeError:
                            print(f"Warning: Branch {branch_name} not found in event {i}")
                            continue

            print(f"Combined dataset created with {total_entries} entries")
            vars = dataset.get()
            for var in vars:
                print(var.GetName())
            return dataset

        # === BINNED MODE: Create RooDataHist ===
        else:
            print(f"Creating RooDataHist (binned) from TTree...")
            
            df = ROOT.RDataFrame(input_tree)

            # Create histogram
            hist_name = "hist_temp"
            hist_model = ROOT.RDF.TH1DModel(hist_name, hist_name, 
                                           hist_bins, var_min, var_max)

            # Get variable name string for RDataFrame
            fit_var_name = var_configs[0][0]

            if  target_brs is None or len(target_brs) == 0:
                if weight_branch is None:
                    histogram = df.Histo1D(hist_model, fit_var_name).GetValue()
                else:
                    histogram = df.Histo1D(hist_model, fit_var_name, weight_branch).GetValue()
                print(f"hist: { histogram.GetEntries()}")
            else:
                hists = []
                for branch in target_brs:
                    if weight_branch is None:
                        hists.append(df.Histo1D(hist_model, branch).GetValue())
                    else:
                        hists.append(df.Histo1D(hist_model, branch, weight_branch).GetValue())
                # Sum histograms
                histogram = hists[0]
                for h in hists[1:]:
                    histogram.Add(h)

            n_entries = histogram.GetEntries()
            print(f"Histogram created with {n_entries} entries")
            
            if n_entries == 0:
                print("Warning: Empty histogram created")
            
            # Create RooDataHist from histogram
            # For binned mode, only use the fit variable (not weight) in arg_set
            # Weight is already incorporated in the histogram bin contents
            arg_set_binned = ROOT.RooArgSet(fit_var)
            datahist = ROOT.RooDataHist("datahist", "Binned dataset", arg_set_binned, histogram)
            print(f"Created RooDataHist with {datahist.sumEntries()} entries")
            print(f"Number of bins: {datahist.numEntries()}")
            
            return datahist


"""
v1.0.0
- Initial version, splitted from fit_tools.py v3.0.0
date : 2026-03-25
"""
