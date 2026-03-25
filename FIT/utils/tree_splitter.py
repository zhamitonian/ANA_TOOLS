import os
import ROOT
import numpy as np
from typing import List, Tuple, Optional, Any, Dict

class TreeSplitter:
    """
    A standalone utility for splitting and processing ROOT Trees based on binning configurations.
    Designed to be reused across different analysis tools.
    """

    def __init__(self, tree_path: str, bin_var_configs: Dict[str, Any]):
        """
        Initialize the processor.

        Args:
            tree_path: Path to the input ROOT file/tree.
            bin_var_configs: Configuration for binning variables.
                             Dict of variable name to binning config, two formats supported:
                             - Uniform binning: (min, max, nbins)  
                             - Custom binning : [bin_boundaries] (length = nbins + 1)
        """
        self.tree_path = tree_path
        self.bin_var_configs = bin_var_configs # Dict[str, tuple/list]
        self.var_names = list(bin_var_configs.keys()) # Keep order
        self.n_dimensions = len(self.var_names)
        
        # Calculate bins per dimension and pre-process configs
        self.bins_per_dim = []
        self.bin_boundaries = [] # List of boundary arrays/tuples
        
        for var in self.var_names:
            cfg = bin_var_configs[var]
            if isinstance(cfg, (list, np.ndarray)):
                # Custom binning: [edge0, edge1, ..., edgeN]
                boundaries = np.array(cfg, dtype=float)
                self.bins_per_dim.append(len(boundaries) - 1)
                self.bin_boundaries.append(boundaries)
            elif isinstance(cfg, tuple) and len(cfg) == 3:
                # Uniform binning: (min, max, nbins)
                min_val, max_val, nbins = cfg
                self.bins_per_dim.append(nbins)
                self.bin_boundaries.append(np.linspace(min_val, max_val, nbins + 1))
            else:
                raise ValueError(f"Invalid binning config for {var}: {cfg}")

        self.total_bins = int(np.prod(self.bins_per_dim))

    def _detect_variable_types(self) -> List[bool]:
        """Check if binning variables are scalar or vector types."""
        if not os.path.exists(self.tree_path):
             # Try assuming tree_path string is "file.root"
             return [False] * self.n_dimensions

        df = ROOT.RDataFrame("event", self.tree_path)
        types = []
        for var_name in self.var_names:
            try:
                col_type = df.GetColumnType(var_name)
                is_vector = "vector" in col_type.lower() or "rvec" in col_type.lower()
                types.append(is_vector)
            except Exception:
                # If column not found or error, assume scalar (safest fallback)
                types.append(False)
        return types

    def _get_bin_indices(self, flat_idx: int) -> List[int]:
        """Convert flat bin index to multi-dimensional indices."""
        indices = []
        remaining = flat_idx
        for n_bins in reversed(self.bins_per_dim):
            indices.append(remaining % n_bins)
            remaining //= n_bins
        return indices[::-1]

    def _get_bin_ranges(self, indices: List[int]) -> List[Tuple[float, float]]:
        """Get the (min, max) range for each dimension based on bin indices."""
        ranges = []
        for dim, idx in enumerate(indices):
            boundaries = self.bin_boundaries[dim]
            
            # Use boundaries array directly
            bin_min = boundaries[idx]
            bin_max = boundaries[idx + 1]
                
            ranges.append((bin_min, bin_max))
        return ranges

    def create_bin_snapshot(self, flat_bin_idx: int, output_file: str, 
                          vec_br_to_keep: Optional[List[str]] = None,
                          additional_cut: str = "") -> bool:
        """
        Creates a snapshot (ROOT file) for a specific bin by filtering the original tree.

        Args:
           flat_bin_idx: The flat index of the bin to process.
           output_file: Path to save the snapshot ROOT file.
           vec_br_to_keep: List of vector branch names to handle specifically,
                             Calculated/filtered branches will be saved.
           additional_cut: Extra string-based cut to apply to the whole tree.

        Returns:
            True if snapshot created successfully with entries, False otherwise.
        """
        
        # Detect variable types (scalar vs vector)
        self.bin_var_types = self._detect_variable_types()

        # Convert flat index to ranges
        bin_indices = self._get_bin_indices(flat_bin_idx)
        ranges = self._get_bin_ranges(bin_indices)
        
        try:
            rf = ROOT.RDataFrame("event", self.tree_path)
            # 1. Apply Global Extra Cut
            if additional_cut:
                rf = rf.Filter(additional_cut, "Additional cut")
            
            # 2. Build Conditions for Each Dimension
            bin_conditions = []
            for dim_idx, (bin_min, bin_max) in enumerate(ranges):
                var_name = self.var_names[dim_idx]
                is_vector = self.bin_var_types[dim_idx]
                
                # Condition: bin_min <= var <= bin_max
                condition = f"({var_name} >= {bin_min:.5f}) && ({var_name} <= {bin_max:.5f})"
                bin_conditions.append(condition)
                print(condition)
                if is_vector:
                    # For vector binning var: Keep event if ANY element matches
                    rf = rf.Filter(f"Any({condition})", f"Dim {dim_idx} has elements in range")
                else:
                    # For scalar binning var: Keep event if scalar matches
                    rf = rf.Filter(condition, f"Dim {dim_idx} in range")
            
            # 3. Handle Vector Branch Filtering (Compaction)
            # If any binning variable is a vector, we likely need to filter *other* vector branches 
            # to only keep elements corresponding to the valid bin
            has_vector_binning = any(self.bin_var_types)
            if has_vector_binning:
                # Combine conditions for all vector dimensions using AND logic
                vector_conditions = [cond for dim_idx, cond in enumerate(bin_conditions) 
                                     if self.bin_var_types[dim_idx]]
                
                if not vector_conditions:
                     # Should theoretically not happen if has_vector_binning is True
                     combined_condition = "1" 
                else:
                    combined_condition = " && ".join(vector_conditions)
                
                # Define a mask column first to ensure consistency across redefinitions
                rf = rf.Define("binning_mask", combined_condition)

                # Collect branches that need filtering
                target_branches = (vec_br_to_keep or [])
                
                for branch_name in target_branches:
                    try:
                        branch_type = rf.GetColumnType(branch_name)
                        is_vec_branch = "vector" in branch_type.lower() or "rvec" in branch_type.lower()
                        
                        if is_vec_branch:
                            # Strategy: Redefine the branch to only contain elements passing the condition
                            # Note: This changes the branch content in the snapshot!
                            rf = rf.Redefine(branch_name, f"{branch_name}[binning_mask]")
                            
                            # Check if any elements remain after filtering
                            rf = rf.Filter(f"{branch_name}.size() > 0") 
                            
                    except Exception as e:
                        print(f"Warning: Could not filter branch {branch_name}: {e}")

            # Save to file
            print(f"Creating snapshot for bin {flat_bin_idx}...")
            snapshot_opts = ROOT.RDF.RSnapshotOptions()
            snapshot_opts.fMode = "RECREATE"
            rf.Snapshot("event", output_file, "", snapshot_opts)
            n_entries = ROOT.RDataFrame("event", output_file).Count().GetValue()
            print(f"Splitted tree has {n_entries} entries")

            return True

        except Exception as e:
            print(f"Error creating snapshot for bin {flat_bin_idx}: {e}")
            if os.path.exists(output_file):
                os.remove(output_file)
            return False

