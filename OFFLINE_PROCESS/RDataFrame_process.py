import ROOT 
from ROOT import TCanvas
import os
from typing import List, Optional, Tuple, Callable
import numpy as np
import random

class RDF_process:
    _defined_functions = set()

    def __init__(self):
        pass

    def set_CMS_variables(self, df:ROOT.RDataFrame, FSPs:List[str], particles:List[str], prefix:Optional[str] ="ee", var2save:Optional[List[str]]=None, useBeamVar:Optional[bool]=False)->ROOT.RDataFrame:
        """
        Calculate kinematic variables in the Center of Mass System (CMS) frame.
        Similar to CalculateCMSVariables function in C++.
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe containing particle 4-momenta
        
        FSPs : List[str]
            name prefix of Final State Particles 

        particles : List[str]
            name prefix of particles which we want to calculate the CMS variables for
            
        var2save : List[str]
            variables among [E_cms, px_cms, py_cms, pz_cms, p_cms, pt_cms, theta_cms, phi_cms] to be save
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with CMS variables
        """ 
        new_df = df

        # Define system 4 momentum
        if not useBeamVar:
            if f"{prefix}_lab_E" not in new_df.GetColumnNames():
                for p in ["px", "py", "pz", "E"]:
                    new_df = new_df.Define(f"{prefix}_lab_{p}", " + ".join([f"{fsp}_{p}" for fsp in FSPs]))
        
        # Define the boost function in the ROOT interpreter
        func_name = "boost_particle"
        if func_name not in self._defined_functions:
            ROOT.gInterpreter.Declare("""
                std::vector<double> boost_particle(double px, double py, double pz, double E, 
                                                double sys_px, double sys_py, double sys_pz, double sys_E) {
                    // Create the boost vector and apply the boost
                    TLorentzVector sys_p4(sys_px, sys_py, sys_pz, sys_E);
                    TVector3 boost = sys_p4.BoostVector();

                    // Create a TLorentzVector for the particle
                    TLorentzVector particleP4;
                    particleP4.SetPxPyPzE(px, py, pz, E);
                    particleP4.Boost(-boost);
                    
                    // Extract CMS variables
                    double E_cms = particleP4.E();
                    double px_cms = particleP4.Px();
                    double py_cms = particleP4.Py();
                    double pz_cms = particleP4.Pz();
                    double p_cms = particleP4.P();
                    double pt_cms = particleP4.Pt();
                    double theta_cms = particleP4.Theta();
                    double phi_cms = particleP4.Phi();
                    
                    return std::vector<double>{E_cms, px_cms, py_cms, pz_cms, p_cms, pt_cms, theta_cms, phi_cms};
                }
            """)
            RDF_process._defined_functions.add(func_name)
        

        all_vars = ["E", "px", "py", "pz", "p", "pt", "theta", "phi"]
        if var2save is None:
            var2save = all_vars
        
        var_indices = {
            "E": 0, 
            "px": 1, 
            "py": 2, 
            "pz": 3, 
            "p": 4, 
            "pt": 5, 
            "theta": 6, 
            "phi": 7
        }

        for particle in particles:
            for var in var2save:
                if var in var_indices and f"{particle}_{prefix}_cms_{var}" not in new_df.GetColumnNames():
                    idx = var_indices[var]
                    if useBeamVar:
                        new_df = new_df.Define(
                            f"{particle}_{prefix}_cms_{var}", 
                            f"boost_particle({particle}_px, {particle}_py, {particle}_pz, {particle}_E, beamPx, beamPy, beamPz, beamE)[{idx}]"
                        )
                    else:
                        new_df = new_df.Define(
                            f"{particle}_{prefix}_cms_{var}", 
                            f"boost_particle({particle}_px, {particle}_py, {particle}_pz, {particle}_E, {prefix}_lab_px, {prefix}_lab_py, {prefix}_lab_pz, {prefix}_lab_E)[{idx}]"
                        )
        
        return new_df


    def calculate_PHI(self, df: ROOT.RDataFrame, particle_pair: Tuple[str, str], pxy_branch: Tuple[str, str] = ("px", "py"),
                      branch_style:str="{name}_ee_cms_{comp}", output_branch:str = None) -> ROOT.RDataFrame:
        """
        Calculate the dot product of vectors x and y where:
        - x = pt1 + pt2
        - y = (pt1 - pt2)
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        particle_pair : tuple[str, str]
            Tuple containing the names of the two particles
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with the calculated PHI value
        """
        new_df = df
        
        # Define the C++ function for vector calculation if not already defined
        func_name = "calculate_vector_dot_product"
        if func_name not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                double calculate_vector_dot_product(
                    double pt1_px, double pt1_py,
                    double pt2_px, double pt2_py
                ) {
                    // Create TVector2 objects for the two particles' transverse momenta
                    TVector2 pt1(pt1_px, pt1_py);
                    TVector2 pt2(pt2_px, pt2_py);
                    
                    TVector2 x = pt1 + pt2;
                    TVector2 y = (pt1 - pt2) * 0.5;
                    
                    double dot_product = x * y / (x.Mod() * y.Mod());  

                    double cross = x.X() * y.Y() - x.Y() * y.X(); 
                    double phi;
                    
                    if(cross >=0 )
                    {
                        phi = acos(dot_product);
                    }
                    else
                    {
                        phi = 2*3.1415926 - acos(dot_product) ;
                    }
                    return phi;
                }
            """)
            RDF_process._defined_functions.add(func_name)
        
        p1, p2 = particle_pair
        p1_px = branch_style.format(name=p1, comp=pxy_branch[0])
        p1_py = branch_style.format(name=p1, comp=pxy_branch[1])
        p2_px = branch_style.format(name=p2, comp=pxy_branch[0])
        p2_py = branch_style.format(name=p2, comp=pxy_branch[1])
        if output_branch is None:
            output_branch = f"PHI_{p1}_{p2}" 

        if output_branch in new_df.GetColumnNames():
            print(f"Warning: Branch '{output_branch}' already exists. It will be overwritten.")
            new_df = new_df.Redefine(
                output_branch,
                f"calculate_vector_dot_product({p1_px}, {p1_py}, {p2_px}, {p2_py})"
            )
        else:
            new_df = new_df.Define(
                output_branch,
                f"calculate_vector_dot_product({p1_px}, {p1_py}, {p2_px}, {p2_py})"
            ) 
        return new_df

    def calculate_pt_diff(self, df: ROOT.RDataFrame, particle_pair: Tuple[str, str]) -> ROOT.RDataFrame:
        """
        Calculate the magnitude of the difference between two vectors' transverse momenta.
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        particle_pair : tuple[str, str]
            Tuple containing the names of the two particles
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with the calculated pt_diff value
        """
        new_df = df
        
        # Define the C++ function for pt difference calculation if not already defined
        func_name = "calculate_pt_diff"
        if func_name not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                double calculate_pt_diff(
                    double pt1_px, double pt1_py,
                    double pt2_px, double pt2_py
                ) {
                    // Create TVector2 objects for the two particles' transverse momenta
                    TVector2 pt1(pt1_px, pt1_py);
                    TVector2 pt2(pt2_px, pt2_py);
                    
                    // Calculate the difference vector
                    TVector2 pt_diff = pt1 - pt2;
                    
                    // Return the magnitude of the difference
                    return pt_diff.Mod();
                }
            """)
            RDF_process._defined_functions.add(func_name)
        
        p1, p2 = particle_pair
        
        new_df = new_df.Define(
            f"pt_diff_{p1}_{p2}",
            f"calculate_pt_diff({p1}_ee_cms_px, {p1}_ee_cms_py, {p2}_ee_cms_px, {p2}_ee_cms_py)"
        )

        return new_df

    def select_best_candidate(self, input_df:ROOT.RDataFrame, var:str, Multi_Rank:bool = False)->ROOT.RDataFrame:
        """
        Optimized function to remove duplicate events, keeping events with minimal value 
        for the given variable. If multiple events have the same minimal value, keep them all.
        Designed to handle million-level events efficiently.
        
        Parameters:
            input_df: ROOT.RDataFrame - Input RDataFrame
            var: str - Name of the variable to be used for selection (minimization)
            
        Returns:
            ROOT.RDataFrame - New RDataFrame containing the best candidates
        """
        # Add entry index for tracking
        if "__entry__" not in input_df.GetColumnNames():
            input_df = input_df.Define("__entry__", "rdfentry_")
        else:
            input_df = input_df.Redefine("__entry__", "rdfentry_")
        
        # Get data using AsNumpy for efficient processing
        #print(f"Extracting data for optimization...")
        data = input_df.AsNumpy(columns=["__experiment__", "__run__", "__event__", var, "__entry__"])
        
        total_entries = len(data["__experiment__"])
        #print(f"Processing {total_entries} entries for best candidate selection...")
        
        # Use dictionary to track minimum values and corresponding entries
        event_candidates = {}
        
        # Single pass through data to find minimum values and collect all entries with that minimum
        for i in range(total_entries):
            # Encode (exp, run, evt) into a single integer, consistent with build_response_matrix.make_event_id
            # Bit layout (from LSB):
            #   0-35 : evt  (36 bits)
            #   36-55: run  (20 bits)
            #   56-63: exp  (8 bits)
            exp_i = int(data["__experiment__"][i])
            run_i = int(data["__run__"][i])
            evt_i = int(data["__event__"][i])
            event_key = (exp_i << 56) | (run_i << 36) | evt_i

            value = data[var][i]
            entry_idx = data["__entry__"][i]
            
            if event_key not in event_candidates:
                # First occurrence of this event
                event_candidates[event_key] = (value, [entry_idx])
            else:
                current_min, current_entries = event_candidates[event_key]
                
                if value < current_min:
                    # Found a better candidate, replace
                    event_candidates[event_key] = (value, [entry_idx])
                elif value == current_min:
                    if Multi_Rank:
                        current_entries.append(entry_idx)
                    # If Multi_Rank=False, we keep only the first one (no action needed)
        
        # Collect all best entries (including ties)
        best_entries = []
        for min_value, entry_list in event_candidates.values():
            best_entries.extend(entry_list)
        
        #print(f"Selected {len(best_entries)} best candidates from {total_entries} entries")
        #print(f"Number of unique events: {len(event_candidates)}")
        
        # Create efficient C++ function for filtering
        func_name = "IsSelectedEntryOptimized"
        if func_name not in RDF_process._defined_functions:
            # Create a sorted vector for binary search efficiency
            sorted_entries = sorted(best_entries)
            entries_str = "{" + ", ".join(map(str, sorted_entries)) + "}"
            
            ROOT.gInterpreter.Declare(f"""
                #include <vector>
                #include <algorithm>
                
                std::vector<Long64_t> g_selected_entries = {entries_str};
                
                void UpdateSelectedEntries(const std::vector<Long64_t>& new_entries) {{
                    g_selected_entries = new_entries;
                    std::sort(g_selected_entries.begin(), g_selected_entries.end());
                }}
                
                bool IsSelectedEntryOptimized(Long64_t entry) {{
                    return std::binary_search(g_selected_entries.begin(), g_selected_entries.end(), entry);
                }}
            """)
            RDF_process._defined_functions.add(func_name)
        else:
            # Update the existing global vector with new entries using the update function
            sorted_entries = sorted(best_entries)
            
            # Create a temporary vector in ROOT and use the update function
            entries_cpp_list = ", ".join(map(str, sorted_entries))
            ROOT.gInterpreter.ProcessLine(f"""
            {{
                std::vector<Long64_t> temp_entries = {{{entries_cpp_list}}};
                UpdateSelectedEntries(temp_entries);
            }}
            """)
        
        # Apply filter
        result_df = input_df.Filter("IsSelectedEntryOptimized(__entry__)")
        if Multi_Rank is False:
            result_df = result_df.Redefine("__candidate__", "1").Redefine("__ncandidates__", "1")
        
        return result_df

    def quick_reweight(self, mc_df: ROOT.RDataFrame, hist_config:Optional[Tuple[str,int,float,float]]=None, data_df:Optional[ROOT.RDataFrame]=None, 
                       h_data:Optional[ROOT.TH1]=None, h_MC:Optional[ROOT.TH1]=None, MC_weight:Optional[str] =None,  simple_Scale: Optional[bool] = True) -> Tuple[ROOT.RDataFrame, ROOT.TH1]:
        """
        Reweight MC to match data distribution using a simple bin-by-bin ratio.
        
        Parameters:
        -----------
        mc_df : ROOT.RDataFrame
            MC dataframe to be reweighted
        hist_config : Optional[Tuple[str, int, float, float]]
            Tuple containing (variable_name, number_of_bins, min_value, max_value)
        data_df : Optional[ROOT.RDataFrame]
            Data dataframe for reference distribution 
        h_data : Optional[ROOT.TH1]
            Data histogram (alternative to providing data_df) , the hist's name should be the variable name
        h_MC : Optional[ROOT.TH1]
            MC histogram (alternative to providing mc_df) , the hist's name should be the variable name
        MC_weight : Optional[str]
            Optional weight column in MC dataframe to be applied when calculating MC histogram
        simple_Scale: bool
            Whether to normalize histograms before calculating weights
            
        Returns:
        --------
        ROOT.RDataFrame: Weighted MC dataframe
        ROOT.TH1: Weight histogram
        """
        if (data_df is None and h_data is None) or (data_df is not None and h_data is not None):
            raise ValueError("Either provide data_df with hist_config OR provide h_data")
        
        if data_df is not None and hist_config is None:
            raise ValueError("When providing data_df, hist_config must also be provided")
        
        if hist_config:
            var, bin, xmin, xmax = hist_config
            h_data = data_df.Histo1D((f"h_data_{var}", f"Data {var}", bin, xmin, xmax), var)
        else:
            var, bin, xmin ,xmax = h_data.GetName(), h_data.GetNbinsX(), h_data.GetXaxis().GetXmin(), h_data.GetXaxis().GetXmax()

        # 支持外部传入 h_MC
        if h_MC is not None:
            h_mc_ptr = h_MC
        else:
            if MC_weight is not None:
                h_mc = mc_df.Histo1D((f"h_mc_{var}", f"MC {var}", bin, xmin, xmax), var, MC_weight)
            else:
                h_mc = mc_df.Histo1D((f"h_mc_{var}", f"MC {var}", bin, xmin, xmax), var)
            h_mc_ptr = h_mc.GetPtr()
        h_data_ptr = h_data
        
        if simple_Scale:
            h_data_ptr.Scale(1.0 / h_data_ptr.Integral())
            h_mc_ptr.Scale(1.0 / h_mc_ptr.Integral())
        
        hist_weight = h_data_ptr.Clone(f"h_weight_{var}")
        weights = []
        for i in range(1, bin + 1):
            data_content = h_data_ptr.GetBinContent(i)
            mc_content = h_mc_ptr.GetBinContent(i)
            weight = data_content / mc_content if mc_content > 0 else 1.0
            print(f"Bin {i}: Data = {data_content}, MC = {mc_content}, Weight = {weight}")
            weights.append(weight)
            hist_weight.SetBinContent(i, weight)
        
        weight_array = np.array(weights, dtype=np.float64)

        func_name = "get_bin_weight"
        if func_name not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare(f"""
                #include <vector>
                #include <ROOT/RVec.hxx>

                double get_bin_weight(double value, double xmin, double xmax, int nbins) {{
                    if (value < xmin || value >= xmax) return 1.0;
                    
                    double bin_width = (xmax - xmin) / nbins;
                    int bin_idx = (int)((value - xmin) / bin_width);
                    
                    static std::vector<double> weights = {{{', '.join(map(str, weights))}}};
                    
                    if (bin_idx >= 0 && bin_idx < weights.size()) {{
                        return weights[bin_idx];
                    }}
                    return 1.0;
                }}

                template <typename T>
                ROOT::VecOps::RVec<double> get_bin_weight(const ROOT::VecOps::RVec<T>& values, double xmin, double xmax, int nbins) {{
                    ROOT::VecOps::RVec<double> result;
                    result.reserve(values.size());
                    for (const auto& v : values) {{
                        result.emplace_back(get_bin_weight(static_cast<double>(v), xmin, xmax, nbins));
                    }}
                    return result;}}
            """)
            RDF_process._defined_functions.add(func_name)
        
        df_weighted = mc_df.Define("data_mc_weight", f"get_bin_weight({var}, {xmin}, {xmax}, {bin})")
        
        return df_weighted, hist_weight

    def calculate_HelicityAngle(self, df: ROOT.RDataFrame, 
                comp_branches: Tuple[str, str, str, str], 
                daughter_branches: Tuple[str, str, str, str], 
                particle_names: Tuple[str, str]) -> ROOT.RDataFrame:
        """
        Calculate helicity angle: In the rest frame of the composite particle, 
        the angle between the daughter particle momentum and the composite particle 
        momentum direction in the CMS frame.
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        comp_branches : Tuple[str, str, str, str]
            Composite particle momentum branches (px, py, pz, E) in CMS frame
        daughter_branches : Tuple[str, str, str, str]
            Daughter particle momentum branches (px, py, pz, E) in CMS frame
        particle_names : Tuple[str, str]
            Names for (composite_particle, daughter_particle)
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with helicity angle
        """
        comp_name, daughter_name = particle_names
        
        # Define scalar version
        func_name_scalar = "calculate_helicity_angle"
        if func_name_scalar not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <TVector3.h>
                #include <TLorentzVector.h>
                
                double calculate_helicity_angle_scalar(
                    double comp_px, double comp_py, double comp_pz, double comp_E,
                    double daughter_px, double daughter_py, double daughter_pz, double daughter_E) 
                {
                    // Create 4-vectors for composite and daughter particles in CMS frame
                    TLorentzVector comp_cms(comp_px, comp_py, comp_pz, comp_E);
                    TLorentzVector daughter_cms(daughter_px, daughter_py, daughter_pz, daughter_E);
                    
                    // Boost daughter to composite rest frame
                    TVector3 boost_vec = comp_cms.BoostVector();
                    TLorentzVector daughter_rest = daughter_cms;
                    daughter_rest.Boost(-boost_vec);
                    
                    // Get momentum direction of composite in CMS frame
                    TVector3 comp_dir = comp_cms.Vect().Unit();
                    
                    // Get momentum direction of daughter in composite rest frame
                    TVector3 daughter_dir = daughter_rest.Vect().Unit();
                    
                    // Calculate cosine of helicity angle
                    double cos_helicity = comp_dir.Dot(daughter_dir);
                    
                    return cos_helicity;
                }
            """)
            RDF_process._defined_functions.add(func_name_scalar)

            new_df = df.Define(
            f"cos_helicity_{comp_name}_{daughter_name}",
            f"calculate_helicity_angle({comp_branches[0]}, {comp_branches[1]}, {comp_branches[2]}, {comp_branches[3]}, "
            f"{daughter_branches[0]}, {daughter_branches[1]}, {daughter_branches[2]}, {comp_branches[3]} )"
        )

        return new_df


    def select_best_candidate_memory_efficient(self, input_df:ROOT.RDataFrame, var:str)->ROOT.RDataFrame:
        """
        Memory-efficient version for extremely large datasets (10M+ events).
        Uses ROOT's built-in functionality to minimize memory usage.
        
        Parameters:
            input_df: ROOT.RDataFrame - Input RDataFrame
            var: str - Name of the variable to be used for selection (minimization)
            
        Returns:
            ROOT.RDataFrame - New RDataFrame containing the best candidates
        """
        print("Using memory-efficient algorithm for very large datasets...")

        # Ensure entry index column exists
        if "__entry__" not in input_df.GetColumnNames():
            df_with_key = input_df.Define("__entry__", "rdfentry_")
        else:
            df_with_key = input_df.Redefine("__entry__", "rdfentry_")

        # Define C++ helper for efficient processing entirely on the C++ side
        func_name = "ProcessBestCandidates"
        if func_name not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <unordered_map>
                #include <unordered_set>
                #include <vector>
                #include <cstdint>

                class BestCandidateProcessor {
                private:
                    // Key: packed (exp, run, evt) using 64-bit integer
                    std::unordered_map<std::uint64_t, std::pair<double, std::vector<Long64_t>>> event_map;

                    static std::uint64_t make_key(int exp, Long64_t run, Long64_t evt) {
                        std::uint64_t exp_u = static_cast<std::uint64_t>(exp);
                        std::uint64_t run_u = static_cast<std::uint64_t>(run);
                        std::uint64_t evt_u = static_cast<std::uint64_t>(evt);
                        return (exp_u << 56) | (run_u << 36) | evt_u;
                    }

                public:
                    void ProcessEntry(int exp, Long64_t run, Long64_t evt, double value, Long64_t entry) {
                        std::uint64_t key = make_key(exp, run, evt);
                        auto it = event_map.find(key);

                        if (it == event_map.end()) {
                            event_map.emplace(key, std::make_pair(value, std::vector<Long64_t>{entry}));
                        } else {
                            double current_min = it->second.first;

                            if (value < current_min) {
                                it->second.first = value;
                                it->second.second.clear();
                                it->second.second.push_back(entry);
                            } else if (value == current_min) {
                                it->second.second.push_back(entry);
                            }
                        }
                    }

                    void FillBestEntrySet(std::unordered_set<Long64_t>& dest) const {
                        for (const auto& kv : event_map) {
                            const auto& entries = kv.second.second;
                            dest.insert(entries.begin(), entries.end());
                        }
                    }

                    size_t GetUniqueEventCount() const {
                        return event_map.size();
                    }
                };

                // Global processor and best-entry set
                BestCandidateProcessor g_processor;
                std::unordered_set<Long64_t> g_best_entries;

                void ResetProcessor() {
                    g_processor = BestCandidateProcessor();
                    g_best_entries.clear();
                }

                void ProcessBestCandidates(int exp, Long64_t run, Long64_t evt, double value, Long64_t entry) {
                    g_processor.ProcessEntry(exp, run, evt, value, entry);
                }

                void FinalizeBestEntries() {
                    g_best_entries.clear();
                    g_processor.FillBestEntrySet(g_best_entries);
                }

                size_t GetUniqueEventCount() {
                    return g_processor.GetUniqueEventCount();
                }

                size_t GetBestEntryCount() {
                    return g_best_entries.size();
                }

                bool IsSelectedEntryMemEff(Long64_t entry) {
                    return g_best_entries.find(entry) != g_best_entries.end();
                }
            """)
            RDF_process._defined_functions.add(func_name)

        # Reset the processor state
        ROOT.ResetProcessor()

        # Extract needed columns once into NumPy arrays and feed them to
        # the C++ processor. This avoids Python-side dictionaries and
        # huge C++ initializers while staying robust for O(10M) entries.
        print("Processing entries to find best candidates...")

        data = df_with_key.AsNumpy(columns=["__experiment__", "__run__", "__event__", var, "__entry__"])
        n_entries = len(data["__experiment__"])

        for i in range(n_entries):
            ROOT.ProcessBestCandidates(
                int(data["__experiment__"][i]),
                int(data["__run__"][i]),
                int(data["__event__"][i]),
                float(data[var][i]),
                int(data["__entry__"][i]),
            )

        # Build the best-entry set once on the C++ side
        ROOT.FinalizeBestEntries()
        unique_count = ROOT.GetUniqueEventCount()
        selected_count = ROOT.GetBestEntryCount()

        print(f"Selected {selected_count} best candidates")
        print(f"Number of unique events: {unique_count}")

        # Apply filter using C++ membership test
        result_df = df_with_key.Filter("IsSelectedEntryMemEff(__entry__)")
        result_df = result_df.Redefine("__candidate__", "1").Redefine("__ncandidates__", "1")

        return result_df

    def calculate_pt_toAxis(self, df: ROOT.RDataFrame, particle:Tuple[str, str, str], axis: Tuple[str, str], 
                            particle_name: str, axis_name:str) -> ROOT.RDataFrame:
        """
        Calculate pt and costheta relative to an axis, such as thrust axis.
        Automatically handles both scalar and vector inputs.
        Output branches: {particle_name}_{axis_name}_pt, {particle_name}_{axis_name}_costheta

        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        particle : Tuple[str, str, str]
            Tuple of particle momentum component branch names (px, py, pz)
        axis : Tuple[str, str]
            Tuple of axis direction branch names (theta, phi)
        particle_name : str
            Name of the particle for naming the output branch
        axis_name : str
            Name of the axis for naming the output branch
        """

        # Define scalar version
        func_name_scalar = "calculate_pt_toAxis_scalar"
        if func_name_scalar not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <tuple>
                std::tuple<double, double> calculate_pt_toAxis_scalar(
                    double particle_px, double particle_py, double particle_pz,
                    double axis_theta, double axis_phi) 
                {
                    TVector3 particle_vec(particle_px, particle_py, particle_pz);
                    TVector3 axis_vec;
                    axis_vec.SetMagThetaPhi(1.0, axis_theta, axis_phi);
                    
                    double pt_toAxis = particle_vec.Perp(axis_vec);
                    double costheta = particle_vec.Dot(axis_vec) / particle_vec.Mag();
                            
                    return std::make_tuple(pt_toAxis, costheta);
                }
            """)
            RDF_process._defined_functions.add(func_name_scalar)

        # Define vector version
        func_name_vec = "calculate_pt_toAxis_vec"
        if func_name_vec not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <ROOT/RVec.hxx>
                #include <tuple>
                using namespace ROOT::VecOps;
                
                std::tuple<RVec<double>, RVec<double>> calculate_pt_toAxis_vec(
                    const RVec<double>& particle_px, 
                    const RVec<double>& particle_py, 
                    const RVec<double>& particle_pz,
                    double axis_theta, double axis_phi) 
                {
                    TVector3 axis_vec;
                    axis_vec.SetMagThetaPhi(1.0, axis_theta, axis_phi);
                    
                    RVec<double> pt_results;
                    RVec<double> costheta_results;
                    pt_results.reserve(particle_px.size());
                    costheta_results.reserve(particle_px.size());
                    
                    for (size_t i = 0; i < particle_px.size(); ++i) {
                        TVector3 particle_vec(particle_px[i], particle_py[i], particle_pz[i]);
                        double pt_toAxis = particle_vec.Perp(axis_vec);
                        double costheta = particle_vec.Dot(axis_vec) / particle_vec.Mag();
                        pt_results.push_back(pt_toAxis);
                        costheta_results.push_back(costheta);
                    }
                    
                    return std::make_tuple(pt_results, costheta_results);
                }
            """)
            RDF_process._defined_functions.add(func_name_vec)

        # Define automatic dispatcher
        func_name_auto = "calculate_pt_toAxis_auto"
        if func_name_auto not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <ROOT/RVec.hxx>
                using namespace ROOT::VecOps;
                
                // Overload for scalar inputs
                auto calculate_pt_toAxis_auto(
                    double particle_px, double particle_py, double particle_pz,
                    double axis_theta, double axis_phi) 
                {
                    return calculate_pt_toAxis_scalar(particle_px, particle_py, particle_pz, 
                                                      axis_theta, axis_phi);
                }
                
                // Overload for vector inputs
                auto calculate_pt_toAxis_auto(
                    const RVec<double>& particle_px, 
                    const RVec<double>& particle_py, 
                    const RVec<double>& particle_pz,
                    double axis_theta, double axis_phi) 
                {
                    return calculate_pt_toAxis_vec(particle_px, particle_py, particle_pz,
                                                  axis_theta, axis_phi);
                }
            """)
            RDF_process._defined_functions.add(func_name_auto)

        # Use the automatic dispatcher to calculate both pt and costheta
        new_df = df.Define(
            f"__{particle_name}_{axis_name}_tuple",
            f"calculate_pt_toAxis_auto({particle[0]}, {particle[1]}, {particle[2]}, {axis[0]}, {axis[1]})"
        )
        
        # Extract pt and costheta from the tuple
        new_df = new_df.Define(
            f"{particle_name}_{axis_name}_pt",
            f"std::get<0>(__{particle_name}_{axis_name}_tuple)"
        ).Define(
            f"{particle_name}_{axis_name}_costheta",
            f"std::get<1>(__{particle_name}_{axis_name}_tuple)"
        )

        return new_df

    def save_all_pairs(self, df: ROOT.RDataFrame, 
                        p1_branches: Tuple[str, str, str],
                        p2_branches: Tuple[str, str, str],
                        mass:Tuple[float, float],
                        particle_name:Optional[Tuple[str, str, str]]=None,
                        cross_mode:bool= True) -> ROOT.RDataFrame:
        """
        Save all combinations of two particle from vector<double> branches.
    
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe containing vector<double> branches for p1 and p2
        p1_branches : Tuple[str, str, str]
            Tuple of p1 momentum component branch names [px, py, pz]
        p2_branches : Tuple[str, str, str]
            Tuple of p2 momentum component branch names [px, py, pz]
        mass : Tuple[float, float]
            Mass of two particles in GeV/c^2 (mass_p1, mass_p2)
        particle_name : Optional[Tuple[str, str, str]]
            Names for (composite_particle, particle1, particle2)
        cross_mode : bool
            If True, perform full cross combinations (i,j loop)
            If False, only pair same indices (i=j)
        
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with combination variables as vectors
        """
        new_df = df
    
        # Define different C++ functions for two modes
        if cross_mode:
            func_name = "calculate_all_pairs_cross"
            if func_name not in self._defined_functions:
                ROOT.gInterpreter.Declare(f"""
                    #include <ROOT/RVec.hxx>
                    #include <TLorentzVector.h>
                    using namespace ROOT::VecOps;
                
                    std::tuple<RVec<double>, RVec<double>, RVec<double>, RVec<double>, RVec<double>, RVec<double>, RVec<int>, RVec<int>>
                    calculate_all_pairs_cross(
                        const RVec<double>& p1_px, const RVec<double>& p1_py, const RVec<double>& p1_pz,
                        const RVec<double>& p2_px, const RVec<double>& p2_py, const RVec<double>& p2_pz,
                        double mass_p1, double mass_p2) 
                    {{
                        RVec<double> E;
                        RVec<double> px;
                        RVec<double> py;
                        RVec<double> pz;
                        RVec<double> helicity_angles;
                        RVec<double> plane_angles;
                        RVec<int> p1_index;
                        RVec<int> p2_index;
                    
                        // Full cross combination: Loop over all p1 and p2 candidates
                        for (size_t i = 0; i < p1_px.size(); ++i) {{
                            TLorentzVector p1;
                            p1.SetXYZM(p1_px[i], p1_py[i], p1_pz[i], mass_p1);
                        
                            // Loop over all p2 candidates (full cross product)
                            for (size_t j = 0; j < p2_px.size(); ++j) {{
                                TLorentzVector p2;
                                p2.SetXYZM(p2_px[j], p2_py[j], p2_pz[j], mass_p2);
                            
                                // Form composite particle candidate
                                TLorentzVector comp_p = p1 + p2;

                                // Calculate helicity angle
                                TVector3 comp_dir = comp_p.Vect().Unit();
                                TVector3 boost_vec = comp_p.BoostVector();
                                TLorentzVector p1_rest = p1;
                                p1_rest.Boost(-boost_vec);
                                TVector3 p1_dir = p1_rest.Vect().Unit();
                                double cos_helicity = comp_dir.Dot(p1_dir);
                                helicity_angles.push_back(cos_helicity);

                                TVector3 beam_dir(0.0, 0.0, 1.0);
                                TVector3 production_norm = beam_dir.Cross(comp_p.Vect());
                                TVector3 decay_norm = comp_p.Vect().Cross(p1_rest.Vect());
                                double plane_angle = production_norm.Angle(decay_norm);
                                plane_angles.push_back(plane_angle);

                                // Store all information
                                E.push_back(comp_p.E());
                                px.push_back(comp_p.Px());
                                py.push_back(comp_p.Py());
                                pz.push_back(comp_p.Pz());
                                p1_index.push_back(i);
                                p2_index.push_back(j);
                            }}
                        }}
                    
                        return std::make_tuple(E, px, py, pz, helicity_angles, plane_angles, p1_index, p2_index);
                    }}
                """)
                self._defined_functions.add(func_name)
        
            call_func = func_name
        else:
            func_name = "calculate_all_pairs_same"
            if func_name not in self._defined_functions:
                ROOT.gInterpreter.Declare(f"""
                    #include <ROOT/RVec.hxx>
                    #include <TLorentzVector.h>
                    using namespace ROOT::VecOps;
                
                    std::tuple<RVec<double>, RVec<double>, RVec<double>, RVec<double>, RVec<double>, RVec<double>, RVec<int>, RVec<int>>
                    calculate_all_pairs_same(
                        const RVec<double>& p1_px, const RVec<double>& p1_py, const RVec<double>& p1_pz,
                        const RVec<double>& p2_px, const RVec<double>& p2_py, const RVec<double>& p2_pz,
                        double mass_p1, double mass_p2) 
                    {{
                        RVec<double> E;
                        RVec<double> px;
                        RVec<double> py;
                        RVec<double> pz;
                        RVec<double> helicity_angles;
                        RVec<double> plane_angles;
                        RVec<int> p1_index;
                        RVec<int> p2_index;
                    
                        // Same index pairing: only i=j
                        size_t n = std::min(p1_px.size(), p2_px.size());
                        for (size_t i = 0; i < n; ++i) {{
                            TLorentzVector p1;
                            p1.SetXYZM(p1_px[i], p1_py[i], p1_pz[i], mass_p1);
                        
                            TLorentzVector p2;
                            p2.SetXYZM(p2_px[i], p2_py[i], p2_pz[i], mass_p2);
                        
                            // Form composite particle candidate
                            TLorentzVector comp_p = p1 + p2;

                            // Calculate helicity angle
                            TVector3 comp_dir = comp_p.Vect().Unit();
                            TVector3 boost_vec = comp_p.BoostVector();
                            TLorentzVector p1_rest = p1;
                            p1_rest.Boost(-boost_vec);
                            TVector3 p1_dir = p1_rest.Vect().Unit();
                            double cos_helicity = comp_dir.Dot(p1_dir);
                            helicity_angles.push_back(cos_helicity);

                            TVector3 beam_dir(0.0, 0.0, 1.0);
                            TVector3 production_norm = beam_dir.Cross(comp_p.Vect());
                            TVector3 decay_norm = comp_p.Vect().Cross(p1_rest.Vect());
                            double plane_angle = production_norm.Angle(decay_norm);
                            plane_angles.push_back(plane_angle);

                            // Store all information
                            E.push_back(comp_p.E());
                            px.push_back(comp_p.Px());
                            py.push_back(comp_p.Py());
                            pz.push_back(comp_p.Pz());
                            p1_index.push_back(i);
                            p2_index.push_back(i);
                        }}
                    
                        return std::make_tuple(E, px, py, pz, helicity_angles, plane_angles, p1_index, p2_index);
                    }}
                """)
                self._defined_functions.add(func_name)
        
            call_func = func_name
    
        # Define combination variables using the appropriate function
        if "Pairs" not in new_df.GetColumnNames():
            new_df = new_df.Define(
                "Pairs",
                f"{call_func}({p1_branches[0]}, {p1_branches[1]}, {p1_branches[2]}, "
                f"{p2_branches[0]}, {p2_branches[1]}, {p2_branches[2]}, {mass[0]}, {mass[1]})"
            )
        else:
            new_df = new_df.Redefine(
                "Pairs",
                f"{call_func}({p1_branches[0]}, {p1_branches[1]}, {p1_branches[2]}, "
                f"{p2_branches[0]}, {p2_branches[1]}, {p2_branches[2]}, {mass[0]}, {mass[1]})"
            )
        
        # Extract individual components as separate columns
        names = particle_name if particle_name else ("comp_p","p1","p2")
        variables = { "E": f"{names[0]}_E", 
                     "px": f"{names[0]}_px", 
                     "py": f"{names[0]}_py", 
                     "pz": f"{names[0]}_pz", 
                     "helicity_angle": f"{names[0]}_helicity_angle",
                     "helicity_phi": f"{names[0]}_helicity_phi",
                     "p1_index": f"{names[1]}_index", 
                     "p2_index": f"{names[2]}_index"}
                    
        key_to_idx = {"E":0, "px":1, "py":2, "pz":3, "helicity_angle":4, "helicity_phi":5, "p1_index":6, "p2_index":7}
                    
        for key, branch in variables.items():
            idx = key_to_idx[key]
            if branch not in new_df.GetColumnNames():
                new_df = new_df.Define(branch, f"std::get<{idx}>(Pairs)")
            else:
                new_df = new_df.Redefine(branch, f"std::get<{idx}>(Pairs)")

        return new_df


    def convert_spherical_to_cartesian(self, df: ROOT.RDataFrame, 
                                       particles: List[str], 
                                       p_branch: str = "p",
                                       costheta_branch: str = "costheta", 
                                       phi_branch: str = "phi",
                                       output_suffix: str = "") -> ROOT.RDataFrame:
        """
        Convert particle kinematics from spherical coordinates (p, cosθ, φ) to Cartesian (px, py, pz).
        Automatically handles both scalar and vector inputs.
        
        Formula:
        - px = p * sin(θ) * cos(φ) = p * sqrt(1 - cos²θ) * cos(φ)
        - py = p * sin(θ) * sin(φ) = p * sqrt(1 - cos²θ) * sin(φ)
        - pz = p * cos(θ)
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        particles : List[str]
            List of particle name prefixes to process (e.g., ["kp", "km"])
        p_branch : str
            Suffix for momentum magnitude branch (default: "p")
        costheta_branch : str
            Suffix for cos(theta) branch (default: "costheta")
        phi_branch : str
            Suffix for phi angle branch (default: "phi")
        output_suffix : str
            Optional suffix for output branches (e.g., "_cms", "_lab")
            Output branches will be: {particle}_px{suffix}, etc.
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with Cartesian momentum components
        
        Example:
        --------
        # For branches: kp_p, kp_costheta, kp_phi
        df = tools.convert_spherical_to_cartesian(df, ["kp", "km"])
        # Creates: kp_px, kp_py, kp_pz
        
        # For CMS frame with suffix
        df = tools.convert_spherical_to_cartesian(df, ["kp"], 
                                                  p_branch="p_cms",
                                                  costheta_branch="costheta_cms",
                                                  phi_branch="phi_cms",
                                                  output_suffix="_cms")
        # Creates: kp_px_cms, kp_py_cms, kp_pz_cms
        """
        new_df = df
        
        # Define scalar version
        func_name_scalar = "spherical_to_cartesian_scalar"
        if func_name_scalar not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <cmath>
                #include <tuple>
                
                std::tuple<double, double, double> spherical_to_cartesian_scalar(
                    double p, double costheta, double phi) 
                {
                    double sintheta = std::sqrt(1.0 - costheta * costheta);
                    double px = p * sintheta * std::cos(phi);
                    double py = p * sintheta * std::sin(phi);
                    double pz = p * costheta;
                    
                    return std::make_tuple(px, py, pz);
                }
            """)
            RDF_process._defined_functions.add(func_name_scalar)
        
        # Define vector version
        func_name_vec = "spherical_to_cartesian_vec"
        if func_name_vec not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <ROOT/RVec.hxx>
                #include <cmath>
                #include <tuple>
                using namespace ROOT::VecOps;
                
                std::tuple<RVec<double>, RVec<double>, RVec<double>> 
                spherical_to_cartesian_vec(
                    const RVec<double>& p, 
                    const RVec<double>& costheta, 
                    const RVec<double>& phi) 
                {
                    RVec<double> px, py, pz;
                    px.reserve(p.size());
                    py.reserve(p.size());
                    pz.reserve(p.size());
                    
                    for (size_t i = 0; i < p.size(); ++i) {
                        double sintheta = std::sqrt(1.0 - costheta[i] * costheta[i]);
                        px.push_back(p[i] * sintheta * std::cos(phi[i]));
                        py.push_back(p[i] * sintheta * std::sin(phi[i]));
                        pz.push_back(p[i] * costheta[i]);
                    }
                    
                    return std::make_tuple(px, py, pz);
                }
            """)
            RDF_process._defined_functions.add(func_name_vec)
        
        # Define automatic dispatcher
        func_name_auto = "spherical_to_cartesian_auto"
        if func_name_auto not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <ROOT/RVec.hxx>
                using namespace ROOT::VecOps;
                
                // Overload for scalar inputs
                auto spherical_to_cartesian_auto(
                    double p, double costheta, double phi) 
                {
                    return spherical_to_cartesian_scalar(p, costheta, phi);
                }
                
                // Overload for vector inputs
                auto spherical_to_cartesian_auto(
                    const RVec<double>& p, 
                    const RVec<double>& costheta, 
                    const RVec<double>& phi) 
                {
                    return spherical_to_cartesian_vec(p, costheta, phi);
                }
            """)
            RDF_process._defined_functions.add(func_name_auto)
        
        # Apply conversion for each particle
        for particle in particles:
            p_name = f"{particle}_{p_branch}"
            costheta_name = f"{particle}_{costheta_branch}"
            phi_name = f"{particle}_{phi_branch}"
            
            # Define combined result
            if f"__cartesian_{particle}__" not in new_df.GetColumnNames():
                new_df = new_df.Define(
                    f"__cartesian_{particle}__",
                    f"spherical_to_cartesian_auto({p_name}, {costheta_name}, {phi_name})"
                )
            else :
                new_df = new_df.Redefine(
                    f"__cartesian_{particle}__",
                    f"spherical_to_cartesian_auto({p_name}, {costheta_name}, {phi_name})"
                )
            
            # Extract individual components
            px_name = f"{particle}_px{output_suffix}"
            py_name = f"{particle}_py{output_suffix}"
            pz_name = f"{particle}_pz{output_suffix}"
            
            if px_name in new_df.GetColumnNames():
                new_df = new_df.Redefine(px_name, f"std::get<0>(__cartesian_{particle}__)")
            else:
                new_df = new_df.Define(px_name, f"std::get<0>(__cartesian_{particle}__)")
            
            if py_name in new_df.GetColumnNames():
                new_df = new_df.Redefine(py_name, f"std::get<1>(__cartesian_{particle}__)")
            else:
                new_df = new_df.Define(py_name, f"std::get<1>(__cartesian_{particle}__)")
            
            if pz_name in new_df.GetColumnNames():
                new_df = new_df.Redefine(pz_name, f"std::get<2>(__cartesian_{particle}__)")
            else:
                new_df = new_df.Define(pz_name, f"std::get<2>(__cartesian_{particle}__)")
        
        return new_df


    def convert_cartesian_to_spherical(self, df: ROOT.RDataFrame, 
                                       particles: List[str],
                                       px_branch: str = "px",
                                       py_branch: str = "py",
                                       pz_branch: str = "pz",
                                       output_suffix: str = "") -> ROOT.RDataFrame:
        """
        Convert particle kinematics from Cartesian coordinates (px, py, pz) to spherical (p, cosθ, φ).
        Automatically handles both scalar and vector inputs.
        
        Formula:
        - p = sqrt(px² + py² + pz²)
        - cosθ = pz / p
        - φ = atan2(py, px)
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        particles : List[str]
            List of particle name prefixes to process (e.g., ["kp", "km"])
        px_branch : str
            Suffix for px branch (default: "px")
        py_branch : str
            Suffix for py branch (default: "py")
        pz_branch : str
            Suffix for pz branch (default: "pz")
        output_suffix : str
            Optional suffix for output branches (e.g., "_cms", "_truth")
            Output branches will be: {particle}_p{suffix}, {particle}_costheta{suffix}, {particle}_phi{suffix}
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with spherical momentum components
        
        Example:
        --------
        # For branches: kp_px, kp_py, kp_pz
        df = tools.convert_cartesian_to_spherical(df, ["kp", "km"])
        # Creates: kp_p, kp_costheta, kp_phi
        
        # For CMS frame
        df = tools.convert_cartesian_to_spherical(df, ["kp"],
                                                  px_branch="px_cms",
                                                  py_branch="py_cms", 
                                                  pz_branch="pz_cms",
                                                  output_suffix="_cms")
        # Creates: kp_p_cms, kp_costheta_cms, kp_phi_cms
        """
        new_df = df
        
        # Define scalar version
        func_name_scalar = "cartesian_to_spherical_scalar"
        if func_name_scalar not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <cmath>
                #include <tuple>
                
                std::tuple<double, double, double> cartesian_to_spherical_scalar(
                    double px, double py, double pz) 
                {
                    double p = std::sqrt(px*px + py*py + pz*pz);
                    double costheta = (p > 0) ? pz / p : 0.0;
                    double phi = std::atan2(py, px);
                    
                    return std::make_tuple(p, costheta, phi);
                }
            """)
            RDF_process._defined_functions.add(func_name_scalar)
        
        # Define vector version
        func_name_vec = "cartesian_to_spherical_vec"
        if func_name_vec not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <ROOT/RVec.hxx>
                #include <cmath>
                #include <tuple>
                using namespace ROOT::VecOps;
                
                std::tuple<RVec<double>, RVec<double>, RVec<double>> 
                cartesian_to_spherical_vec(
                    const RVec<double>& px, 
                    const RVec<double>& py, 
                    const RVec<double>& pz) 
                {
                    RVec<double> p, costheta, phi;
                    p.reserve(px.size());
                    costheta.reserve(px.size());
                    phi.reserve(px.size());
                    
                    for (size_t i = 0; i < px.size(); ++i) {
                        double p_val = std::sqrt(px[i]*px[i] + py[i]*py[i] + pz[i]*pz[i]);
                        double costheta_val = (p_val > 0) ? pz[i] / p_val : 0.0;
                        double phi_val = std::atan2(py[i], px[i]);
                        
                        p.push_back(p_val);
                        costheta.push_back(costheta_val);
                        phi.push_back(phi_val);
                    }
                    
                    return std::make_tuple(p, costheta, phi);
                }
            """)
            RDF_process._defined_functions.add(func_name_vec)
        
        # Define automatic dispatcher
        func_name_auto = "cartesian_to_spherical_auto"
        if func_name_auto not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <ROOT/RVec.hxx>
                using namespace ROOT::VecOps;
                
                // Overload for scalar inputs
                auto cartesian_to_spherical_auto(
                    double px, double py, double pz) 
                {
                    return cartesian_to_spherical_scalar(px, py, pz);
                }
                
                // Overload for vector inputs
                auto cartesian_to_spherical_auto(
                    const RVec<double>& px, 
                    const RVec<double>& py, 
                    const RVec<double>& pz) 
                {
                    return cartesian_to_spherical_vec(px, py, pz);
                }
            """)
            RDF_process._defined_functions.add(func_name_auto)
        
        # Apply conversion for each particle
        for particle in particles:
            px_name = f"{particle}_{px_branch}"
            py_name = f"{particle}_{py_branch}"
            pz_name = f"{particle}_{pz_branch}"
            
            # Define combined result
            if f"__spherical_{particle}__" not in new_df.GetColumnNames():
                new_df = new_df.Define(
                    f"__spherical_{particle}__",
                    f"cartesian_to_spherical_auto({px_name}, {py_name}, {pz_name})"
                )
            else:
                new_df = new_df.Redefine(
                    f"__spherical_{particle}__",
                    f"cartesian_to_spherical_auto({px_name}, {py_name}, {pz_name})"
                )
            
            # Extract individual components
            p_name = f"{particle}_p{output_suffix}"
            costheta_name = f"{particle}_costheta{output_suffix}"
            phi_name = f"{particle}_phi{output_suffix}"
            
            if p_name in new_df.GetColumnNames():
                new_df = new_df.Redefine(p_name, f"std::get<0>(__spherical_{particle}__)")
            else:
                new_df = new_df.Define(p_name, f"std::get<0>(__spherical_{particle}__)")
            
            if costheta_name in new_df.GetColumnNames():
                new_df = new_df.Redefine(costheta_name, f"std::get<1>(__spherical_{particle}__)")
            else:
                new_df = new_df.Define(costheta_name, f"std::get<1>(__spherical_{particle}__)")
            
            if phi_name in new_df.GetColumnNames():
                new_df = new_df.Redefine(phi_name, f"std::get<2>(__spherical_{particle}__)")
            else:
                new_df = new_df.Define(phi_name, f"std::get<2>(__spherical_{particle}__)")
        
        return new_df

    def cal_pola_angles(self, df: ROOT.RDataFrame, 
                                p1_branches : Tuple[str, str, str],
                                p2_branches : Tuple[str, str, str],
                                mass: Tuple[float, float], axis: Tuple[str, str],
                                mother_index_branches: Optional[Tuple[str, str]] = None,
                                output_names: Tuple[str, str] = ("cos_theta", "phi")) -> ROOT.RDataFrame:
        """
        Calculate the polarization angles for a given DataFrame.
        """

        new_df = df

        func_name = "calculate_polarization_angles"
        if func_name not in self._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <ROOT/RVec.hxx>
                #include <TLorentzVector.h>
                using namespace ROOT::VecOps;
                
                std::tuple<double, double> 
                calculate_angles(
                    double p1_px, double p1_py, double p1_pz,
                    double p2_px, double p2_py, double p2_pz,
                    double mass_p1, double mass_p2,
                    double axis_costheta, double axis_phi,
                    int mother_index_p1 =0, int mother_index_p2 = 0) // two index actually not used in this function, 
                {
                    TLorentzVector p1,p2;
                    p1.SetXYZM(p1_px, p1_py, p1_pz, mass_p1);
                    p2.SetXYZM(p2_px, p2_py, p2_pz, mass_p2);
                    TLorentzVector parent = p1 + p2;

                    // --- define coordinate system in parent rest frame ---
                    // z: parent flight direction
                    TVector3 z_hat = parent.Vect().Unit();

                    // reference axis from spherical coordinates (axis_costheta, axis_phi)
                    double axis_sintheta = sqrt(1.0 - axis_costheta * axis_costheta);
                    TVector3 axis_vec(axis_sintheta * cos(axis_phi),
                                    axis_sintheta * sin(axis_phi),
                                    axis_costheta);

                    // y = z x axis  (normal to the plane spanned by z and axis)
                    TVector3 y_hat = z_hat.Cross(axis_vec).Unit();

                    // x = y x z  (right-handed: x x y = z)
                    TVector3 x_hat = y_hat.Cross(z_hat);

                    // --- boost p1 into parent rest frame ---
                    TLorentzVector p1_rest = p1;
                    p1_rest.Boost(-parent.BoostVector());
                    TVector3 p1_vec = p1_rest.Vect();

                    // --- project onto new axes ---
                    double cos_theta = p1_vec.Dot(z_hat) / p1_vec.Mag();
                    double phi       = atan2(p1_vec.Dot(y_hat), p1_vec.Dot(x_hat));

                    return std::make_tuple(cos_theta, phi);
                }

                std::tuple<RVec<double>, RVec<double>>
                calculate_angles(
                    RVec<double> p1_px, RVec<double> p1_py, RVec<double> p1_pz,
                    RVec<double> p2_px, RVec<double> p2_py, RVec<double> p2_pz,
                    double mass_p1, double mass_p2,
                    double axis_costheta, double axis_phi,
                    RVec<int> mother_index_p1, RVec<int> mother_index_p2)
                {
                    size_t n = mother_index_p1.size();
                    RVec<double> cos_theta(n);
                    RVec<double> phi(n);
                    for (size_t i = 0; i < n; ++i) {
                        std::tie(cos_theta[i], phi[i]) = calculate_angles(
                            p1_px[mother_index_p1[i]], p1_py[mother_index_p1[i]], p1_pz[mother_index_p1[i]],
                            p2_px[mother_index_p2[i]], p2_py[mother_index_p2[i]], p2_pz[mother_index_p2[i]],
                            mass_p1, mass_p2,
                            axis_costheta, axis_phi
                        );
                    }
                    return std::make_tuple(cos_theta, phi);
                }
                """)
            self._defined_functions.add(func_name)

        if "Pairs" not in df.GetColumnNames():
            new_df = new_df.Define("Pairs", f"calculate_angles({p1_branches[0]}, {p1_branches[1]}, {p1_branches[2]}, {p2_branches[0]}," 
                                            f"{p2_branches[1]}, {p2_branches[2]}, {mass[0]}, {mass[1]}, {axis[0]}, {axis[1]}, {mother_index_branches[0]}, {mother_index_branches[1]})")
        else: 
            print("Warning: 'Pairs' column already exists. Overwriting with new angles.")
            new_df = new_df.Redefine("Pairs", f"calculate_angles({p1_branches[0]}, {p1_branches[1]}, {p1_branches[2]}, {p2_branches[0]}," 
                                            f"{p2_branches[1]}, {p2_branches[2]}, {mass[0]}, {mass[1]}, {axis[0]}, {axis[1]}, {mother_index_branches[0]}, {mother_index_branches[1]})")
        
        if output_names[0] in df.GetColumnNames() or output_names[1] in df.GetColumnNames():
            print(f"Warning: Output columns {output_names} already exist. The calculation will not be written.")
            return df
        else :
            new_df = new_df.Define(output_names[0], f"std::get<0>(Pairs)")
            new_df = new_df.Define(output_names[1], f"std::get<1>(Pairs)")

        return new_df







