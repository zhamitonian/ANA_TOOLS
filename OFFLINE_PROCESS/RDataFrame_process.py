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

    def select_phi(self, df:ROOT.RDataFrame, kaon_vars:Optional[List[str]] = None)->ROOT.RDataFrame:
        
        phi_mass = 1.019455

        new_df = df
        
        pairs = [("kp01", "km01"), ("kp01", "km02"), ("kp02", "km01"), ("kp02", "km02") ]
        
        for kp, km in pairs:
            new_df = new_df.Define(
                f"M_{kp}{km}",
                f"sqrt(pow({kp}_E + {km}_E, 2) - pow({kp}_px + {km}_px, 2) - pow({kp}_py + {km}_py, 2) - pow({kp}_pz + {km}_pz, 2))"
            )
            new_df = new_df.Define(
                f"diff_M_{kp}{km}",
                f"abs(M_{kp}{km} - {phi_mass})"
            )
        
        
        min_diff_expr = "std::min(std::min(std::min(diff_M_kp01km01, diff_M_kp01km02), diff_M_kp02km01), diff_M_kp02km02)"
        new_df = new_df.Define("min_mass_diff", min_diff_expr)
        
        for i, (kp, km) in enumerate(pairs):
            new_df = new_df.Define(
                f"is_{kp}{km}",
                f"diff_M_{kp}{km} == min_mass_diff"
            )
        
        if kaon_vars is None:
            branch_names = ["E", "px", "py", "pz" ]
            #branch_names += ["E_CMS", "p_CMS", "px_CMS", "py_CMS" ]
            #branch_names += ['p' ,'theta','phi', 'nCDCHits', 'nPXDHits', 'nSVDHits', 'dr', 'dz']
        else: 
            branch_names = kaon_vars
        
        # define M_phi
        mass_expr = " + ".join([
            f"M_{kp}{km} * is_{kp}{km}" for kp, km in pairs
        ])
        new_df = new_df.Define("phi_M", mass_expr)
        
        for comp in branch_names:
            # phi K+ K-
            kp_expr = " + ".join([
                f"{kp}_{comp} * (is_{kp}km01 || is_{kp}km02)" for kp in ["kp01", "kp02"]
            ])
            new_df = new_df.Define(f"phikp_{comp}", kp_expr)
            
            km_expr = " + ".join([
                f"{km}_{comp} * (is_kp01{km} || is_kp02{km})" for km in ["km01", "km02"]
            ])
            new_df = new_df.Define(f"phikm_{comp}", km_expr)
            
            # ee K+ K-
            eekp_expr = " + ".join([
                f"kp01_{comp} * (is_kp02km01 || is_kp02km02)",
                f"kp02_{comp} * (is_kp01km01 || is_kp01km02)"
            ])
            new_df = new_df.Define(f"eekp_{comp}", eekp_expr)
            
            eekm_expr = " + ".join([
                f"km01_{comp} * (is_kp01km02 || is_kp02km02)",
                f"km02_{comp} * (is_kp01km01 || is_kp02km01)"
            ])
            new_df = new_df.Define(f"eekm_{comp}", eekm_expr)
        
        # M_kk
        new_df = new_df.Define(
            "M_kk",
            "sqrt(pow(eekp_E + eekm_E, 2) - pow(eekp_px + eekm_px, 2) - pow(eekp_py + eekm_py, 2) - pow(eekp_pz + eekm_pz, 2))"
        )
        
        # 计算四个kaon的总不变质量
        new_df = new_df.Define(
            "M_phikk",
            "sqrt(pow(kp01_E + km01_E + kp02_E + km02_E, 2) - " +
            "pow(kp01_px + km01_px + kp02_px + km02_px, 2) - " +
            "pow(kp01_py + km01_py + kp02_py + km02_py, 2) - " +
            "pow(kp01_pz + km01_pz + kp02_pz + km02_pz, 2))"
        )
  
        return new_df

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

    def convert_lab_variables(self, df:ROOT.RDataFrame, particles:List[str], var2save:Optional[List[str]]=None )->ROOT.RDataFrame:
        """
        Convert particles' 4-momentum to calculate kinematic variables in lab frame.
        Calculate invariant mass, phi angle, theta angle, momentum, transverse momentum, etc.
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe containing particle 4-momenta (px, py, pz, E)
        particles : List[str]
            List of particle name prefixes to process
        var2save : Optional[List[str]]
            Variables to calculate among ["M", "p", "pt", "theta", "phi"]
            If None, all variables will be calculated
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with calculated kinematic variables
        """ 
        new_df = df
        
        # Define the kinematic calculation function in the ROOT interpreter
        func_name = "calculate_kinematics"
        if func_name not in self._defined_functions:
            ROOT.gInterpreter.Declare("""
                std::vector<double> calculate_kinematics(double px, double py, double pz, double E) {
                    // Create a TLorentzVector for the particle
                    TLorentzVector particle;
                    particle.SetPxPyPzE(px, py, pz, E);
                    
                    // Calculate kinematic variables
                    double M = particle.M();           // Invariant mass
                    double p = particle.P();           // Total momentum
                    double pt = particle.Pt();         // Transverse momentum
                    double theta = particle.Theta();   // Polar angle (0 to pi)
                    double phi = particle.Phi();       // Azimuthal angle (-pi to pi)
                    
                    return std::vector<double>{M, p, pt, theta, phi};
                }
            """)
            RDF_process._defined_functions.add(func_name)
        
        # Define available variables and their indices in the returned vector
        all_vars = ["M", "p", "pt", "theta", "phi"]
        if var2save is None:
            var2save = all_vars
        
        var_indices = {
            "M": 0,      # Invariant mass
            "p": 1,      # Total momentum
            "pt": 2,     # Transverse momentum
            "theta": 3,  # Polar angle
            "phi": 4     # Azimuthal angle
        }

        # Calculate kinematic variables for each particle
        for particle in particles:
            for var in var2save:
                if var in var_indices:
                    column_name = f"{particle}_{var}"
                    if column_name not in new_df.GetColumnNames():
                        idx = var_indices[var]
                        new_df = new_df.Define(
                            column_name, 
                            f"calculate_kinematics({particle}_px, {particle}_py, {particle}_pz, {particle}_E)[{idx}]"
                        )
        
        return new_df

    def select_diPhi(self, df:ROOT.RDataFrame,kaon_vars:Optional[List[str]] = None)->ROOT.RDataFrame: 
        
        phi_mass = 1.019455

        new_df = df
        
        pairs = [("kp01", "km01"), ("kp01", "km02"), ("kp02", "km01"), ("kp02", "km02")]
        
        for kp, km in pairs:
            new_df = new_df.Define(
                f"M_{kp}{km}",
                f"sqrt(pow({kp}_E + {km}_E, 2) - pow({kp}_px + {km}_px, 2) - pow({kp}_py + {km}_py, 2) - pow({kp}_pz + {km}_pz, 2))"
            )
        
        # Combination 1: (kp01+km01, kp02+km02)
        # Combination 2: (kp01+km02, kp02+km01)
        new_df = new_df.Define(
            "dist_comb1", 
            f"sqrt(pow(M_kp01km01 - {phi_mass}, 2) + pow(M_kp02km02 - {phi_mass}, 2))"
        )
        new_df = new_df.Define(
            "dist_comb2", 
            f"sqrt(pow(M_kp01km02 - {phi_mass}, 2) + pow(M_kp02km01 - {phi_mass}, 2))"
        )
        
        # Determine which combination is better (smaller combined distance)
        new_df = new_df.Define("is_comb1", "dist_comb1 < dist_comb2")
        
        if kaon_vars is None:
            branch_names = ["E", "px", "py", "pz" ,"theta", "phi"]
        else:   
            branch_names = kaon_vars
        
        # Define masses of the two phi candidates
        new_df = new_df.Define(
            "phi1_M", 
            "is_comb1 ? M_kp01km01 : M_kp01km02"
        )
        
        new_df = new_df.Define(
            "phi2_M", 
            "is_comb1 ? M_kp02km02 : M_kp02km01"
        )
        
        new_df = new_df.Define(
            "delta_M",
            "is_comb1 ? dist_comb1 : dist_comb2"
        )

        # Define components for phi1
        for comp in branch_names:
            new_df = new_df.Define(
                f"phi1kp_{comp}", 
                f"is_comb1 ? kp01_{comp} : kp01_{comp}"
            )
            new_df = new_df.Define(
                f"phi1km_{comp}", 
                f"is_comb1 ? km01_{comp} : km02_{comp}"
            )
        
        # Define components for phi2
        for comp in branch_names:
            new_df = new_df.Define(
                f"phi2kp_{comp}", 
                f"is_comb1 ? kp02_{comp} : kp02_{comp}"
            )
            new_df = new_df.Define(
                f"phi2km_{comp}", 
                f"is_comb1 ? km02_{comp} : km01_{comp}"
            )
        
        # Calculate combined momentum for each phi
        kinematics = ["E", "px", "py", "pz"]
        for comp in kinematics:
            new_df = new_df.Define(
                f"phi1_{comp}", 
                f"phi1kp_{comp} + phi1km_{comp}"
            )
            new_df = new_df.Define(
                f"phi2_{comp}", 
                f"phi2kp_{comp} + phi2km_{comp}"
            )
        
        return new_df

    def select_diOmega(self, df:ROOT.RDataFrame, pion_vars:Optional[List[str]] = None)->ROOT.RDataFrame:
        """
        Select two omega candidates from 2π+ 2π- 2π0 combinations.
        Minimize sqrt((m_omega1 - m_pdg)^2 + (m_omega2 - m_pdg)^2)
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        pion_vars : Optional[List[str]]
            List of pion variable names to process
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with omega selections
        """
        omega_mass = 0.78266  # PDG mass of omega meson in GeV

        new_df = df
        
        # Define all possible combinations for omega candidates (π+π-π0)
        combinations = [
            (("pip01", "pim01", "pi01"), ("pip02", "pim02", "pi02")),  # Combination 1
            (("pip01", "pim01", "pi02"), ("pip02", "pim02", "pi01")),  # Combination 2
            (("pip01", "pim02", "pi01"), ("pip02", "pim01", "pi02")),  # Combination 3
            (("pip01", "pim02", "pi02"), ("pip02", "pim01", "pi01"))   # Combination 4
        ]
        
        # Calculate invariant mass for each triplet π+π-π0
        for i, combo_pair in enumerate(combinations):
            for j, (pip, pim, pi0) in enumerate(combo_pair):
                new_df = new_df.Define(
                    f"M_{pip}{pim}{pi0}",
                    f"sqrt(pow({pip}_E + {pim}_E + {pi0}_E, 2) - pow({pip}_px + {pim}_px + {pi0}_px, 2) - "
                    f"pow({pip}_py + {pim}_py + {pi0}_py, 2) - pow({pip}_pz + {pim}_pz + {pi0}_pz, 2))"
                )
        
        # Calculate distances for each combination
        for i, ((pip1, pim1, pi01), (pip2, pim2, pi02)) in enumerate(combinations):
            new_df = new_df.Define(
                f"dist_comb{i+1}", 
                f"sqrt(pow(M_{pip1}{pim1}{pi01} - {omega_mass}, 2) + pow(M_{pip2}{pim2}{pi02} - {omega_mass}, 2))"
            )
        
        # Determine which combination has the smallest distance
        min_dist_expr = "std::min(std::min(std::min(dist_comb1, dist_comb2), dist_comb3), dist_comb4)"
        new_df = new_df.Define("min_omega_dist", min_dist_expr)
        
        for i in range(1, 5):
            new_df = new_df.Define(
                f"is_comb{i}",
                f"dist_comb{i} == min_omega_dist"
            )
        
        if pion_vars is None:
            branch_names = ["E", "px", "py", "pz"]
        else:
            branch_names = pion_vars
        
        # Define masses of the two omega candidates
        mass_expr_omega1 = " + ".join([
            f"M_{combo[0][0]}{combo[0][1]}{combo[0][2]} * is_comb{i+1}" 
            for i, combo in enumerate(combinations)
        ])
        
        mass_expr_omega2 = " + ".join([
            f"M_{combo[1][0]}{combo[1][1]}{combo[1][2]} * is_comb{i+1}" 
            for i, combo in enumerate(combinations)
        ])
        
        new_df = new_df.Define("omega1_M", mass_expr_omega1)
        new_df = new_df.Define("omega2_M", mass_expr_omega2)
        
        # Define components for omega1 particles (π+, π-, π0)
        for comp in branch_names:
            # For π+ in omega1
            pip_expr_omega1 = " + ".join([
                f"{combo[0][0]}_{comp} * is_comb{i+1}" 
                for i, combo in enumerate(combinations)
            ])
            new_df = new_df.Define(f"omega1pip_{comp}", pip_expr_omega1)
            
            # For π- in omega1
            pim_expr_omega1 = " + ".join([
                f"{combo[0][1]}_{comp} * is_comb{i+1}" 
                for i, combo in enumerate(combinations)
            ])
            new_df = new_df.Define(f"omega1pim_{comp}", pim_expr_omega1)
            
            # For π0 in omega1
            pi0_expr_omega1 = " + ".join([
                f"{combo[0][2]}_{comp} * is_comb{i+1}" 
                for i, combo in enumerate(combinations)
            ])
            new_df = new_df.Define(f"omega1pi0_{comp}", pi0_expr_omega1)
        
        # Define components for omega2 particles (π+, π-, π0)
        for comp in branch_names:
            # For π+ in omega2
            pip_expr_omega2 = " + ".join([
                f"{combo[1][0]}_{comp} * is_comb{i+1}" 
                for i, combo in enumerate(combinations)
            ])
            new_df = new_df.Define(f"omega2pip_{comp}", pip_expr_omega2)
            
            # For π- in omega2
            pim_expr_omega2 = " + ".join([
                f"{combo[1][1]}_{comp} * is_comb{i+1}" 
                for i, combo in enumerate(combinations)
            ])
            new_df = new_df.Define(f"omega2pim_{comp}", pim_expr_omega2)
            
            # For π0 in omega2
            pi0_expr_omega2 = " + ".join([
                f"{combo[1][2]}_{comp} * is_comb{i+1}" 
                for i, combo in enumerate(combinations)
            ])
            new_df = new_df.Define(f"omega2pi0_{comp}", pi0_expr_omega2)
        
        # Calculate combined momentum for each omega
        kinematics = ["E", "px", "py", "pz"]
        for comp in kinematics:
            new_df = new_df.Define(
                f"omega1_{comp}", 
                f"omega1pip_{comp} + omega1pim_{comp} + omega1pi0_{comp}"
            )
            new_df = new_df.Define(
                f"omega2_{comp}", 
                f"omega2pip_{comp} + omega2pim_{comp} + omega2pi0_{comp}"
            )
        
        return new_df
    

    def reconstruct_diKstar(self, df:ROOT.RDataFrame) -> ROOT.RDataFrame:
        """
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe containing K+ and K- candidates
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with diK* system variables
        """
        new_df = df
        
        # Define the invariant mass of the diK* system
        new_df = new_df.Define(
            "Kstar_M",
            "sqrt(pow(kp_E + pim_E, 2) - pow(kp_px + pim_px, 2) - pow(kp_py + pim_py, 2) - pow(kp_pz + pim_pz, 2))"
        )
        new_df = new_df.Define("Kstar_E", "kp_E + pim_E")
        new_df = new_df.Define("Kstar_px", "kp_px + pim_px")
        new_df = new_df.Define("Kstar_py", "kp_py + pim_py")
        new_df = new_df.Define("Kstar_pz", "kp_pz + pim_pz")

        new_df = new_df.Define(
            "antiKstar_M",
            "sqrt(pow(km_E + pip_E, 2) - pow(km_px + pip_px, 2) - pow(km_py + pip_py, 2) - pow(km_pz + pip_pz, 2))"
        )                                                                                                                           

        new_df = new_df.Define("antiKstar_E", "km_E + pip_E")
        new_df = new_df.Define("antiKstar_px", "km_px + pip_px")
        new_df = new_df.Define("antiKstar_py", "km_py + pip_py")
        new_df = new_df.Define("antiKstar_pz", "km_pz + pip_pz")
        
        return new_df
    

    def calculate_PHI(self, df: ROOT.RDataFrame, particle_pair: Tuple[str, str]) -> ROOT.RDataFrame:
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
        
        if p1 == "phiA":
            print(f"Calculating PHI for {p1} and {p2} using CMS variables")
            new_df = new_df.Define(
                f"PHI_{p1}_{p2}",
                #f"calculate_vector_dot_product({p1}_px, {p1}_py, {p2}_px, {p2}_py)"
                #f"calculate_vector_dot_product({p1}_ee_cms_px, {p1}_ee_cms_py, {p2}_ee_cms_px, {p2}_ee_cms_py)"
                f"calculate_vector_dot_product({p1}_px_CMS, {p1}_py_CMS, {p2}_px_CMS, {p2}_py_CMS)"
                #f"calculate_vector_dot_product({p1}_diGam_cms_px, {p1}_diGam_cms_py, {p2}_diGam_cms_px, {p2}_diGam_cms_py)"
            )
        else: 
            new_df = new_df.Define(
                f"PHI_{p1}_{p2}",
                f"calculate_vector_dot_product({p1}_ee_cms_px, {p1}_ee_cms_py, {p2}_ee_cms_px, {p2}_ee_cms_py)"
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
        # Key: (experiment, run, event), Value: (min_value, [list_of_entry_indices])
        event_candidates = {}
        
        # Single pass through data to find minimum values and collect all entries with that minimum
        for i in range(total_entries):
            event_key = (data["__experiment__"][i], 
                        data["__run__"][i], 
                        data["__event__"][i])
            
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

    def quick_reweight(self, mc_df: ROOT.RDataFrame, hist_config:Optional[Tuple[str,int,float,float]]=None, data_df:Optional[ROOT.RDataFrame]=None, h_data:Optional[ROOT.TH1]=None, simple_Scale: Optional[bool] = True) -> ROOT.RDataFrame:
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
        simple_Scale : bool
            Whether to normalize histograms before calculating weights
            
        Returns:
        --------
        ROOT.RDataFrame: Weighted MC dataframe
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

        h_mc = mc_df.Histo1D((f"h_mc_{var}", f"MC {var}", bin, xmin, xmax), var)
        h_mc_ptr = h_mc.GetPtr()
        h_data_ptr = h_data.GetPtr()
        
        if simple_Scale:
            h_data_ptr.Scale(1.0 / h_data_ptr.Integral())
            h_mc_ptr.Scale(1.0 / h_mc_ptr.Integral())
        
        weights = []
        for i in range(1, bin + 1):
            data_content = h_data_ptr.GetBinContent(i)
            mc_content = h_mc_ptr.GetBinContent(i)
            weight = data_content / mc_content if mc_content > 0 else 1.0
            print(f"Bin {i}: Data = {data_content}, MC = {mc_content}, Weight = {weight}")
            weights.append(weight)
        
        weight_array = np.array(weights, dtype=np.float64)

        func_name = "get_bin_weight"
        if func_name not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare(f"""
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
            """)
            RDF_process._defined_functions.add(func_name)
        
        df_weighted = mc_df.Define("data_mc_weight", f"get_bin_weight({var}, {xmin}, {xmax}, {bin})")
        
        return df_weighted

    def calculate_HelicityAngle(self, df: ROOT.RDataFrame, particle_pair: Tuple[Tuple[str,str], Tuple[str,str]],other_particle:Optional[List[str]] = None) -> ROOT.RDataFrame:
        """
        Calculate helicity angles according to the definition:
        1. In the rest frame of the intermediate particle, the angle between 
        the daughter particle momentum and the intermediate particle momentum 
        direction in the CMS frame.
        2. The angle between two decay planes in the CMS frame.
        
        Parameters:
        -----------
        df : ROOT.RDataFrame
            Input dataframe
        particle_pair : tuple[Tuple[str,str], Tuple[str,str]]
            Name of the two particle pairs, e.g. (("phi1", "phi2"), ("Kstar", "antiKstar"))
        other_particle : Optional[List[str]]
            List of other final state particles, used for ee cms variables calculation
            
        Returns:
        --------
        ROOT.RDataFrame: Updated dataframe with helicity angles
        """
        # Correctly unpack the nested tuples
        ((p11, p12), (p21, p22)) = particle_pair
        inter_p1, inter_p2 = "A", "B" 
        new_df = df

        # Define intermediate particle 1 ,2 variables 
        for comp in ["E", "px", "py", "pz"]:
            new_df = new_df.Define(f"{inter_p1}_{comp}", f"{p11}_{comp} + {p12}_{comp}")
            new_df = new_df.Define(f"{inter_p2}_{comp}", f"{p21}_{comp} + {p22}_{comp}")

        new_df = self.set_CMS_variables(new_df,
                                        FSPs=[p11, p12, p21, p22] + (other_particle if other_particle else []),
                                        particles=[p11, p12, p21, p22, inter_p1, inter_p2],
                                        prefix="ee",
                                        var2save=["E", "px", "py", "pz", "p", "pt", "theta", "phi"])
        new_df = self.set_CMS_variables(new_df,
                                        FSPs=[f"{inter_p1}"],
                                        particles=[p11, p12],
                                        prefix=inter_p1,
                                        var2save=["E", "px", "py", "pz", "p", "pt", "theta", "phi"])
        new_df = self.set_CMS_variables(new_df,
                                        FSPs=[f"{inter_p2}"],
                                        particles=[p21, p22],
                                        prefix=inter_p2,
                                        var2save=["E", "px", "py", "pz", "p", "pt", "theta", "phi"]) 

        func_name = "calculate_helicity_angles"
        if func_name not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                std::vector<double> calculate_helicity_angles(
                    double p1_cms_px , double p1_cms_py, double p1_cms_pz,
                    double p2_cms_px , double p2_cms_py, double p2_cms_pz,
                    double p11_p1_cms_px, double p11_p1_cms_py, double p11_p1_cms_pz,
                    double p21_p2_cms_px, double p21_p2_cms_py, double p21_p2_cms_pz
                ) {
                    TVector3 inter1(p1_cms_px, p1_cms_py, p1_cms_pz);
                    TVector3 inter2(p2_cms_px, p2_cms_py, p2_cms_pz);
                    TVector3 p11(p11_p1_cms_px, p11_p1_cms_py, p11_p1_cms_pz);
                    TVector3 p21(p21_p2_cms_px, p21_p2_cms_py, p21_p2_cms_pz);
                    
                    inter1 = inter1.Unit();
                    inter2 = inter2.Unit();
                    p11 = p11.Unit();
                    p21 = p21.Unit();

                    double cos_helicity_angle_1 = inter1.Dot(p11);
                    double cos_helicity_angle_2 = inter2.Dot(p21);
                    
                    TVector3 plane1_cross = inter1.Cross(p11);
                    TVector3 plane2_cross = inter2.Cross(p21);

                    plane1_cross = plane1_cross.Unit();
                    plane2_cross = plane2_cross.Unit();                   

                    double cos_helicity_angle_3 = plane1_cross.Dot(plane2_cross);

                    std::vector<double> result = {cos_helicity_angle_1, cos_helicity_angle_2, cos_helicity_angle_3};
                    
                    return result; 
                    
                    }

                """)    
            RDF_process._defined_functions.add(func_name)

        for i in range(0, 3):
            new_df = new_df.Define(f"helicity_angles{i}",
                                f"calculate_helicity_angles({inter_p1}_ee_cms_px,{inter_p1}_ee_cms_py,{inter_p1}_ee_cms_pz,"
                                f"{inter_p2}_ee_cms_px,{inter_p2}_ee_cms_py,{inter_p2}_ee_cms_pz,"
                                f"{p11}_{inter_p1}_cms_px,{p11}_{inter_p1}_cms_py,{p11}_{inter_p1}_cms_pz,"
                                f"{p21}_{inter_p2}_cms_px,{p21}_{inter_p2}_cms_py,{p21}_{inter_p2}_cms_pz)[{i}]")
        
        return new_df

        
        # Define the C++ function for helicity angle calculation
        if not hasattr(self, 'helicity_func_defined'):
            ROOT.gInterpreter.Declare("""
                struct HelicityResult {
                    double helicity_angle_1;      // Helicity angle for first intermediate particle
                    double helicity_angle_2;      // Helicity angle for second intermediate particle
                    double opening_angle;         // Opening angle between two intermediate particles
                    double azimuthal_phi;         // Your original PHI calculation
                    double decay_plane_angle;     // Angle between two decay planes in CMS frame
                };
                
                HelicityResult calculate_helicity_angles(
                    double inter1_px, double inter1_py, double inter1_pz, double inter1_E,
                    double daughter1a_px, double daughter1a_py, double daughter1a_pz, double daughter1a_E,
                    double daughter1b_px, double daughter1b_py, double daughter1b_pz, double daughter1b_E,
                    double inter2_px, double inter2_py, double inter2_pz, double inter2_E,
                    double daughter2a_px, double daughter2a_py, double daughter2a_pz, double daughter2a_E,
                    double daughter2b_px, double daughter2b_py, double daughter2b_pz, double daughter2b_E
                ) {
                    HelicityResult result;
                    
                    // Create TLorentzVector for intermediate particles
                    TLorentzVector inter1(inter1_px, inter1_py, inter1_pz, inter1_E);
                    TLorentzVector inter2(inter2_px, inter2_py, inter2_pz, inter2_E);
                    TLorentzVector daughter1a(daughter1a_px, daughter1a_py, daughter1a_pz, daughter1a_E);
                    TLorentzVector daughter1b(daughter1b_px, daughter1b_py, daughter1b_pz, daughter1b_E);
                    TLorentzVector daughter2a(daughter2a_px, daughter2a_py, daughter2a_pz, daughter2a_E);
                    TLorentzVector daughter2b(daughter2b_px, daughter2b_py, daughter2b_pz, daughter2b_E);
                    
                    // Total system (CMS frame reference)
                    TLorentzVector total_system = inter1 + inter2;
                    TVector3 cms_boost = total_system.BoostVector();
                    
                    // Boost all particles to CMS frame
                    TLorentzVector inter1_cms = inter1;
                    TLorentzVector inter2_cms = inter2;
                    TLorentzVector daughter1a_cms = daughter1a;
                    TLorentzVector daughter1b_cms = daughter1b;
                    TLorentzVector daughter2a_cms = daughter2a;
                    TLorentzVector daughter2b_cms = daughter2b;
                    
                    inter1_cms.Boost(-cms_boost);
                    inter2_cms.Boost(-cms_boost);
                    daughter1a_cms.Boost(-cms_boost);
                    daughter1b_cms.Boost(-cms_boost);
                    daughter2a_cms.Boost(-cms_boost);
                    daughter2b_cms.Boost(-cms_boost);
                    
                    // === Helicity angle for first intermediate particle ===
                    // 1. Get intermediate particle direction in CMS frame
                    TVector3 inter1_direction_cms = inter1_cms.Vect().Unit();
                    
                    // 2. Boost daughter to intermediate particle rest frame
                    TVector3 boost1 = inter1.BoostVector();
                    TLorentzVector daughter1a_rest = daughter1a;
                    daughter1a_rest.Boost(-boost1);
                    
                    // 3. Get daughter direction in intermediate particle rest frame
                    TVector3 daughter1a_direction_rest = daughter1a_rest.Vect().Unit();
                    
                    // 4. Calculate helicity angle (angle between these two directions)
                    double cos_theta1 = inter1_direction_cms.Dot(daughter1a_direction_rest);
                    result.helicity_angle_1 = acos(abs(cos_theta1));
                    
                    // === Helicity angle for second intermediate particle ===
                    TVector3 inter2_direction_cms = inter2_cms.Vect().Unit();
                    
                    TVector3 boost2 = inter2.BoostVector();
                    TLorentzVector daughter2a_rest = daughter2a;
                    daughter2a_rest.Boost(-boost2);
                    
                    TVector3 daughter2a_direction_rest = daughter2a_rest.Vect().Unit();
                    
                    double cos_theta2 = inter2_direction_cms.Dot(daughter2a_direction_rest);
                    result.helicity_angle_2 = acos(abs(cos_theta2));
                    
                    // === Angle between decay planes in CMS frame ===
                    // Decay plane 1: defined by inter1 momentum and daughter1a momentum in CMS
                    TVector3 inter1_vec_cms = inter1_cms.Vect();
                    TVector3 daughter1a_vec_cms = daughter1a_cms.Vect();
                    TVector3 normal1 = inter1_vec_cms.Cross(daughter1a_vec_cms);
                    
                    // Decay plane 2: defined by inter2 momentum and daughter2a momentum in CMS  
                    TVector3 inter2_vec_cms = inter2_cms.Vect();
                    TVector3 daughter2a_vec_cms = daughter2a_cms.Vect();
                    TVector3 normal2 = inter2_vec_cms.Cross(daughter2a_vec_cms);

                    // Angle between the two planes (angle between their normal vectors)
                    if (normal1.Mag() > 0 && normal2.Mag() > 0) {
                        double cos_plane_angle = normal1.Dot(normal2) / (normal1.Mag() * normal2.Mag());
                        result.decay_plane_angle = acos(abs(cos_plane_angle));  // Take absolute value for [0, π/2]
                    } else {
                        result.decay_plane_angle = 0.0;  // Degenerate case
                    }
                    
                    // === Additional useful angles ===
                    // Opening angle between two intermediate particles
                    result.opening_angle = inter1.Vect().Angle(inter2.Vect());
                    
                    // Azimuthal angle (your original PHI calculation)
                    TVector2 pt1(inter1_px, inter1_py);
                    TVector2 pt2(inter2_px, inter2_py);
                    TVector2 x = pt1 + pt2;
                    TVector2 y = (pt1 - pt2) * 0.5;
                    
                    if (x.Mod() > 0 && y.Mod() > 0) {
                        double dot_product = x * y / (x.Mod() * y.Mod());
                        double cross = x.X() * y.Y() - x.Y() * y.X();
                        
                        if(cross >= 0) {
                            result.azimuthal_phi = acos(dot_product);
                        } else {
                            result.azimuthal_phi = 2*M_PI - acos(dot_product);
                        }
                    } else {
                        result.azimuthal_phi = 0.0;
                    }
                    
                    return result;
                }
            """)
            self.helicity_func_defined = True
        
        p1, p2 = particle_pair
        
        # For diphi analysis: phi1 and phi2 are intermediate particles
        # phi1kp, phi1km and phi2kp, phi2km are their daughters
        new_df = new_df.Define(
            f"helicity_result_{p1[0]}_{p2[0]}",
            f"calculate_helicity_angles("
            # First intermediate particle (phi1)
            f"{p1[0]}_px, {p1[0]}_py, {p1[0]}_pz, {p1[0]}_E, "
            # Daughters of phi1 (K+ and K-)
            f"{p1[0]}kp_px, {p1[0]}kp_py, {p1[0]}kp_pz, {p1[0]}kp_E, "
            f"{p1[0]}km_px, {p1[0]}km_py, {p1[0]}km_pz, {p1[0]}km_E, "
            # Second intermediate particle (phi2)
            f"{p2[0]}_px, {p2[0]}_py, {p2[0]}_pz, {p2[0]}_E, "
            # Daughters of phi2 (K+ and K-)
            f"{p2[0]}kp_px, {p2[0]}kp_py, {p2[0]}kp_pz, {p2[0]}kp_E, "
            f"{p2[0]}km_px, {p2[0]}km_py, {p2[0]}km_pz, {p2[0]}km_E)"
        )
        
        # Extract individual components
        new_df = new_df.Define(f"helicity_angle_{p1[0]}", f"helicity_result_{p1[0]}_{p2[0]}.helicity_angle_1")
        new_df = new_df.Define(f"helicity_angle_{p2[0]}", f"helicity_result_{p1[0]}_{p2[0]}.helicity_angle_2")
        new_df = new_df.Define(f"decay_plane_angle", f"helicity_result_{p1[0]}_{p2[0]}.decay_plane_angle")
        new_df = new_df.Define(f"opening_angle_{p1[0]}_{p2[0]}", f"helicity_result_{p1[0]}_{p2[0]}.opening_angle")
        new_df = new_df.Define(f"PHI_{p1[0]}_{p2[0]}", f"helicity_result_{p1[0]}_{p2[0]}.azimuthal_phi")

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
        
        # Create a temporary column for composite key
        df_with_key = input_df.Define(
            "__composite_key__", 
            "std::to_string(__experiment__) + \"_\" + std::to_string(__run__) + \"_\" + std::to_string(__event__)"
        ).Define("__entry__", "rdfentry_")
        
        # Define C++ helper functions for efficient processing
        func_name = "ProcessBestCandidates"
        if func_name not in RDF_process._defined_functions:
            ROOT.gInterpreter.Declare("""
                #include <unordered_map>
                #include <vector>
                #include <string>
                #include <limits>
                
                class BestCandidateProcessor {
                private:
                    std::unordered_map<std::string, std::pair<double, std::vector<Long64_t>>> event_map;
                    
                public:
                    void ProcessEntry(const std::string& key, double value, Long64_t entry) {
                        auto it = event_map.find(key);
                        
                        if (it == event_map.end()) {
                            // First occurrence
                            event_map[key] = std::make_pair(value, std::vector<Long64_t>{entry});
                        } else {
                            double current_min = it->second.first;
                            
                            if (value < current_min) {
                                // Better candidate found
                                it->second.first = value;
                                it->second.second.clear();
                                it->second.second.push_back(entry);
                            } else if (value == current_min) {
                                // Same minimal value, add to list
                                it->second.second.push_back(entry);
                            }
                        }
                    }
                    
                    std::vector<Long64_t> GetBestEntries() const {
                        std::vector<Long64_t> result;
                        for (const auto& pair : event_map) {
                            const auto& entries = pair.second.second;
                            result.insert(result.end(), entries.begin(), entries.end());
                        }
                        return result;
                    }
                    
                    size_t GetUniqueEventCount() const {
                        return event_map.size();
                    }
                };
                
                // Global processor instance
                BestCandidateProcessor g_processor;
                
                void ResetProcessor() {
                    g_processor = BestCandidateProcessor();
                }
                
                void ProcessBestCandidates(const std::string& key, double value, Long64_t entry) {
                    g_processor.ProcessEntry(key, value, entry);
                }
                
                std::vector<Long64_t> GetBestEntries() {
                    return g_processor.GetBestEntries();
                }
                
                size_t GetUniqueEventCount() {
                    return g_processor.GetUniqueEventCount();
                }
            """)
            RDF_process._defined_functions.add(func_name)
        
        # Reset the processor
        ROOT.ResetProcessor()
        
        # Process entries using ROOT's Foreach
        print("Processing entries to find best candidates...")
        df_with_key.Foreach("ProcessBestCandidates(__composite_key__, {}, __entry__)".format(var))
        
        # Get results
        best_entries = ROOT.GetBestEntries()
        unique_count = ROOT.GetUniqueEventCount()
        
        print(f"Selected {len(best_entries)} best candidates")
        print(f"Number of unique events: {unique_count}")
        
        # Create filter function
        filter_func_name = "IsSelectedEntryMemEff"
        if filter_func_name not in RDF_process._defined_functions:
            # Convert to set for O(1) lookup
            entries_set = set(best_entries)
            entries_str = "{" + ", ".join(map(str, sorted(entries_set))) + "}"
            
            ROOT.gInterpreter.Declare(f"""
                #include <unordered_set>
                
                bool IsSelectedEntryMemEff(Long64_t entry) {{
                    static std::unordered_set<Long64_t> selected_entries = {entries_str};
                    return selected_entries.find(entry) != selected_entries.end();
                }}
            """)
            RDF_process._defined_functions.add(filter_func_name)
        
        # Apply filter
        result_df = df_with_key.Filter("IsSelectedEntryMemEff(__entry__)")
        result_df = result_df.Redefine("__candidate__", "1").Redefine("__ncandidates__", "1")
        
        return result_df

    def put_lineshape(df_mc:ROOT.RDataFrame,df_truth:ROOT.RDataFrame,
                      Func:Callable[[float], float]
                     ):
        """
        df_truth : ROOT.RDataFrame - Input RDataFrame with MC truth , should provide a flat shape mc 

        just for the ISRphiKK now.
        """
        random.seed()   

        event_set = set()
        selection_indices = []

        ROOT.ROOT.EnableImplicitMT()  # Enable multi-threading for RDataFrame
        num_threads = ROOT.ROOT.GetThreadPoolSize()
        print(f"Using {num_threads} threads for processing...")

        def get_selected_entries(entry, slot):
            i = getattr(entry, "rdfentry_")
            mass = getattr(entry, "mc_vpho_M")  # Make sure this column exists in df_truth
            f_M = Func(mass)
            dice = random.random()
            
            if f_M > dice:
                # Store event identifier as a tuple
                event_id = (getattr(entry, "__experiment__"), getattr(entry, "__run__"), getattr(entry, "__event__"))
                
                # Use a thread-safe approach to update shared collections
                with ROOT.std.mutex():  # This assumes ROOT provides a mutex for thread safety
                    event_set.add(event_id)
                    selection_indices.append(i)
                    
                # Optional: Print only every Nth event or for debugging
                if i % 1000 == 0:
                    print(f"Entry: {i} M_vpho: {mass} Func(M_vpho): {f_M} dice: {dice}")

        df_truth.ForeachSlot(get_selected_entries)

        def is_selected(exp, run, evt):
            return (exp, run, evt) in event_set

        df_mc = df_mc.Filter(is_selected, ["__experiment__", "__run__", "__event__"])

         

