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