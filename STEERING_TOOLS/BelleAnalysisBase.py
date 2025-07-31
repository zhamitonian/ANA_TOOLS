#!/usr/bin/env basf2
# -*- coding: utf-8 -*-

###################################################
# Base class for Belle I and Belle II analyses
# Handles differences between Belle I and Belle II
###################################################

import basf2 as b2
import modularAnalysis as ma
import variables.utils as vu
import variables.collections as vc
import os
import sys
from variables import variables as var
import string
import random

class BelleAnalysisBase:
    """
    Base class for Belle I and Belle II analyses.
    Manages differences between experiments and provides common functionality.
    """
    
    def __init__(self, belle_version='belle2', analysis_mode='data'):
        """
        Initialize the Belle analysis base
        
        Args:
            belle_version: Belle experiment version ('belle1' or 'belle2')
            analysis_mode: Analysis mode ('data', 'sMC', or 'gMC')
        """
        # Belle experiment version
        self.belle_version = belle_version
        self.is_belle1 = (belle_version == 'belle1')
        
        # Analysis mode
        self.analysis_mode = analysis_mode
        self.is_signal_mc = analysis_mode == 'sMC'
        self.is_generic_mc = analysis_mode == 'gMC'
        
        # Initialization state
        self.has_initialized_conversion = False
        
        # Command line parameters
        self.input_file = None
        self.output_file = None
        self.is_test_mode = False
        self.command_line_options = {}
    
    @staticmethod
    def get_random_id(size=6, chars=string.ascii_uppercase + string.digits):
        """Generate random ID for unique particle list names"""
        return ''.join(random.choice(chars) for _ in range(size))
    
    def parse_arguments(self, args=None, default_test_files=None):
        """
        Parse command line arguments
        
        Args:
            args: List of command line arguments (defaults to sys.argv[1:])
            default_test_files: Dictionary mapping modes to default test files
        
        Returns:
            Tuple of (belle_version, analysis_mode, input_file, output_file)
        """
        if args is None:
            args = sys.argv[1:]
            
        if default_test_files is None:
            default_test_files = {
                'belle1_sMC': "/group/belle2/users2022/wangz/data_gMC_belle1/MC_tagged_ISRphiKK_16Mar25/sample.root",
                'belle2_sMC': "/group/belle2/users2022/wangz/SignalMC/ISR_phiKK/mdst/sample_20250228203731_15.root",
            }

        def print_usage():
            print("Usage: basf2 analysis.py [belle_version] [analysis_mode] [input_file] [output_file] [options]")
            print("  belle_version: belle1 or belle2")
            print("  analysis_mode: data, sMC, or gMC")
            print("  input_file: Path to input mdst file")
            print("  output_file: Path to output root file")
            print("Options:")
            print("  --test: Use default test files instead of specified input/output")
            sys.exit(1)
        
        # Check for minimum number of arguments
        if len(args) < 4:
            print_usage()

        # Get Belle version and analysis mode from arguments
        belle_version = args[0]
        if belle_version not in ['belle1', 'belle2']:
            print(f"Error: Invalid Belle version '{belle_version}'. Must be 'belle1' or 'belle2'")
            print_usage()
            
        analysis_mode = args[1]
        if analysis_mode not in ['data', 'sMC', 'gMC']:
            print(f"Error: Invalid analysis mode '{analysis_mode}'. Must be 'data', 'sMC', or 'gMC'")
            print_usage()

        # Get input and output files from command line
        input_file = args[2]
        output_file = args[3]

        # Check for test flag
        is_test_mode = "--test" in args

        # In test mode, override input/output with defaults
        if is_test_mode:
            combined_mode = f"{belle_version}_{analysis_mode}"
            if combined_mode in default_test_files:
                input_file = default_test_files[combined_mode]
            output_file = "test.root"
        
        # Store values
        self.belle_version = belle_version
        self.is_belle1 = (belle_version == 'belle1')
        self.analysis_mode = analysis_mode
        self.is_signal_mc = analysis_mode == 'sMC'
        self.is_generic_mc = analysis_mode == 'gMC'
        self.input_file = input_file
        self.output_file = output_file
        self.is_test_mode = is_test_mode
        
        # Parse additional options
        for arg in args[4:]:
            if arg.startswith('--'):
                option = arg[2:]
                if '=' in option:
                    key, value = option.split('=', 1)
                    self.command_line_options[key] = value
                else:
                    self.command_line_options[option] = True
                    
        return belle_version, analysis_mode, input_file, output_file
    
    def setup_environment(self):
        """Set up environment variables and conditions for the appropriate Belle experiment"""
        if self.is_belle1:
            # Set Belle I specific environment variables
            os.environ["USE_GRAND_REPROCESS_DATA"] = "1"
            os.environ["PGUSER"] = "g0db"
            # Set Belle II conditions for Belle I data
            b2.conditions.globaltags = ['B2BII']
            return True
        return False
    
    def setup_IO(self, path, input_file=None,output_file=None):
        """
        Set up input handling for the appropriate Belle experiment
        
        Args:
            path: Analysis path to add modules to
            input_file: Path to input file (defaults to self.input_file)
            
        Returns:
            True if Belle I input was set up, False otherwise
        """
        if output_file is not None:
            self.output_file = output_file

        if input_file is None:
            input_file = self.input_file
            
        if self.is_belle1 and not self.has_initialized_conversion:
            # Import needed modules for conversion
            from b2biiConversion import convertBelleMdstToBelleIIMdst
            
            # Use Belle I to Belle II conversion
            convertBelleMdstToBelleIIMdst(
                input_file, 
                enableNisKsFinder=False, 
                enableEvtcls=True, 
                HadronA=False, 
                HadronB=False, 
                path=path
            )
            
            self.has_initialized_conversion = True
            return True
        elif not self.has_initialized_conversion:
            # Standard Belle II input
            ma.inputMdst(environmentType='default', filename=input_file, path=path)
            self.has_initialized_conversion = True
            return False
            
        return self.is_belle1
    
        
    def setup_common_aliases(self):
        """Set up common aliases used for all Belle analyses"""
        # CMS frame kinematics
        var.addAlias('p_CMS', 'useCMSFrame(p)')
        var.addAlias('E_CMS', 'useCMSFrame(E)')
        var.addAlias('pt_CMS', 'useCMSFrame(pt)')
        var.addAlias('px_CMS', 'useCMSFrame(px)')
        var.addAlias('py_CMS', 'useCMSFrame(py)')
        var.addAlias('pz_CMS', 'useCMSFrame(pz)')
        var.addAlias('theta_CMS', 'useCMSFrame(cosTheta)')
        var.addAlias('phi_CMS', 'useCMSFrame(phi)')
        
        # Useful aliases for both Belle I and Belle II
        var.addAlias('clusterE_NaN', 'ifNANgiveX(clusterE, -1)')
        var.addAlias('EoverP', 'formula(ifNANgiveX(clusterE, -1) / p)')
        var.addAlias('ROE_extraE', 'roeEextra(goodGamma)')
        var.addAlias('ROE_nTracks', 'nROE_Charged(goodGamma)')
        
        # Angular variables
        var.addAlias('daughterAngle_CMS', 'useCMSFrame(daughterAngle(0,1))')
        var.addAlias('daughterAngle_LAB', 'daughterAngle(0,1)')
        var.addAlias('RECM', 'formula((E_CMS**2 - p_CMS**2)**0.5)')
        var.addAlias('REC2M','formula(E_CMS**2 - p_CMS**2)')
        var.addAlias('Umiss', 'formula(E_CMS - p_CMS)')

        #
        var.addAlias('isSig',"ifNANgiveX(isSignal, -1)")

        # PID variables - different for Belle I vs Belle II
        if self.is_belle1:
            var.addAlias('Lkpi', 'atcPIDBelle(3,2)')
            var.addAlias('Lppi', 'atcPIDBelle(4,2)')
            var.addAlias('Lpk', 'atcPIDBelle(4,3)')
            var.addAlias('Lke', 'atcPIDBelle(3,0)')
            var.addAlias('Lpie', 'atcPIDBelle(2,0)')
        else:
            var.addAlias('Lkpi', 'binaryPID(321,211)')
            var.addAlias('Lppi', 'binaryPID(2212,211)')
            var.addAlias('Lpk', 'binaryPID(2212,321)')
            var.addAlias('Lke', 'binaryPID(321,11)')
            var.addAlias('Lpie', 'binaryPID(211,11)')
        
        # Trigger
        var.addAlias('hieftdl', 'L1FTDL(hie)')
        var.addAlias('hiepsnm', 'L1PSNM(hie)')
        var.addAlias('hiescale', 'L1Prescale(hie)')
        var.addAlias('sttftdl', 'L1FTDL(stt)')
        var.addAlias('sttpsnm', 'L1PSNM(stt)')
        var.addAlias('sttscale', 'L1Prescale(stt)')


    def setup_mc_truth_general(self, path, decay_chain , output_file=None):
        """
        Set up MC truth information with configurable decay structure
        
        Args:
            path: Analysis path to add modules to
            output_file: Output file path for ntuple
            decay_structure: Dictionary with decay structure information (optional)
                Default structure is for phi K K analysis
        
        Returns:
            List of truth variable names
        """
        #MCParticleInfo = BelleAnalysisBase.MCParticleInfo

        if output_file == None:
            output_file = self.output_file
        
        # Create particle info objects with full list names
        particles = self.parse_decay_chain(decay_chain)
        firstP = particles[0]

        # Fill particle lists from MC
        for p in particles:
            if len(p.daughters) ==0:
                #ma.fillParticleListFromMC(p.list_name, f'genMotherPDG == {p.mother.pdg} and mcPrimary>0', path=path)
                ma.fillParticleListFromMC(p.list_name, 'mcPrimary>0', path=path)

        # Reconstruct decay chain (bottom-up)
        for p in reversed(particles):
            if p.get_decay_string():
                ma.reconstructDecay(p.get_decay_string(), ' ', path=path)
                #if p.name == "vpho":
                    #ma.reconstructDecay(p.get_decay_string(), ' ', path=path)
                #else:
                    #ma.reconstructMCDecay(p.get_decay_string(), ' ', path=path)

        # use dM to avoid miscombination
        dM_sum = ""
        for p in particles:
            if p.name != "vpho" : 
                dM_sum += firstP.get_daughter_access_string(p, "abs(dM)") 
                dM_sum += " + "
        dM_sum = dM_sum[:-2]  # Remove last " + "

        var.addAlias('dM_sum', dM_sum)

        ma.rankByLowest(firstP.list_name, 'dM_sum', numBest= 1 , path=path)

        # Add daughter variables
        for p in particles:
            p.variables = vu.create_aliases(['E_CMS', 'pt_CMS' ,'E', 'p', 'px', 'py', 'pz' , 'M' ,'theta'], firstP.get_daughter_access_string(p,"{variable}"),p.prefix)
            if p.name == "vpho":
                p.variables += vu.create_aliases(['m2Recoil', 'pRecoilTheta'], firstP.get_daughter_access_string(p,"{variable}"),p.prefix)
            if p.name in ["K+", "K-", "pi+", "pi-"]:
                p.variables += vu.create_aliases(['dr','dz' ], firstP.get_daughter_access_string(p,"{variable}"),p.prefix)
            

        # Combine all truth variables
        truth_vars =  []
        for p in particles:
            truth_vars += p.variables
        
        # Save to event extra info
        for var_name in truth_vars:
            ma.variablesToEventExtraInfo(firstP.list_name, variables={var_name: var_name}, path=path)
        
        ma.variablesToNtuple(firstP.list_name, truth_vars, filename=output_file, treename='truth', path=path)
        
        return truth_vars

    def get_standard_variable_collections(self):
        """
        Get standard collections of variables for ISR analyses
        
        Returns:
            Dictionary of variable collections
        """
        # Event variables
        event_vars = ["hieftdl", "hiepsnm", "hiescale", "sttftdl", "sttpsnm", "sttscale"] + ["beamE", "beamPx", "beamPy", "beamPz"]
        
        # Photon and kaon variables
        CMS_kinematics = ['E_CMS', 'p_CMS', "px_CMS", "py_CMS", "pz_CMS", "pt_CMS","theta_CMS", "phi_CMS"]
        gam_vars = vc.kinematics +  CMS_kinematics + ['phi', 'theta','M'] 
        kaon_vars = vc.kinematics + CMS_kinematics + ['phi', 'theta','M'] + ['Lkpi', 'Lpk','kaonID','Lke']  + ['nCDCHits', 'nPXDHits', 'nSVDHits']+ ['dr', 'dz']
        pion_vars = vc.kinematics + CMS_kinematics + ['phi', 'theta','M'] + ['Lkpi', 'Lppi','pionID','Lpie'] + ['nCDCHits', 'nPXDHits', 'nSVDHits']  + ['dr', 'dz'] 
        proton_vars = vc.kinematics + CMS_kinematics + ['phi', 'theta','M'] + [ 'Lpk','Lppi','protonID'] + ['nCDCHits' , 'nPXDHits', 'nSVDHits'] + ['dr', 'dz']
        reso_vars = vc.kinematics + CMS_kinematics + ['M']
        vpho_vars = vc.kinematics + CMS_kinematics + ['M','m2Recoil','pRecoilTheta']  + vc.recoil_kinematics
        
        # Full collection
        return {
            'event': event_vars,
            'gamma': gam_vars ,
            'kaon': kaon_vars ,
            'resonance': reso_vars,
            'vpho': vpho_vars,
            'pion': pion_vars,
            'proton': proton_vars,
            'vertex': ["chiProb","Chi2"]
            }
    
    def save_variables_before_fitting_general(self,decay_chain,path):
        """
        Save variables before fitting for later comparison
        
        Args:
            path: Analysis path to add modules to
        
        Returns:
            List of before-fit variable aliases
        """
        #MCParticleInfo = BelleAnalysisBase.MCParticleInfo

        particles = self.parse_decay_chain(decay_chain)

        firstP = particles[0]

        # Build event kinematics
        ma.buildEventKinematics(inputListNames=firstP.list_name , path=path)
        
        # Create dictionary of variables to save
        var_dict = {}

        for p in particles:
            var_dict.update({
                firstP.get_daughter_access_string(p,f"{var}"):f"{p.prefix}_bf_{var}"
                for var in ["E","theta","px","py","pz"]
            })

        # Save system variables
        #var_dict.update({firstP.get_daughter_access_string(vpho,f"{var}"): f"sys_{var}" for var in vc.recoil_kinematics + ["m2Recoil"]})
        var_dict.update({var: f"sys_{var}" for var in ["visibleEnergyOfEventCMS"]} ) 

        # Save variables to extra info
        ma.variablesToExtraInfo(firstP.list_name, variables=var_dict, path=path)
        
        # Create aliases for before-fit variables
        bf_fit_vars = vu.create_aliases(list(var_dict.values()), 'extraInfo({variable})', '')
        
        return bf_fit_vars


    @staticmethod
    def get_pdg_code(particle_name):
        """
        Returns the PDG code for a given particle name.
        
        Args:
            particle_name: String name of the particle (e.g., 'K+', 'phi')
            
        Returns:
            int: PDG code for the particle, or None if not found
        """
        pdg_map = {
            # Leptons
            'e-': 11,
            'e+': -11,
            'mu-': 13,
            'mu+': -13,
            'tau-': 15,
            'tau+': -15,
            'nu_e': 12,
            'nu_mu': 14,
            'nu_tau': 16,
            'anti_nu_e': -12,
            'anti_nu_mu': -14,
            'anti_nu_tau': -16,
            
            # Photons and virtual particles
            'gamma': 22,
            'vpho': 10022,  # Virtual photon
            
            # Mesons
            'pi0': 111,
            'pi+': 211,
            'pi-': -211,
            'rho0': 113,
            'rho+': 213,
            'rho-': -213,
            'eta': 221,
            'omega': 223,
            'phi': 333,
            'K+': 321,
            'K-': -321,
            'K0': 311,
            'anti-K0': -311,
            'K*0': 313,
            'anti-K*0': -313,
            'K_S0': 310,
            'K_L0': 130,
            'D+': 411,
            'D-': -411,
            'D0': 421,
            'anti-D0': -421,
            'D_s+': 431,
            'D_s-': -431,
            'J/psi': 443,
            'psi(2S)': 100443,
            'Upsilon(1S)': 553,
            'Upsilon(2S)': 100553,
            'Upsilon(3S)': 200553,
            
            # Baryons
            'p': 2212,
            'anti-p': -2212,
            'n': 2112,
            'anti-n': -2112,
            'Lambda0': 3122,
            'anti-Lambda0': -3122,
            'Sigma+': 3222,
            'Sigma0': 3212,
            'Sigma-': 3112,
            'Xi0': 3322,
            'Xi-': 3312,
            'Omega-': 3334,
            
            # B mesons
            'B+': 521,
            'B-': -521,
            'B0': 511,
            'anti-B0': -511,
            'B_s0': 531,
            'anti-B_s0': -531,
            
            # Quarkonia and exotic states
            'chi_c0': 10441,
            'chi_c1': 20443,
            'chi_c2': 445,
            'chi_b0': 10551,
            'chi_b1': 20553,
            'chi_b2': 555,
            'X(3872)': 9120443,
            'Z_c(3900)+': 9940213,
            'Z_c(3900)-': -9940213
        }
        
        # Also handle alternative naming conventions
        if particle_name not in pdg_map:
            # Try with/without charge signs
            baseless_name = particle_name.rstrip('+-0')
            if f"{baseless_name}+" in pdg_map and particle_name.endswith('-'):
                return -pdg_map[f"{baseless_name}+"]
            elif f"{baseless_name}+" in pdg_map and particle_name.endswith('+'):
                return pdg_map[f"{baseless_name}+"]
                
            # Handle anti-particle naming
            if particle_name.startswith(('anti-', 'anti_')):
                base_name = particle_name[5:]
                if base_name in pdg_map:
                    return -pdg_map[base_name]
        
        return pdg_map.get(particle_name)


    class MCParticleInfo:
        """
        Class to hold information about a particle in the MC truth decay chain.
        """
        def __init__(self, list_name , mother=None, daughters=None):
            """
            Initialize MC particle information.
            
            Args:
                list_name: Particle list name (e.g., 'phi:mc_phi', 'K+:mc_phikp')
                pdg: PDG code of the particle
                mother: Parent particle info object
                daughters: List of daughter particle info objects
            """
            self.list_name = list_name
            self.mother = mother
            self.daughters = daughters or []
            self.variables = []
            
            # Parse the list name to extract type and prefix
            parts = list_name.split(':')
            self.name = parts[0]  # Particle type (e.g., 'phi', 'K+')
            self.prefix = parts[1]  # List prefix (e.g., 'mc_phi')

            self.pdg = BelleAnalysisBase.get_pdg_code(f"{self.name}")
        
        def add_daughter(self, daughter):
            """Add a daughter particle"""
            self.daughters.append(daughter)
            daughter.mother = self
        
        def get_path_to_ancestor(self,ancestor):
            """
            Get the path from this particle to an ancestor particle.
            Returns a list of indices representing daughter indices from ancestor down to self.
        
            Args:
                ancestor: Ancestor MCParticleInfo object
            
            Returns:
                List of indices if ancestor is found in the lineage,
                None if ancestor is not an ancestor of this particle
            """
            if self == ancestor:
                return []
            
            if self.mother is None:
                return None
            
            if self.mother == ancestor:
                for i , sibling in enumerate(self.mother.daughters):
                    if sibling == self:
                        return [i]
           # Recursively check 
            parent_path = self.mother.get_path_to_ancestor(ancestor)
            if parent_path is not None:
                for i , sibling in enumerate(self.mother.daughters):
                    if sibling == self:
                        return  parent_path + [i]

        def get_daughter_access_string(self,descendant,var_name):
            """
            Get the daughter access string based on a path of indices.
        
            Args:
                var_name: Variable name to include at the end
            
            Returns:
                String with basf2 daughter access syntax
            """
            path = descendant.get_path_to_ancestor(self)

            if path is None or len(path) == 0:
                return var_name
            
            access_str = ""
            for idx in path:
                access_str += f"daughter({idx},"

            access_str += var_name
            access_str += ")" * len(path)

            return access_str
        
        def get_decay_string(self):
            """Generate decay string for reconstruction"""
            if not self.daughters:
                return None
            
            daughter_lists = " ".join(d.list_name for d in self.daughters)
            return f"{self.list_name} -> {daughter_lists}"


    def parse_decay_chain(self, decay_string):
        """
        Parse a decay chain string with nested structure and create MCParticleInfo objects.
        
        Args:
            decay_string: String representation of a decay chain with nested structure.
                Format: [A:list_A -> [B:list_B -> C:list_C D:list_D] E:list_E]
                Where brackets indicate particles and their decays, and -> separates
                parent from children.
                
        Returns:
            List of MCParticleInfo objects in the decay chain, with the first being the root.
        """
        # Dictionary to store all created MCParticleInfo objects by list_name
        particles = {}
        
        def parse_decay(decay_str, parent=None):
            """
            Recursively parse a decay string and create MCParticleInfo objects
            
            Args:
                decay_str: Decay string to parse
                parent: Parent MCParticleInfo object (if any)
                
            Returns:
                MCParticleInfo object for the parent particle of this decay
            """
            # Remove outer brackets if present
            decay_str = decay_str.strip()
            if decay_str.startswith('[') and decay_str.endswith(']'):
                decay_str = decay_str[1:-1].strip()
            
            # Split into parent and children parts
            if '->' in decay_str:
                parent_str, children_str = [s.strip() for s in decay_str.split('->', 1)]
                
                # Create parent particle
                parent_full_name = parent_str.strip()
                
                if parent_full_name not in particles:
                    particles[parent_full_name] = self.MCParticleInfo(parent_full_name) 
                current_parent = particles[parent_full_name]
                
                # If this particle already has a different parent, warn about it
                if parent and current_parent.mother and current_parent.mother != parent:
                    print(f"Warning: {parent_full_name} already has parent {current_parent.mother.list_name}")
                
                # Connect to parent if provided
                if parent and current_parent.mother is None:
                    parent.add_daughter(current_parent)
                
                # Parse children with bracket awareness
                children = []
                current_child = ""
                bracket_level = 0
                
                for char in children_str:
                    if char == '[':
                        bracket_level += 1
                        current_child += char
                    elif char == ']':
                        bracket_level -= 1
                        current_child += char
                    elif char.isspace() and bracket_level == 0:
                        # Space outside brackets separates children
                        if current_child.strip():
                            children.append(current_child.strip())
                        current_child = ""
                    else:
                        current_child += char
                
                # Add the last child if present
                if current_child.strip():
                    children.append(current_child.strip())
                
                # Process each child
                for child in children:
                    if '[' in child:
                        # This is a nested decay
                        parse_decay(child, current_parent)
                    else:
                        # This is a simple particle
                        child_full_name = child.strip()
                        
                        if child_full_name not in particles:
                            particles[child_full_name] = self.MCParticleInfo(child_full_name, mother=current_parent)
                        
                        current_parent.add_daughter(particles[child_full_name])
                
                return current_parent
            else:
                # Single particle without decay
                parent_full_name = decay_str.strip()
                
                if parent_full_name not in particles:
                    particles[parent_full_name] = self.MCParticleInfo(parent_full_name, mother=parent)
                
                if parent:
                    parent.add_daughter(particles[parent_full_name])
                
                return particles[parent_full_name]
        
        # Parse the full decay chain
        root = parse_decay(decay_string)
        
        # Return all particles with root first
        all_particles = [root]
        for particle in particles.values():
            if particle != root:
                all_particles.append(particle)
        
        return all_particles