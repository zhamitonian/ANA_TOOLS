#!/usr/bin/env basf2
# -*- coding: utf-8 -*-

###################################################
# Base class for Belle I and Belle II analyses
# Handles differences between Belle I and Belle II
###################################################

## version 3.0.0
## author: wangz
## date: 2026-06-25


import basf2 as b2
import modularAnalysis as ma
import variables.utils as vu
import variables.collections as vc
import os
import sys
from variables import variables as var
import string
import random
from .DecayString_parser import parse_decay_chain
from basf2 import B2INFO, B2ERROR, B2DEBUG, B2WARNING

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
        # Analysis mode
        self.analysis_mode = analysis_mode
        self._help_var_setting()
        
        # Initialization state
        self.has_initialized_conversion = False
        
        self.input_file: str = ""
        self.output_file: str = ""
        B2WARNING(f"BelleAnalysisBase initialized with belle_version: {self.belle_version}, analysis_mode: {self.analysis_mode}")
    
    def _help_var_setting(self):
        self.is_belle1 = (self.belle_version == 'belle1')
        self.is_signal_mc = (self.analysis_mode == 'sMC')
        self.is_generic_mc = (self.analysis_mode == 'gMC')
        self.is_MC = self.is_signal_mc or self.is_generic_mc


    @staticmethod
    def get_random_id(size=6, chars=string.ascii_uppercase + string.digits):
        """Generate random ID for unique particle list names"""
        return ''.join(random.choice(chars) for _ in range(size))
    

    def parse_arguments(self):
        """Parse positional steering arguments.

        Accepts up to four positional arguments in order:
            mode    : Analysis mode (data, sMC, gMC) -- optional
            version : Belle version (belle1, belle2) -- optional
            input   : input filename (optional)
            output  : output filename (optional)

        Missing values are left as the current object defaults.

        Returns:
            Tuple (belle_version, analysis_mode, input_file_or_None, output_file_or_None)
        """
        import argparse

        parser = argparse.ArgumentParser(add_help=True)
        parser.add_argument('mode', nargs='?', choices=['data', 'sMC', 'gMC'], default=None, help = "Analysis mode: data, sMC, gMC")
        parser.add_argument('version', nargs='?', choices=['belle1', 'belle2'], default=None, help = "Belle version: belle1, belle2")
        parser.add_argument('input', nargs='?', default=None, help = "Input filename")
        parser.add_argument('output', nargs='?', default=None, help = "Output filename")

        args = parser.parse_args()

        # If user didn't provide any positional args, return defaults
        if args.mode is None and args.version is None and args.input is None and args.output is None:
            return self.belle_version, self.analysis_mode, (self.input_file or None), (self.output_file or None)

        # Defaults from object
        belle_version = self.belle_version
        analysis_mode = self.analysis_mode
        input_file = self.input_file or None
        output_file = self.output_file or None

        # Update only if positional args were provided by the user
        if args.mode is not None:
            analysis_mode = args.mode
        if args.version is not None:
            belle_version = args.version
        if args.input is not None:
            input_file = args.input
        if args.output is not None:
            output_file = args.output

        # Store normalized values on the object
        self.belle_version = belle_version
        self.analysis_mode = analysis_mode
        self._help_var_setting()
        if input_file is not None:
            self.input_file = input_file
        if output_file is not None:
            self.output_file = output_file

        # Apply environment changes if needed
        B2WARNING(f"Updating BelleAnalysisBase to belle_version={self.belle_version}, analysis_mode={self.analysis_mode}")

        return belle_version, analysis_mode, input_file, output_file

    def setup_environment(self):
        """Set up environment variables and conditions for the appropriate Belle experiment"""
        if self.is_belle1:
            # Set Belle I specific environment variables
            os.environ["USE_GRAND_REPROCESS_DATA"] = "1"
            os.environ["PGUSER"] = "g0db"
            
            b2.conditions.disable_globaltag_replay()

            b2.conditions.globaltags=['B2BII','BellePID',
                                          'b2bii_beamParameters_with_smearing']
            if self.is_MC:
                b2.conditions.prepend_globaltag('B2BII_MC')
 
            b2.conditions.prepend_globaltag('Legacy_CollisionAxisCMS_Belle')
            b2.conditions.prepend_globaltag('analysis_b2bii')
            B2INFO("Environment set for Belle I analysis")
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
            # implementation for reconstruction, if need read generate level distribution, 
            # set applySkim = False
            #if self.is_signal_mc:
            convertBelleMdstToBelleIIMdst(
                input_file, 
                enableNisKsFinder=True,
                enableEvtcls=True, 
                HadronA=True, 
                HadronB=True, 
                applySkim=True,
                path=path
            )
#            else:
#                convertBelleMdstToBelleIIMdst(
#                    input_file, 
#                    applySkim=False, 
#                    useBelleDBServer=None, 
#                    convertBeamParameters=True,
#                    generatorLevelReconstruction=False, generatorLevelMCMatching=False, entrySequences=None,
#                    matchType2E9oE25Threshold=-1.1, enableNisKsFinder=True, HadronA=True, HadronB=True,
#                    enableRecTrg=False, enableEvtcls=True, SmearTrack=2, enableLocalDB=True,
#                    path=path)
#           
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
        var.addAlias('cosTheta_CMS', 'useCMSFrame(cosTheta)')
        var.addAlias('theta_CMS', 'useCMSFrame(theta)')
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
        if output_file == None:
            output_file = self.output_file
        
        # Create particle info objects with full list names
        particles = parse_decay_chain(decay_chain)
        firstP = particles[0]

        # Fill particle lists from MC
        for p in particles:
            if len(p.daughters) ==0:
                #ma.fillParticleListFromMC(p.list_name, f'genMotherPDG == {p.mother.pdg} and mcPrimary>0', path=path)
                ma.fillParticleListFromMC(p.list_name, 'mcPrimary>0', path=path)

        # Reconstruct decay chain (bottom-up)
        for p in reversed(particles):
            if p.get_decay_string():
                B2WARNING(f"MC truth reconstructing decay: {p.get_decay_string()}")
                ma.reconstructDecay(p.get_decay_string(), ' ', path=path)

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
            p.variables = vu.create_aliases(['E_CMS', 'p_CMS', 'cosTheta_CMS', 'phi_CMS', 'px_CMS', 'py_CMS', 'pz_CMS'], firstP.get_daughter_access_string(p,"{variable}"),p.prefix)
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


"""
version history:
v2.0.0
- 2025.12.16

v3.0.0
- 2026-06-25
- remove save_variables_before_fitting_general and standard variable collection
- optimize argument parsing  
- move class MC particle to DecatString_parser.py

core principles: this is just a helper class, physics analysis part shall not be included here


"""