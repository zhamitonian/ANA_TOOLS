#!/usr/bin/env basf2
# -*- coding: utf-8 -*-

###################################################
# Utility classes for ISR analyses
# Provides reusable tools for different ISR analyses
###################################################

import basf2 as b2
import modularAnalysis as ma
import variables.collections as vc
import variables.utils as vu
from variables import variables as var
from kinfit import fitKinematic3C
import vertex
import sys
import os

# Import the base class
from .BelleAnalysisBase import BelleAnalysisBase

class ISRAnalysisTools(BelleAnalysisBase):
    """
    Utility class providing tools for ISR analyses.
    Handles common operations like pi0 veto, variable saving, etc.
    Inherits from BelleAnalysisBase to handle Belle I/II differences.
    """
    def __init__(self, mode='tagged', belle_version='belle2',analysis_mode='data'):
        """
        Initialize ISR analysis tools
        
        Args:
            mode: ISR analysis mode ('tagged' or 'untagged')
            belle_version: Belle version ('belle1' or 'belle2')
        """
        super().__init__(belle_version=belle_version,analysis_mode=analysis_mode)
        self.isr_mode = mode
        print(f"ISRAnalysisTools initialized with mode: {self.isr_mode}, belle_version: {self.belle_version}, analysis_mode: {self.analysis_mode}")

    def get_extra_gammaE(self, particle_list, path):
        ma.buildRestOfEvent(target_list_name=particle_list, path=path)

        roe_gamma = b2.create_path()
        photonList = 'gamma:extra_gamma' + self.get_random_id()
        ma.fillParticleList(decayString=photonList,
                    cut=f'[isInRestOfEvent == 1]',
                    path=roe_gamma)


        var.addAlias('extra_clusterE', f'totalEnergyOfParticlesInList({photonList})')

        ma.rankByLowest(particleList=photonList,
                        variable='E_CMS',
                        numBest=1,
                        path=roe_gamma)

        ma.variableToSignalSideExtraInfo(
            photonList,
            varToExtraInfo={"E_CMS":"extra_gamE"},
            path=roe_gamma
        )
        ma.variableToSignalSideExtraInfo(
            particle_list,
            varToExtraInfo={"extra_clusterE":"extra_ClusterE"},
            path=roe_gamma
        )

        path.for_each('RestOfEvent', 'RestOfEvents',roe_gamma)


    def write_pi0_veto(self, particle_list, photon_selection, path, rank_use = None , kinfit = False):
        """
        Add pi0 veto variables to the event extraInfo field.
        
        Args:
            particle_list: Target particle list to veto against
            photon_selection: Photon selection criteria
            path: Analysis path to add modules to
        """
        self.rank_use = rank_use if rank_use else 3
        self.kinfit = kinfit

        # Build rest of event
        ma.buildRestOfEvent(target_list_name=particle_list, path=path)
        
        # Create new path for ROE with a unique name
       # roe_path = b2.create_path(f"roe_path_{self.get_random_id()}")
        roe_path = b2.create_path()
        
        # Get random list names
        random_posfix = self.get_random_id()
        signalList = particle_list + random_posfix
        photonList = 'gamma:writePi0Veto' + random_posfix
        highEPhotonList = 'gamma:highE' + random_posfix
        otherPhotonList = 'gamma:other' + random_posfix
        pi0List_withHighE = 'pi0:withHighE' + random_posfix
        pi0List_general = 'pi0:general' + random_posfix
        # use for chi squared, only for tagged mode  
        vphoList_noISR = 'vpho:noISR' + random_posfix
        vphoList_alpha = 'vpho:alpha' + random_posfix # reconstruct with particles_list(a vpho)  pi0
        vphoList_beta = 'vpho:beta' + random_posfix  # reconstruct with vpho:noISR pi0
        
        # Fill signal side particle list
        ma.fillSignalSideParticleList(outputListName=signalList,
                                      decayString=f'^{particle_list}',
                                      path=roe_path)
        
        # Handle high energy photon differently based on ISR mode
        if self.isr_mode == 'tagged':
            # For tagged mode, use the ISR photon from the decay chain
            ma.fillSignalSideParticleList(
                outputListName=highEPhotonList,
                decayString=f'{particle_list} -> ^gamma vpho',
                path=roe_path
            )
            if kinfit:
                ma.fillSignalSideParticleList(
                    outputListName=vphoList_noISR,
                    decayString=f'{particle_list} -> gamma ^vpho',
                    path=roe_path
                )
        
        # Fill photon list with ROE photons
        ma.fillParticleList(decayString=photonList,
                            cut=f'[isInRestOfEvent == 1] and {photon_selection}',
                            path=roe_path)
        
        # For untagged mode, find highest energy photon from ROE
        if self.isr_mode == 'untagged':
            ma.rankByHighest(particleList=photonList,
                             variable='E_CMS', 
                             numBest=0,
                             path=roe_path)
                            
            var.addAlias('E_rank','extraInfo(E_CMS_rank)')
            ma.cutAndCopyList(highEPhotonList, photonList, 'E_rank == 1', path=roe_path)
            ma.cutAndCopyList(otherPhotonList, photonList, 'E_rank != 1', path=roe_path)
        
        # Reconstruct pi0s
        ma.reconstructDecay(decayString=f'{pi0List_withHighE} -> {highEPhotonList} {otherPhotonList if self.isr_mode == "untagged" else photonList}',
                            cut='0.080 < M < 0.200',
                            path=roe_path)
                            
        # General pi0 reconstruction
        pi0_cuts = {'eff60':'0.03<InvM ',
                    'eff50':'0.03<InvM ',
                    'eff40':'0.03<InvM ',
                    'eff30':'0.03<InvM and -1.5<daughterDiffOf(0,1,phi)<1.5 and daughterAngle(0,1)<1.4',
                    'eff20':'0.03<InvM and -1.0<daughterDiffOf(0,1,phi)<1.0 and daughterAngle(0,1)<0.9',  
                    'eff10':'0.03<InvM and -0.9<daughterDiffOf(0,1,phi)<0.9 and daughterAngle(0,1)<0.8'}
   

        ma.reconstructDecay(decayString=f'{pi0List_general} -> {photonList} {photonList}',
                            cut=pi0_cuts[self.pi0_veto_cut],
                            path=roe_path)
        
        # Rank by mass difference
        var.addAlias('abs_dM','abs(dM)')
        ma.rankByLowest(particleList=pi0List_withHighE,
                        variable='abs_dM',
                        numBest=1,
                        path=roe_path)
        
        # Number of ranks depends on mode
        ma.rankByLowest(particleList=pi0List_general,
                        variable='abs_dM',
                        numBest=0,
                        path=roe_path)

        var.addAlias('dM_rank', 'extraInfo(abs_dM_rank)')
        
        # Create and fill rank-specific lists
        pi0_rank_lists = []
        for i in range(1, self.rank_use + 1):
            list_name = f'{pi0List_general}_rank{i}'
            pi0_rank_lists.append(list_name)
            ma.cutAndCopyList(list_name, pi0List_general, f'dM_rank == {i}', path=roe_path)
        
        # Define variable mappings for pi0 veto
        pi0_variable_mapping = {
            'M': 'M',
            'daughter(0,p_CMS)': 'p0',
            'daughter(1,p_CMS)': 'p1',
            'daughter(0,E_CMS)': 'E0',
            'daughter(1,E_CMS)': 'E1',
            'daughter(0,theta_CMS)': 'theta0',
            'daughter(1,theta_CMS)': 'theta1',
            'dM': 'dM'
        }

        # Create varToExtraInfo dictionaries
        def create_extrainfo_dict(prefix, var_mapping):
            return {var: f'pi0veto_{prefix}_{name}_{self.pi0_veto_cut}' 
                    for var, name in var_mapping.items()}

        # Save variables to signal side extra info
        ma.variableToSignalSideExtraInfo(
            pi0List_withHighE,
            varToExtraInfo=create_extrainfo_dict('highE', pi0_variable_mapping),
            path=roe_path
        )

        # Save ranked pi0 info
        for i, list_name in enumerate(pi0_rank_lists, 1):
            ma.variableToSignalSideExtraInfo(
                list_name,
                varToExtraInfo=create_extrainfo_dict(f'rank{i}', pi0_variable_mapping),
                path=roe_path
            )

        # Add pi0 count
        mass_window = {'eff10': '0.127<InvM<0.139', 'eff20': '0.121<InvM<0.142', 'eff30': '0.120<InvM<0.145',
                       'eff40':'0.120<InvM<0.145','eff50':'0.105<InvM<0.150','eff60':'0.03<InvM'}
        
        pi0List_mass = f'{pi0List_general}_mass'
        ma.cutAndCopyList(pi0List_mass, pi0List_general, mass_window[self.pi0_veto_cut], path=roe_path)

        ma.variableToSignalSideExtraInfo(
            particle_list, # not use pi0List_general here
            varToExtraInfo={f'nParticlesInList({pi0List_mass})': f'pi0veto_count_{self.pi0_veto_cut}'},
            path=roe_path
        ) 
        
        # use pi0 to perform kinematic fit
        if self.isr_mode == 'tagged' and kinfit:
            ma.reconstructDecay(f'{vphoList_alpha} -> {highEPhotonList} {vphoList_noISR} {pi0List_mass}','', path=roe_path)
            fitKinematic3C(f'{vphoList_alpha}', path=roe_path)
            ma.rankByLowest(vphoList_alpha, 'extraInfo(OrcaKinFitChi2)', 1, path=roe_path)
            ma.variableToSignalSideExtraInfo(
            vphoList_alpha,
            varToExtraInfo={f'ifNANgiveX(extraInfo(OrcaKinFitChi2), 99999)': f'chisq_alpha_{self.pi0_veto_cut}'},
            path=roe_path
            ) 

            '''
            ma.reconstructDecay(f'{vphoList_beta} -> {vphoList_noISR} {pi0List_general}','', path=roe_path)
            vertex.kFit(f'{vphoList_beta}', conf_level = 0.0, fit_type='fourC', daughtersUpdate=True, path =roe_path)
            ma.rankByLowest(vphoList_beta, 'extraInfo(chiSquared)', 1, path=roe_path)
            ma.variableToSignalSideExtraInfo(
            vphoList_beta,
            varToExtraInfo={f'ifNANgiveX(extraInfo(chiSquared), 99999)': f'chisq_beta_{self.pi0_veto_cut}'},
            path=roe_path
            ) 
            '''


        # Execute the ROE path to ensure variables are properly stored
        path.for_each('RestOfEvent', 'RestOfEvents', roe_path)

    
    def setup_pi0_veto_aliases(self):
        """
        Set up aliases for pi0 veto variables
        
        Args:
            rank_count: Number of ranks to create aliases for (defaults to mode-specific value)
        
        Returns:
            List of pi0 veto variable names
        """
            
        # Define pi0 veto variables to be aliased
        pi0_veto_aliases = {
            'highE': ['M', 'p0', 'p1', 'E0', 'E1', 'theta0', 'theta1', 'dM'],
        }
        
        # Add rank aliases
        for i in range(1, self.rank_use+ 1):
            pi0_veto_aliases[f'rank{i}'] = ['M', 'p0', 'p1', 'E0', 'E1', 'theta0', 'theta1', 'dM']
        
        # Also add the count variable
        var.addAlias(f'pi0veto_count_{self.pi0_veto_cut}', f'extraInfo(pi0veto_count_{self.pi0_veto_cut})')
        var.addAlias(f'chisq_alpha_{self.pi0_veto_cut}', f'extraInfo(chisq_alpha_{self.pi0_veto_cut})')
        #var.addAlias(f'chisq_beta_{self.pi0_veto_cut}', f'extraInfo(chisq_beta_{self.pi0_veto_cut})')
        
        # Create aliases in a loop
        for veto_type, variables in pi0_veto_aliases.items():
            for v in variables:
                alias_name = f'pi0veto_{veto_type}_{v}_{self.pi0_veto_cut}'
                var.addAlias(alias_name, f'extraInfo({alias_name})')
        
        # Define variables needed for ntuple
        pi0Veto_vars = [f'pi0veto_{veto_type}_{v}_{self.pi0_veto_cut}' 
                        for veto_type in pi0_veto_aliases 
                        for v in pi0_veto_aliases[veto_type]] + [f'pi0veto_count_{self.pi0_veto_cut}'] + ([f'chisq_alpha_{self.pi0_veto_cut}'] if self.kinfit else [])  # ,f'chisq_beta_{self.pi0_veto_cut}']
        
        return pi0Veto_vars
    
    def get_piVeto_cut(self,which_cut):
        """
        Get the pi0 veto cut string based on the cut efficiency
        
        Args:
            which_cut: Index of cut efficiency
        
        Returns:
            Pi0 veto cut string
        """
        self.pi0_veto_cut = which_cut
        photon_cuts = {
            'eff60' : '[[clusterNHits>1.5] and [0.2967< clusterTheta<2.6180]] and' + \
                   '[[clusterReg==1 and E>0.0225] or ' + \
                   '[clusterReg==2 and E>0.020] or ' + \
                   '[clusterReg==3 and E>0.020]]',
            'eff50' : '[[clusterNHits>1.5] and [0.2967< clusterTheta<2.6180]] and' + \
                   '[[clusterReg==1 and E>0.025] or ' + \
                   '[clusterReg==2 and E>0.025] or ' + \
                   '[clusterReg==3 and E>0.040]]',
            'eff40' : '[[clusterNHits>1.5] and [0.2967< clusterTheta<2.6180]] and' + \
                   '[[clusterReg==1 and E>0.080] or ' + \
                   '[clusterReg==2 and E>0.030] or ' + \
                   '[clusterReg==3 and E>0.060]]',
            'eff30' : '[[clusterNHits>1.5] and [0.2967< clusterTheta<2.6180]] and' + \
                   '[[clusterReg==1 and E>0.080] or ' + \
                   '[clusterReg==2 and E>0.030] or ' + \
                   '[clusterReg==3 and E>0.060]]',
            'eff20' : '[[clusterNHits>1.5] and [0.2967< clusterTheta<2.6180]] and' + \
                   '[[clusterReg==1 and E>0.120] or ' + \
                   '[clusterReg==2 and E>0.030] or ' + \
                   '[clusterReg==3 and E>0.080]] and clusterE1E9>0.4',
            'eff10' : '[[clusterNHits>1.5] and [0.2967< clusterTheta<2.6180]] and' + \
                   '[[clusterReg==1 and E>0.200] or ' + \
                   '[clusterReg==2 and E>0.100] or ' + \
                   '[clusterReg==3 and E>0.180]] and clusterE1E9>0.5'
        }
        return photon_cuts[which_cut]


    def setup_mc_truth(self, path, decay_chain=None , output_file=None):
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
        MCParticleInfo = BelleAnalysisBase.MCParticleInfo

        if output_file == None:
            output_file = self.output_file
        if decay_chain == None:
            decay_chain = '[vpho:mc_vpho -> [phi:mc_phi -> K+:mc_phikp K-:mc_phikm] K+:mc_vphokp K-:mc_vphokm]'
            
        
        # Create particle info objects with full list names
        particles = self.parse_decay_chain(decay_chain)
        vpho = particles[0]
        print(f"{vpho.list_name}")
        phi = next((p for p in vpho.daughters if p.name == 'phi'), None)

        ISR_photon = MCParticleInfo('gamma:mc_isr')
        ee = MCParticleInfo('vpho:mc_ee')
        
        ee.add_daughter(ISR_photon)
        ee.add_daughter(vpho)

        # Fill particle lists from MC
        ma.fillParticleListFromMC(ISR_photon.list_name, 'mcPrimary >0 ', path=path)
        ma.rankByHighest(ISR_photon.list_name, variable="useCMSFrame(E)", numBest=1, path=path)
        for p in particles:
            if len(p.daughters) ==0:
                ma.fillParticleListFromMC(p.list_name, f'genMotherPDG == {p.mother.pdg} and mcPrimary>0', path=path)

        particles.insert(0,ee)
        particles.append(ISR_photon)

        # Reconstruct decay chain (bottom-up)
        for p in reversed(particles):
            if p.get_decay_string():
                ma.reconstructDecay(p.get_decay_string(), ' ', path=path)

        # Define variables for different resonances
        vpho.variables = vu.create_aliases(vc.inv_mass + vc.kinematics, ee.get_daughter_access_string(vpho,"{variable}"),vpho.prefix)
        if phi:
            phi.variables = vu.create_aliases(['M'],ee.get_daughter_access_string(phi,"{variable}"),phi.prefix)
        ISR_photon.variables = vu.create_aliases(['E', 'theta', 'phi', 'E_CMS', 'theta_CMS', 'phi_CMS','M'], ee.get_daughter_access_string(ISR_photon,"{variable}"),ISR_photon.prefix)
        lists = ""
        for p in particles:
            if len(p.daughters) == 0 and p.mother.name not in ["pi0","eta"] and p.list_name != 'gamma:mc_isr':  # for phi K K analysis
                lists += f'{p.list_name},'
        lists = lists[:-1]
        print(f"lists: {lists}")
        var.addAlias('mc_sqrts',f'invMassInLists({lists})')

        # Add daughter variables
        for p in particles:
            if len(p.daughters) == 0 and p.name == 'gamma':
                p.variables = vu.create_aliases(['E', 'theta', 'phi', 'E_CMS', 'theta_CMS', 'phi_CMS'], ee.get_daughter_access_string(p,"{variable}"),p.prefix)
            elif len(p.daughters) == 0:
                p.variables = vu.create_aliases(['E', 'px', 'py', 'pz'], ee.get_daughter_access_string(p,"{variable}"),p.prefix)
 
        
        # Combine all truth variables
        truth_vars =  []
        for p in particles:
            truth_vars += p.variables
        truth_vars += ['mc_sqrts']
        
        # Save to event extra info
        for var_name in truth_vars:
            ma.variablesToEventExtraInfo(ee.list_name, variables={var_name: var_name}, path=path)
        
        # Create truth ntuple if output file is provided
        if output_file:
            ma.variablesToNtuple(ee.list_name, truth_vars, filename=output_file, treename='truth', path=path)
        
        return truth_vars


    def save_variables_before_fitting(self,decay_chain,path):
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
        vpho = next((p for p in firstP.daughters if p.name == 'vpho'), None)

        # Build event kinematics
        ma.buildEventKinematics(inputListNames=firstP.list_name , path=path)
        
        # Add visible energy alias
        if self.isr_mode == "tagged":
            var.addAlias('allPhotonE', f'totalEnergyOfParticlesInList(gamma:ISR)')
        
        # Create dictionary of variables to save
        var_dict = {}

        for p in particles:
            var_dict.update({
                firstP.get_daughter_access_string(p,f"{var}"):f"{p.prefix}_bf_{var}"
                for var in ["E","theta","px","py","pz"]
            })

        # Save system variables
        var_dict.update({firstP.get_daughter_access_string(vpho,f"{var}"): f"sys_{var}" for var in vc.recoil_kinematics + ["m2Recoil"]})
        var_dict.update({var: f"sys_{var}" for var in ["visibleEnergyOfEventCMS"] + (["allPhotonE"] if self.isr_mode == "tagged" else []) })

        # Save variables to extra info
        ma.variablesToExtraInfo(firstP.list_name, variables=var_dict, path=path)
        
        # Create aliases for before-fit variables
        bf_fit_vars = vu.create_aliases(list(var_dict.values()), 'extraInfo({variable})', '')
        
        return bf_fit_vars


