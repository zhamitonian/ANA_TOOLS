"""
parser Basf2 decay string
"""
from basf2 import B2INFO, B2ERROR, B2DEBUG, B2WARNING

"""
version 1.0.0
author: wangz
date: 2024-06-25
"""

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

        self.pdg = get_pdg_code(f"{self.name}")
    
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


def parse_decay_chain(decay_string):
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
                particles[parent_full_name] = MCParticleInfo(parent_full_name) 
            current_parent = particles[parent_full_name]
            
            # If this particle already has a different parent, warn about it
            if parent and current_parent.mother and current_parent.mother != parent:
                B2WARNING(f"{parent_full_name} already has parent {current_parent.mother.list_name}")
            
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
                        particles[child_full_name] = MCParticleInfo(child_full_name, mother=current_parent)
                    
                    current_parent.add_daughter(particles[child_full_name])
            
            return current_parent
        else:
            # Single particle without decay
            parent_full_name = decay_str.strip()
            
            if parent_full_name not in particles:
                particles[parent_full_name] = MCParticleInfo(parent_full_name, mother=parent)
            
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

"""
version history:
v1.0.0
- initial version
- seperated from BelleAnalysisBase.py
"""