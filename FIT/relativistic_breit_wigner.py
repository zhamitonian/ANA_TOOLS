"""
Helper functions for defining Relativistic Breit-Wigner in RooFit

RooFit doesn't have a native relativistic Breit-Wigner PDF.
This module provides functions to create relativistic BW using RooGenericPdf.

Author: Generated for SpinAlignment Analysis
Date: 2025-11-06
"""

import ROOT
from math import sqrt


def define_relativistic_breit_wigner(w, reso, mass, width, daughter_mass=0.493677, L=1, radius=3.0):
    """
    Define relativistic Breit-Wigner with energy-dependent width in RooWorkspace
    
    This implements the Relativistic Breit-Wigner formula commonly used in 
    particle physics for resonances:
    
    BW(m) = N * (m0 * Gamma(m)) / ((m^2 - m0^2)^2 + (m0*Gamma(m))^2)
    
    where Gamma(m) is the energy-dependent width:
    Gamma(m) = Gamma0 * (m0/m) * (q(m)/q(m0))^(2L+1) * F_L^2(q(m),q(m0))
    
    For the Blatt-Weisskopf form factors:
    - L=0 (S-wave): F_L = 1
    - L=1 (P-wave): F_L = sqrt(1 + (q0*R)^2) / sqrt(1 + (q*R)^2)
    - L=2 (D-wave): F_L = sqrt(9 + 3*(q0*R)^2 + (q0*R)^4) / sqrt(9 + 3*(q*R)^2 + (q*R)^4)
    
    Args:
        w: RooWorkspace object
        reso: resonance variable name (e.g., "phi")
        mass: pole mass in GeV
        width: pole width in GeV
        daughter_mass: daughter particle mass in GeV (for phi->K+K-, use kaon mass = 0.493677 GeV)
        L: orbital angular momentum (0=S-wave, 1=P-wave, 2=D-wave)
        radius: interaction radius in GeV^-1 (typical value: 3-5 GeV^-1, corresponding to ~0.6-1.0 fm)
    
    Example:
        # For phi(1020) -> K+K- (P-wave)
        define_relativistic_breit_wigner(w, "phi", 1.019461, 0.004249, 0.493677, L=1)
        
        # For f0(980) -> pi+pi- (S-wave, but better use Flatté)
        define_relativistic_breit_wigner(w, "f0", 0.990, 0.070, 0.13957, L=0)
        
        # For rho(770) -> pi+pi- (P-wave)
        define_relativistic_breit_wigner(w, "rho", 0.7755, 0.1494, 0.13957, L=1)
    """
    
    var_name = f"{reso}_M"
    
    # Calculate breakup momentum q(m) for two-body decay
    # For identical particles: q(m) = sqrt(m^2/4 - m_daughter^2)
    # General case: q(m) = sqrt((m^2 - (m1+m2)^2) * (m^2 - (m1-m2)^2)) / (2*m)
    # Here we use identical particles formula
    q_formula = f"sqrt(max(0.0, {var_name}*{var_name}/4.0 - {daughter_mass}*{daughter_mass}))"
    q0_value = sqrt(max(0.0, mass*mass/4.0 - daughter_mass*daughter_mass))
    
    # Blatt-Weisskopf form factor
    if L == 0:  # S-wave
        form_factor_sq = "1.0"
    elif L == 1:  # P-wave
        z_formula = f"pow(({q_formula})*{radius}, 2)"
        z0_value = (q0_value * radius)**2
        form_factor_sq = f"(1.0 + {z0_value}) / (1.0 + {z_formula})"
    elif L == 2:  # D-wave
        z_formula = f"pow(({q_formula})*{radius}, 2)"
        z0_value = (q0_value * radius)**2
        numerator = 9.0 + 3.0*z0_value + z0_value*z0_value
        form_factor_sq = f"({numerator}) / (9.0 + 3.0*{z_formula} + {z_formula}*{z_formula})"
    else:
        raise ValueError(f"Unsupported orbital angular momentum L={L}. Use L=0,1,2")
    
    # Running width: Gamma(m) = Gamma0 * (m0/m) * (q(m)/q(m0))^(2L+1) * F_L^2
    if q0_value > 0:
        running_width = f"{width} * ({mass}/{var_name}) * pow(({q_formula})/{q0_value}, {2*L+1}) * ({form_factor_sq})"
    else:
        # Below threshold, set width to zero
        running_width = "0.0"
        print(f"Warning: q0 = 0, resonance mass below threshold!")
    
    # Relativistic Breit-Wigner formula
    # BW(m) = (m0 * Gamma(m)) / ((m^2 - m0^2)^2 + (m0*Gamma(m))^2)
    bw_numerator = f"{mass} * ({running_width})"
    bw_denominator = f"pow({var_name}*{var_name} - {mass}*{mass}, 2) + pow({bw_numerator}, 2)"
    bw_formula = f"({bw_numerator}) / ({bw_denominator})"
    
    # Create the PDF
    w.factory(f"RooGenericPdf::reso_bw('{bw_formula}', {{{var_name}}})") 
    
    print(f"  Defined relativistic Breit-Wigner for {reso}:")
    print(f"  Mass: {mass} GeV, Width: {width} GeV")
    print(f"  Daughter mass: {daughter_mass} GeV")
    print(f"  Angular momentum L={L}, Radius={radius} GeV^-1")
    print(f"  Breakup momentum q0: {q0_value:.6f} GeV")



def define_flatte_formula(w, reso, mass, g1, g2, m1a, m1b, m2a, m2b):
    """
    Define Flatté formula for coupled-channel resonances
    
    Commonly used for scalar mesons like f0(980) and a0(980) which couple 
    to both pipi and KK channels.
    
    Formula: 
    BW(m) = 1 / ((m0^2 - m^2)^2 + (Gamma_tot(m))^2)
    
    where Gamma_tot(m) = rho1(m)*g1^2 + rho2(m)*g2^2
    and rho_i(m) = 2*q_i/m is the phase space factor for channel i
    
    Args:
        w: RooWorkspace object
        reso: resonance variable name
        mass: pole mass in GeV
        g1, g2: coupling constants for channel 1 and 2 in GeV
        m1a, m1b: masses of particles in channel 1 in GeV
        m2a, m2b: masses of particles in channel 2 in GeV
        
    Example:
        # For f0(980) -> pipi/KK
        define_flatte_formula(w, "f0", 0.965, 0.165, 0.695, 
                             0.13957, 0.13957,  # pi+ pi-
                             0.493677, 0.493677) # K+ K-
    """
    
    var_name = f"{reso}_M"
    
    # Phase space factors
    # q_i = sqrt((m^2 - (m_ia+m_ib)^2)*(m^2 - (m_ia-m_ib)^2))/(2m)
    # rho_i = 2*q_i/m
    q1_sq = f"max(0.0, ({var_name}*{var_name} - {(m1a+m1b)**2}) * ({var_name}*{var_name} - {(m1a-m1b)**2}))"
    q2_sq = f"max(0.0, ({var_name}*{var_name} - {(m2a+m2b)**2}) * ({var_name}*{var_name} - {(m2a-m2b)**2}))"
    
    rho1 = f"sqrt({q1_sq}) / ({var_name}*{var_name})"
    rho2 = f"sqrt({q2_sq}) / ({var_name}*{var_name})"
    
    # Total width
    total_width = f"({rho1})*{g1}*{g1} + ({rho2})*{g2}*{g2}"
    
    # Flatté amplitude squared
    # Using m^2 - m0^2 instead of m - m0 for relativistic case
    flatte_formula = f"1.0 / (pow({mass}*{mass} - {var_name}*{var_name}, 2) + pow({total_width}, 2))"
    
    w.factory(f"RooGenericPdf::reso_bw('{flatte_formula}', {{{var_name}}})") 
    
    print(f"✓ Defined Flatté formula for {reso}:")
    print(f"  Mass: {mass} GeV")
    print(f"  g1={g1} GeV (channel: {m1a},{m1b} GeV)")
    print(f"  g2={g2} GeV (channel: {m2a},{m2b} GeV)")

