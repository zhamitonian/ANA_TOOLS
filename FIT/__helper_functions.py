def define_relativistic_breit_wigner(w, reso, mass, width, m1=0.49368, m2=0.49368, R=3.0, spin=1):
    """
    Define a relativistic Breit-Wigner PDF using RooGenericPdf in the given RooWorkspace `w`.
    Parameters:
    - w: RooWorkspace where the PDF will be defined
    - reso: Name of the resonance variable (e.g., "phi_M")
    - mass: Mass of the resonance (e.g., 1.019461 for phi
    - width: Width of the resonance (e.g., 0.004266 for phi)
    - m1: Mass of daughter particle 1 (default: 0.49368 GeV for K+)
    - m2: Mass of daughter particle 2 (default: 0.49368 GeV for K-)
    - R: Interaction radius (default: 3.0 GeV^-1)
    - spin: Spin of the resonance (default: 1 for phi meson)
    """
    
    # Define helper variables and formulas
    w.factory(f"bw_m0[{mass}]")      # Resonance mass
    w.factory(f"bw_width[{width}]")  # Resonance width
    w.factory(f"bw_m1[{m1}]")        # Daughter 1 mass
    w.factory(f"bw_m2[{m2}]")        # Daughter 2 mass
    w.factory(f"bw_R[{R}]")          # Barrier radius
    w.factory(f"bw_L[{spin}]")       # Spin

    # Helper function: momentum in rest frame
    # q(m) = sqrt((m^2 - (m1+m2)^2) * (m^2 - (m1-m2)^2)) / (2*m)
    q_formula = "sqrt((pow({m},2) - pow(bw_m1+bw_m2,2)) * (pow({m},2) - pow(bw_m1-bw_m2,2))) / (2*{m})"

    # Momentum at resonance mass
    q0_formula = q_formula.format(m="bw_m0")

    # Momentum at current mass
    q_formula_m = q_formula.format(m=f"{reso}_M")

    # Blatt-Weisskopf barrier factors for L=1
    # F_1(z) = sqrt(2*z^2 / (z^2 + 1))
    # where z = q * R

    # z at resonance mass
    z0_formula = f"({q0_formula}) * bw_R"
    F0_formula = f"sqrt(2*pow({z0_formula},2) / (pow({z0_formula},2) + 1))"

    # z at current mass
    z_formula = f"({q_formula_m}) * bw_R"
    F_formula = f"sqrt(2*pow({z_formula},2) / (pow({z_formula},2) + 1))"

    # Energy-dependent width
    # Gamma(m) = Gamma_0 * (q/q0)^(2L+1) * (m0/m) * (F(z)/F(z0))^2
    # For L=1: (2L+1) = 3
    width_ratio = f"pow(({q_formula_m})/({q0_formula}), 3)"
    mass_ratio = f"bw_m0 / {reso}_M"
    barrier_ratio = f"pow(({F_formula})/({F0_formula}), 2)"

    gamma_m = f"bw_width * {width_ratio} * {mass_ratio} * {barrier_ratio}"

    # Relativistic Breit-Wigner formula
    # BW(m) = 1 / ((m^2 - m0^2)^2 + m0^2 * Gamma(m)^2)
    # Add normalization factor for better numerical stability
    numerator = "1"
    denominator = f"(pow({reso}_M,2) - pow(bw_m0,2))*(pow({reso}_M,2) - pow(bw_m0,2)) + pow(bw_m0,2) * pow({gamma_m},2)"

    rel_bw_formula = f"{numerator} / ({denominator})"
    """
    print("=== Relativistic Breit-Wigner Formula ===")
    print(f"q(m) = {q_formula_m}")
    print(f"q(m0) = {q0_formula}")
    print(f"Gamma(m) = {gamma_m}")
    print(f"BW = {rel_bw_formula}")
    print("=" * 50)
    """
    # Create the PDF using RooGenericPdf
    w.factory(f"RooGenericPdf::reso_bw('{rel_bw_formula}', {{{reso}_M, bw_m0, bw_width, bw_m1, bw_m2, bw_R, bw_L}})")
