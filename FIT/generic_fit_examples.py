"""
Usage examples showing the power of GenericFit.

This demonstrates how a single GenericFit class can handle all fitting scenarios
by focusing on PDF configuration rather than specialized fit classes.
"""

import ROOT
from FIT.generic_fit import GenericFit
from FIT.pdf_builders import PDF_REGISTRY


# ============================================================================
# Example 1: Simple 1D resonance fit
# ============================================================================
def example_1d_resonance():
    """
    Simplest case: 1D resonance fit (phi meson).
    """
    f = ROOT.TFile("data.root")
    tree = f.Get("tree")
    
    fit = GenericFit(
        tree=tree,
        output_dir="output/phi_1d",
        
        # Define variable
        variables=[("phi_M", 1.0, 1.06)],
        
        # Signal: Breit-Wigner ⊗ Gaussian
        signal_pdfs=[{
            "var": "phi_M",
            "type": "bw_gauss",
            "config": {"mass": 1.0195, "width": 0.004249}
        }],
        
        # Background: 1st order polynomial
        background_pdfs=[{
            "var": "phi_M",
            "type": "polynomial",
            "config": {"order": 1}
        }],
        
        # Model: sig + bkg
        model_structure="nsig[100,0,10000]*sig_phi_M + nbkg[50,0,10000]*bkg_phi_M",
        
        # Plot configuration
        plot_config={
            "nbin": 60,
            "components": {
                "sig_phi_M": {"color": ROOT.kRed, "style": 4, "width": 3},
                "bkg_phi_M": {"color": ROOT.kGreen+2, "style": 7, "width": 3}
            }
        }
    )
    
    result, fit_results = fit.run()
    return fit_results


# ============================================================================
# Example 2: 2D fit (di-phi)
# ============================================================================
def example_2d_diphi():
    """
    2D fit: ISR gamma -> phi phi.
    
    This is more complex but still uses the same GenericFit!
    """
    f = ROOT.TFile("data.root")
    tree = f.Get("tree")
    
    fit = GenericFit(
        tree=tree,
        output_dir="output/diphi_2d",
        
        # Two variables
        variables=[
            ("phi1_M", 1.0, 1.06),
            ("phi2_M", 1.0, 1.06)
        ],
        
        # Signal PDFs for each phi
        signal_pdfs=[
            {
                "var": "phi1_M",
                "type": "bw_gauss",
                "config": {"mass": 1.0195, "width": 0.004249}
            },
            {
                "var": "phi2_M",
                "type": "bw_gauss",
                "config": {"mass": 1.0195, "width": 0.004249}
            }
        ],
        
        # Background PDFs for each dimension
        background_pdfs=[
            {
                "var": "phi1_M",
                "type": "polynomial",
                "config": {"order": 1}
            },
            {
                "var": "phi2_M",
                "type": "polynomial",
                "config": {"order": 1}
            }
        ],
        
        # Combine PDFs for 2D
        combined_pdfs={
            "sig_2d": "PROD::sig_2d(sig_phi1_M, sig_phi2_M)",           # Both signal
            "bkg_flat": "PROD::bkg_flat(bkg_phi1_M, bkg_phi2_M)",       # Both background
            "bkg_mixed1": "PROD::bkg_mixed1(sig_phi1_M, bkg_phi2_M)",   # phi1 sig, phi2 bkg
            "bkg_mixed2": "PROD::bkg_mixed2(bkg_phi1_M, sig_phi2_M)"    # phi1 bkg, phi2 sig
        },
        
        # Model with 4 components
        model_structure=(
            "nsig[100,0,5000]*sig_2d + "
            "nbkg1[50,0,2000]*bkg_flat + "
            "nbkg2[50,0,2000]*bkg_mixed1 + "
            "nbkg3[50,0,2000]*bkg_mixed2"
        ),
        
        # Plot both projections
        plot_vars=["phi1_M", "phi2_M"],
        plot_config={
            "nbin": 60,
            "components": {
                "sig_2d": {"color": ROOT.kRed, "style": 4, "width": 3},
                "bkg_flat": {"color": ROOT.kGreen+2, "style": 7, "width": 3},
                "bkg_mixed1": {"color": ROOT.kMagenta, "style": 5, "width": 3},
                "bkg_mixed2": {"color": ROOT.kCyan, "style": 2, "width": 3}
            }
        }
    )
    
    result, fit_results = fit.run()
    return fit_results


# ============================================================================
# Example 3: Chi-square template fit
# ============================================================================
def example_chisq_template():
    """
    Template fit using MC for signal shape.
    """
    data_file = ROOT.TFile("data.root")
    data_tree = data_file.Get("tree")
    
    mc_file = ROOT.TFile("signal_mc.root")
    mc_tree = mc_file.Get("tree")
    
    fit = GenericFit(
        tree=data_tree,
        output_dir="output/chisq_template",
        
        # Chi-square variable
        variables=[("chisq", 0, 50)],
        
        # Signal: Template from MC (needs custom builder - see below)
        signal_pdfs=[{
            "var": "chisq",
            "type": "keys_pdf",  # Kernel density estimation
            "config": {"template_tree": mc_tree, "mirror": "Both", "rho": 2.0}
        }],
        
        # Background: Exponential
        background_pdfs=[{
            "var": "chisq",
            "type": "exponential",
            "config": {}
        }],
        
        # Model
        model_structure="nsig[1000,0,50000]*sig_chisq + nbkg[1000,0,50000]*bkg_chisq",
        
        plot_config={"nbin": 80}
    )
    
    result, fit_results = fit.run()
    return fit_results


# ============================================================================
# Example 4: Trying different signal PDFs (systematic study)
# ============================================================================
def example_systematic_comparison():
    """
    Compare different PDF choices for the same fit.
    
    This shows the real power: change PDF type without changing the fit logic!
    """
    f = ROOT.TFile("data.root")
    tree = f.Get("tree")
    
    # Test different signal PDFs
    signal_types = ["bw_gauss", "voigtian", "crystal_ball", "double_gauss_bw"]
    
    results = {}
    
    for sig_type in signal_types:
        print(f"\n{'='*60}")
        print(f"Testing signal PDF: {sig_type}")
        print(f"{'='*60}")
        
        fit = GenericFit(
            tree=tree,
            output_dir=f"output/systematic_{sig_type}",
            variables=[("phi_M", 1.0, 1.06)],
            
            # Just change the type!
            signal_pdfs=[{
                "var": "phi_M",
                "type": sig_type,
                "config": {"mass": 1.0195, "width": 0.004249, "mean": 1.0195}
            }],
            
            background_pdfs=[{
                "var": "phi_M",
                "type": "polynomial",
                "config": {"order": 1}
            }],
            
            model_structure="nsig[100,0,10000]*sig_phi_M + nbkg[50,0,10000]*bkg_phi_M"
        )
        
        result, fit_results = fit.run()
        results[sig_type] = fit_results
    
    # Compare
    print("\n" + "="*60)
    print("SYSTEMATIC COMPARISON")
    print("="*60)
    for sig_type, res in results.items():
        print(f"{sig_type:20s}: Nsig = {res['nsig']:.1f} ± {res['nsig_err']:.1f}")


# ============================================================================
# Example 5: Add custom PDF builder for special case
# ============================================================================
def example_custom_template_pdf():
    """
    Add a custom PDF builder for RooKeysPdf (template from MC).
    """
    from FIT.pdf_builders import PDFBuilder
    
    class KeysPdfBuilder(PDFBuilder):
        """Build RooKeysPdf from MC template."""
        
        def build(self, workspace, var_name, config):
            template_tree = config["template_tree"]
            mirror = config.get("mirror", "Both")
            rho = config.get("rho", 2.0)
            
            # Create temporary tools to make dataset from MC tree
            from FIT.fit_tools import FIT_UTILS
            var_range = (workspace.var(var_name).getMin(), workspace.var(var_name).getMax())
            temp_tools = FIT_UTILS(var_config=[(var_name, var_range[0], var_range[1])])
            
            # Create dataset from MC tree
            mc_dataset = temp_tools.handle_dataset(template_tree, workspace, None, False)
            workspace.Import(mc_dataset, ROOT.RooFit.Rename(f"mc_dataset_{var_name}"))
            
            # Create RooKeysPdf
            pdf_name = f"sig_{var_name}"
            mirror_opt = getattr(ROOT.RooKeysPdf, f"Mirror{mirror}")
            
            keys_pdf = ROOT.RooKeysPdf(
                pdf_name,
                pdf_name,
                workspace.var(var_name),
                mc_dataset,
                mirror_opt,
                rho
            )
            
            getattr(workspace, "import")(keys_pdf)
            
            return pdf_name
    
    # Register the custom builder
    PDF_REGISTRY.register_signal("keys_pdf", KeysPdfBuilder())
    
    print("Custom KeysPdf builder registered!")
    print(f"Available signal PDFs: {PDF_REGISTRY.list_signal_types()}")


# ============================================================================
# Example 6: Multi-component fit
# ============================================================================
def example_multi_component():
    """
    Fit with multiple signal components.
    
    For example: J/psi + psi(2S) in the same mass range.
    """
    f = ROOT.TFile("data.root")
    tree = f.Get("tree")
    
    fit = GenericFit(
        tree=tree,
        output_dir="output/jpsi_psi2s",
        
        variables=[("dimuon_M", 2.8, 4.0)],
        
        # Two signal PDFs (J/psi and psi(2S))
        signal_pdfs=[
            {
                "var": "dimuon_M",
                "type": "crystal_ball",
                "config": {"mean": 3.0969, "sigma": 0.04, "alpha": 1.5, "n": 2.0}
            },
            {
                "var": "dimuon_M", 
                "type": "crystal_ball",
                "config": {"mean": 3.686, "sigma": 0.04, "alpha": 1.5, "n": 2.0}
            }
        ],
        
        background_pdfs=[{
            "var": "dimuon_M",
            "type": "exponential",
            "config": {}
        }],
        
        # Note: We need to rename the second signal PDF to avoid collision
        # This could be done by extending the PDF builder to accept a custom name
        model_structure=(
            "n_jpsi[1000,0,100000]*sig_dimuon_M + "  # First signal is J/psi
            "n_psi2s[100,0,10000]*sig2_dimuon_M + "  # Second signal is psi(2S)
            "nbkg[500,0,100000]*bkg_dimuon_M"
        )
    )
    
    # Note: This example would need enhancement to PDF builders to support
    # multiple PDFs of the same type for the same variable with different names


# ============================================================================
# Summary
# ============================================================================
def print_summary():
    """
    Print summary of the generic fit approach.
    """
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║          GenericFit: Universal Fitting Framework             ║
    ╚══════════════════════════════════════════════════════════════╝
    
    KEY INSIGHT:
    All fits follow the same workflow, only the PDF configurations differ!
    
    ADVANTAGES:
    ✓ Single class handles 1D, 2D, multi-D fits
    ✓ Focus on PDF builders, not fit classes
    ✓ Easy to compare different PDF choices
    ✓ Simple to add new PDF types
    ✓ Clear separation of concerns
    
    WORKFLOW:
    1. Define variables: [("var", min, max), ...]
    2. Specify signal PDFs: [{"var": "x", "type": "bw_gauss", "config": {...}}, ...]
    3. Specify background PDFs: [{"var": "x", "type": "polynomial", "config": {...}}, ...]
    4. Define model structure: "nsig[...]*sig + nbkg[...]*bkg"
    5. Run!
    
    TO ADD NEW FIT SCENARIO:
    - Create new PDF builders in pdf_builders.py
    - Use GenericFit with appropriate configuration
    - No need for new fit classes!
    """)


if __name__ == "__main__":
    print_summary()
    
    print("\nAvailable examples:")
    print("1. example_1d_resonance() - Simple 1D phi fit")
    print("2. example_2d_diphi() - 2D di-phi fit")
    print("3. example_chisq_template() - Template fit")
    print("4. example_systematic_comparison() - Compare PDFs")
    print("5. example_custom_template_pdf() - Add custom PDF builder")
    print("6. example_multi_component() - Multi-signal fit")
