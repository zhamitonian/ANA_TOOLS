"""
Usage examples for GenericFit with model parser.

The new approach allows expressing all operations directly in the model string:
- No need for separate 'operations' parameter
- Support PROD, FCONV, parentheses directly in model
- More intuitive and flexible
"""

import ROOT
from FIT.generic_fit import GenericFit


# Example 1: Simple 1D resonance fit
def example_1d_fit(tree):
    """Simple resonance fit with BW+Gauss signal and polynomial background."""
    fit = GenericFit(
        tree=tree,
        output_dir="output/phi_fit",
        variables=[("phi_M", 1.0, 1.06)],
        pdfs=[
            {
                "name": "sig",
                "var": "phi_M",
                "type": "bw_gauss",
                "config": {
                    "mass": 1.0195,
                    "width": 0.004249,
                    "sigma": 0.0015
                }
            },
            {
                "name": "bkg",
                "var": "phi_M",
                "type": "polynomial",
                "config": {"order": 1}
            }
        ],
        # Simple model: yields * PDFs
        model="nsig[100,0,10000]*sig + nbkg[50,0,10000]*bkg"
    )
    
    result, fit_results = fit.run()
    return result, fit_results


# Example 2: 2D fit with product PDFs (inline)
def example_2d_diphi_fit(tree):
    """2D di-phi fit with correlated and uncorrelated backgrounds."""
    fit = GenericFit(
        tree=tree,
        output_dir="output/diphi_fit",
        variables=[("phi1_M", 1.0, 1.06), ("phi2_M", 1.0, 1.06)],
        pdfs=[
            # Signal PDFs for each phi
            {"name": "sig1", "var": "phi1_M", "type": "bw_gauss",
             "config": {"mass": 1.0195, "width": 0.004249, "sigma": 0.0015}},
            {"name": "sig2", "var": "phi2_M", "type": "bw_gauss",
             "config": {"mass": 1.0195, "width": 0.004249, "sigma": 0.0015}},
            # Background PDFs
            {"name": "bkg1", "var": "phi1_M", "type": "polynomial",
             "config": {"order": 1}},
            {"name": "bkg2", "var": "phi2_M", "type": "polynomial",
             "config": {"order": 1}}
        ],
        # Model with PROD operations inline - no separate operations parameter!
        model=(
            "nsig[100,0,5000]*PROD(sig1, sig2) + "           # Both signal
            "nbkg_flat[50,0,2000]*PROD(bkg1, bkg2) + "       # Both background
            "nbkg_mix1[50,0,2000]*PROD(sig1, bkg2) + "       # Mixed
            "nbkg_mix2[50,0,2000]*PROD(bkg1, sig2)"          # Mixed
        ),
        plot_vars=["phi1_M", "phi2_M"]
    )
    
    result, fit_results = fit.run()
    return result, fit_results


# Example 3: Convolution fit (inline FCONV)
def example_convolution_fit(tree):
    """Fit with convolution of Breit-Wigner and Gaussian."""
    fit = GenericFit(
        tree=tree,
        output_dir="output/conv_fit",
        variables=[("mass", 1.0, 1.06)],
        pdfs=[
            # Separate BW and Gaussian for convolution
            {"name": "bw", "var": "mass", "type": "voigtian",
             "config": {"mass": 1.0195, "width": 0.004249, "sigma": 0.001}},
            {"name": "resolution", "var": "mass", "type": "gaussian",
             "config": {"mean": 0.0, "sigma": 0.001}},
            {"name": "bkg", "var": "mass", "type": "exponential",
             "config": {"tau": -1.0}}
        ],
        # FCONV directly in model string!
        model="nsig[100,0,10000]*FCONV(mass, bw, resolution) + nbkg[50,0,10000]*bkg"
    )
    
    result, fit_results = fit.run()
    return result, fit_results


# Example 4: Complex expression with parentheses
def example_complex_model(tree):
    """Demonstrate parentheses for grouping operations."""
    fit = GenericFit(
        tree=tree,
        output_dir="output/complex_fit",
        variables=[("mass", 1.0, 1.06)],
        pdfs=[
            {"name": "sig1", "var": "mass", "type": "gaussian",
             "config": {"mean": 1.0195, "sigma": 0.001}},
            {"name": "sig2", "var": "mass", "type": "gaussian",
             "config": {"mean": 1.0200, "sigma": 0.002}},
            {"name": "bkg", "var": "mass", "type": "polynomial",
             "config": {"order": 2}}
        ],
        # Use parentheses to create sum of signals, then multiply by yield
        # Parser will automatically create intermediate SUM PDF
        model="nsig[100,0,10000]*(sig1 + sig2) + nbkg[50,0,10000]*bkg"
    )
    
    result, fit_results = fit.run()
    return result, fit_results


# Example 5: 3D fit with multiple products
def example_3d_fit(tree):
    """3D fit demonstrating multiple PROD operations."""
    fit = GenericFit(
        tree=tree,
        output_dir="output/3d_fit",
        variables=[
            ("var1", 1.0, 1.06),
            ("var2", 1.0, 1.06),
            ("var3", 0, 50)
        ],
        pdfs=[
            # Signal PDFs
            {"name": "s1", "var": "var1", "type": "bw_gauss",
             "config": {"mass": 1.0195, "width": 0.004249, "sigma": 0.0015}},
            {"name": "s2", "var": "var2", "type": "bw_gauss",
             "config": {"mass": 1.0195, "width": 0.004249, "sigma": 0.0015}},
            {"name": "s3", "var": "var3", "type": "gaussian",
             "config": {"mean": 10, "sigma": 5}},
            # Background PDFs
            {"name": "b1", "var": "var1", "type": "polynomial", "config": {"order": 1}},
            {"name": "b2", "var": "var2", "type": "polynomial", "config": {"order": 1}},
            {"name": "b3", "var": "var3", "type": "exponential", "config": {"tau": -0.1}}
        ],
        # 3D product: PROD of 3 PDFs
        model=(
            "nsig[100,0,5000]*PROD(s1, s2, s3) + "
            "nbkg[50,0,2000]*PROD(b1, b2, b3)"
        ),
        plot_vars=["var1", "var2", "var3"]
    )
    
    result, fit_results = fit.run()
    return result, fit_results


# Example 6: Mixed operations
def example_mixed_operations(tree):
    """
    Demonstrate complex model with multiple operation types.
    Model: nsig * ((BW conv Gauss) + Crystal Ball) + nbkg * Polynomial
    """
    fit = GenericFit(
        tree=tree,
        output_dir="output/mixed_fit",
        variables=[("mass", 1.0, 1.06)],
        pdfs=[
            {"name": "bw", "var": "mass", "type": "voigtian",
             "config": {"mass": 1.0195, "width": 0.004249, "sigma": 0.0}},
            {"name": "gauss", "var": "mass", "type": "gaussian",
             "config": {"mean": 0.0, "sigma": 0.001}},
            {"name": "cb", "var": "mass", "type": "crystal_ball",
             "config": {"mean": 1.0195, "sigma": 0.002, "alpha": 1.5, "n": 2}},
            {"name": "bkg", "var": "mass", "type": "chebychev",
             "config": {"order": 2}}
        ],
        # Complex model: convolution + crystal ball (in parentheses), then background
        model="nsig[100,0,10000]*(FCONV(mass, bw, gauss) + cb) + nbkg[50,0,10000]*bkg"
    )
    
    result, fit_results = fit.run()
    return result, fit_results


if __name__ == "__main__":
    # Load example data
    f = ROOT.TFile("test_data.root")
    tree = f.Get("tree")
    
    # Run examples
    print("\\n=== Example 1: Simple 1D fit ===")
    example_1d_fit(tree)
    
    print("\\n=== Example 2: 2D di-phi fit ===")
    example_2d_diphi_fit(tree)
    
    print("\\n=== Example 3: Convolution fit ===")
    example_convolution_fit(tree)
    
    print("\\n=== Example 4: Complex model with parentheses ===")
    example_complex_model(tree)
    
    print("\\n=== Example 5: 3D fit ===")
    example_3d_fit(tree)
    
    print("\\n=== Example 6: Mixed operations ===")
    example_mixed_operations(tree)
