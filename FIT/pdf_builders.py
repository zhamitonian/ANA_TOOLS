"""
PDF builder classes for constructing different types of signal and background PDFs.

This module provides a flexible way to construct various PDFs for fitting.
Each builder encapsulates the logic for creating a specific type of PDF.

version: 2.1
date   : 2026-02-04
author : wang zheng
"""

import ROOT
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional


class PDFBuilder(ABC):
    """Abstract base class for PDF builders."""
    
    # Subclasses should define available parameters with default values
    # Format: {"param_name": default_value, ...}
    # where default_value can be: number (fixed), (min,max), or (init,min,max)
    PARAMETERS: Dict[str, Any] = {}
    
    def get_available_params(self) -> Dict[str, Any]:
        """
        Get available parameters and their default values for this PDF type.
        
        Returns:
            Dict mapping parameter names to default values
        """
        return self.PARAMETERS.copy()
    
    def get_params(self, config: Dict[str, Any], pdf_name: str) -> Dict[str, str]:
        """
        Get all parameters from PARAMETERS, overridden by config values.
        Process all at once and return formatted strings ready for RooFit factory.
        
        Args:
            config: User configuration dictionary
            pdf_name: PDF name for unique variable naming
            
        Returns:
            Dict mapping parameter names to RooFit factory strings
            
        Raises:
            ValueError: If config contains invalid parameter names
            
        Example:
            params = self.get_params(config, pdf_name)
            mass_str = params["mass"]
            sigma_str = params["sigma"]
        """
        # Validate parameter names
        invalid_keys = [key for key in config.keys() if key not in self.PARAMETERS]
        if invalid_keys:
            available = list(self.PARAMETERS.keys())
            raise ValueError(
                f"Invalid parameter(s): {invalid_keys}. "
                f"Available parameters for {self.__class__.__name__}: {available}"
            )
        
        result = {}
        for key in self.PARAMETERS.keys():
            # Get value from config, or fall back to default
            if key in config:
                val = config[key]
            else:
                val = self.PARAMETERS[key]
            
            # Format and store
            formatted = self._format_param(key, val, pdf_name)
            result[key] = formatted
        
        return result
    
    def parse_param(self, key: str, config: Dict[str, Any], pdf_name: str, default_val=None) -> str:
        """
        Parse parameter following RooFit factory syntax (backward compatibility).
        
        Args:
            key: Parameter name
            config: Configuration dictionary
            pdf_name: PDF name for unique variable naming
            default_val: Default value if key not in config
            
        Returns:
            String for RooFit factory: "value" or "name[min,max]" or "name[init,min,max]"
            
        Examples:
            value: 0.497 -> "0.497" (fixed)
            value: (0.49, 0.50) -> "key_pdfname[0.49, 0.50]" (float, no init)
            value: (0.497, 0.49, 0.50) -> "key_pdfname[0.497, 0.49, 0.50]" (float with init)
        """
        val = config.get(key, default_val)
        if val is None:
            return None
        return self._format_param(key, val, pdf_name)
    
    def _format_param(self, key: str, val: Any, pdf_name: str) -> str:
        """Format parameter value to RooFit factory syntax string."""
        if val is None:
            return None
        if isinstance(val, (int, float)):
            # Fixed value
            return str(val)
        elif isinstance(val, (list, tuple)):
            if len(val) == 2:
                # [min, max] - no initial value
                return f"{key}_{pdf_name}[{val[0]}, {val[1]}]"
            elif len(val) == 3:
                # [init, min, max]
                return f"{key}_{pdf_name}[{val[0]}, {val[1]}, {val[2]}]"
            else:
                raise ValueError(f"Invalid range for {key}: {val}. Use value, (min,max), or (init,min,max)")
        else:
            return str(val)
    
    @abstractmethod
    def build(self, workspace: ROOT.RooWorkspace, var_name: str, 
              config: Dict[str, Any], pdf_name: str) -> str:
        """
        Build a PDF in the workspace.
        
        Args:
            workspace: RooFit workspace
            var_name: Name of the observable variable
            config: Configuration dictionary with parameters
            pdf_name: Name for the PDF (required, no default)
            
        Returns:
            str: Name of the created PDF in the workspace
        """
        pass


# ============================================================================
# PDF Builders (no signal/background distinction)
# ============================================================================

class BreitWignerGaussBuilder(PDFBuilder):
    """
    Build Breit-Wigner convoluted with Gaussian (BW ⊗ Gauss).
    
    Typical use: resonance signal PDF with detector resolution.
    
    Config parameters (RooFit factory style):
        - mass: resonance mass (fixed value)
        - width: resonance width (fixed value)
        - resolution: value or (min, max) or (init, min, max) - default: (0.00083, 0.0002, 0.0014)
    """
    
    PARAMETERS = {
        "mass": None,  # Required
        "width": None,  # Required
        "resolution": (0.00083, 0.0002, 0.0014),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        # Breit-Wigner
        workspace.factory(f"BreitWigner::bw_{var_name}({var_name}, {params['mass']}, {params['width']})")
        
        # Gaussian smearing
        workspace.factory(f"Gaussian::gauss_{var_name}({var_name}, 0, {params['resolution']})")
        
        # Convolution
        workspace.factory(f"FCONV::{pdf_name}({var_name}, bw_{var_name}, gauss_{var_name})")
        
        return pdf_name

class DoubleGaussianBuilder(PDFBuilder):
    """
    Build double Gaussian PDF (G1 + G2).
    
    Sum of two Gaussians with different means and widths.
    Useful for modeling backgrounds or signals with asymmetric resolution.
    
    Config parameters (RooFit factory style):
        - mean1: value (fixed) or (min, max) or (init, min, max)
        - mean2: value (fixed) or (min, max) or (init, min, max)  
        - sigma1: value (fixed) or (min, max) or (init, min, max)
        - sigma2: value (fixed) or (min, max) or (init, min, max)
        - frac: value (fixed) or (min, max) or (init, min, max) - fraction of first Gaussian
        - same_mean: if True, both Gaussians share the same mean (default: False)
    
    Examples:
        mean1: 0.497  # Fixed at 0.497
        mean1: (0.490, 0.505)  # Float in range, no initial value
        mean1: (0.497, 0.490, 0.505)  # Float with initial value 0.497
    """
    
    PARAMETERS = {
        "mean1": 0,
        "mean2": 0,
        "sigma1": (0.0005, 0.0001, 0.005),
        "sigma2": (0.002, 0.001, 0.01),
        "frac": (0.5, 0, 1),
        "same_mean": False,
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        same_mean = config.get("same_mean", self.PARAMETERS["same_mean"])
        
        if same_mean:
            # Both Gaussians share the same mean
            workspace.factory(
                f"Gaussian::gauss1_{pdf_name}({var_name}, {params['mean1']}, {params['sigma1']})"
            )
            workspace.factory(
                f"Gaussian::gauss2_{pdf_name}({var_name}, {params['mean1']}, {params['sigma2']})"
            )
        else:
            # Different means for each Gaussian
            workspace.factory(
                f"Gaussian::gauss1_{pdf_name}({var_name}, {params['mean1']}, {params['sigma1']})"
            )
            workspace.factory(
                f"Gaussian::gauss2_{pdf_name}({var_name}, {params['mean2']}, {params['sigma2']})"
            )
        
        # Sum of Gaussians
        workspace.factory(
            f"SUM::{pdf_name}({params['frac']} * gauss1_{pdf_name}, gauss2_{pdf_name})"
        )
        
        return pdf_name

class DoubleGaussianBreitWignerBuilder(PDFBuilder):
    """
    Build Breit-Wigner convoluted with double Gaussian (BW ⊗ (G1 + G2)).
    
    For better resolution modeling with core + tail components.
    
    Config parameters (RooFit factory style):
        - mass: resonance mass (fixed)
        - width: resonance width (fixed)
        - sigma1: value or (min, max) or (init, min, max) - core Gaussian
        - sigma2: value or (min, max) or (init, min, max) - tail Gaussian
        - frac: value or (min, max) or (init, min, max) - fraction of core
    """
    
    PARAMETERS = {
        "mass": None,  # Required
        "width": None,  # Required
        "sigma1": (0.0005, 0.0001, 0.005),
        "sigma2": (0.002, 0.001, 0.01),
        "frac": (0.7, 0, 1),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        # Breit-Wigner
        workspace.factory(f"BreitWigner::bw_{pdf_name}({var_name}, {params['mass']}, {params['width']})")
        
        # Core Gaussian
        workspace.factory(f"Gaussian::gauss1_{pdf_name}({var_name}, 0, {params['sigma1']})")
        
        # Tail Gaussian
        workspace.factory(f"Gaussian::gauss2_{pdf_name}({var_name}, 0, {params['sigma2']})")
        
        # Sum of Gaussians
        workspace.factory(
            f"SUM::double_gauss_{pdf_name}({params['frac']} * gauss1_{pdf_name}, gauss2_{pdf_name})"
        )
        
        # Convolution
        workspace.factory(
            f"FCONV::{pdf_name}({var_name}, bw_{pdf_name}, double_gauss_{pdf_name})"
        )
        
        return pdf_name


class CrystalBallBuilder(PDFBuilder):
    """
    Build Crystal Ball PDF.
    
    Gaussian core with power-law tail, useful for asymmetric line shapes.
    
    Config parameters (RooFit factory style):
        - mean: value or (min, max) or (init, min, max)
        - sigma: value or (min, max) or (init, min, max)
        - alpha: value or (min, max) or (init, min, max) - tail parameter
        - n: value or (min, max) or (init, min, max) - tail power
    """
    
    PARAMETERS = {
        "mean": None,  # Required
        "sigma": (0.001, 0.0001, 0.01),
        "alpha": (1.5, 0, 5),
        "n": (2.0, 0, 10),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        workspace.factory(
            f"CBShape::{pdf_name}({var_name}, {params['mean']}, {params['sigma']}, {params['alpha']}, {params['n']})"
        )
        
        return pdf_name


class VoigtianBuilder(PDFBuilder):
    """
    Build Voigtian PDF (convolution of Breit-Wigner and Gaussian analytically).
    
    Faster alternative to numerical FFT convolution.
    
    Config parameters (RooFit factory style):
        - mean: value or (min, max) or (init, min, max)
        - width: value or (min, max) or (init, min, max) - BW width
        - sigma: value or (min, max) or (init, min, max) - Gaussian width
    """
    
    PARAMETERS = {
        "mean": None,  # Required
        "width": None,  # Required
        "sigma": (0.001, 0.0001, 0.01),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        workspace.factory(
            f"Voigtian::{pdf_name}({var_name}, {params['mean']}, {params['width']}, {params['sigma']})"
        )
        
        return pdf_name


class GaussianBuilder(PDFBuilder):
    """
    Build simple Gaussian PDF.
    
    Config parameters (RooFit factory style):
        - mean: value or (min, max) or (init, min, max)
        - sigma: value or (min, max) or (init, min, max)
    """
    
    PARAMETERS = {
        "mean": 0,
        "sigma": (0.01, 0.001, 0.1),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        workspace.factory(f"Gaussian::{pdf_name}({var_name}, {params['mean']}, {params['sigma']})")
        
        return pdf_name


class PolynomialBuilder(PDFBuilder):
    """
    Build polynomial background PDF.
    
    Config parameters:
        - order: polynomial order (default: 1)
        - coef: value or (min, max) or (init, min, max) - coefficient range (default: (-10, 10))
        - coef0, coef1, ... : individual coefficient settings (optional, overrides coef)
    """
    
    PARAMETERS = {
        "order": 1,
        "coef": (-10, 10),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        order = config.get("order", self.PARAMETERS["order"])
        
        coef_list = []
        for i in range(order + 1):
            # Check for individual coef setting
            if f"coef{i}" in config:
                coef_str = self._format_param(f"coef{i}", config[f"coef{i}"], pdf_name)
            else:
                # Use generic coef setting but with unique names
                coef_default = config.get("coef", self.PARAMETERS["coef"])
                coef_str = self._format_param(f"coef{i}", coef_default, pdf_name)
            coef_list.append(coef_str)
        
        workspace.factory(f"Polynomial::{pdf_name}({var_name}, {{{', '.join(coef_list)}}})")
        
        return pdf_name


class ChebychevBuilder(PDFBuilder):
    """
    Build Chebychev polynomial background PDF.
    
    Config parameters:
        - order: polynomial order (default: 1)
        - coef: value or (min, max) or (init, min, max) - coefficient range (default: (-10, 10))
        - coef0, coef1, ... : individual coefficient settings (optional, overrides coef)
    """
    
    PARAMETERS = {
        "order": 1,
        "coef": (-10, 10),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        order = config.get("order", self.PARAMETERS["order"])
        
        coef_list = []
        for i in range(order + 1):
            # Check for individual coef setting
            if f"coef{i}" in config:
                coef_str = self._format_param(f"coef{i}", config[f"coef{i}"], pdf_name)
            else:
                # Use generic coef setting but with unique names
                coef_default = config.get("coef", self.PARAMETERS["coef"])
                coef_str = self._format_param(f"coef{i}", coef_default, pdf_name)
            coef_list.append(coef_str)
        test = ", ".join(coef_list)
        print("Chebychev coefficients:", test)
        
        workspace.factory(f"Chebychev::{pdf_name}({var_name}, {{{', '.join(coef_list)}}})")
        
        return pdf_name


class ArgusBGBuilder(PDFBuilder):
    """
    Build ARGUS background PDF.
    
    Commonly used for kinematic endpoints.
    
    Config parameters (RooFit factory style):
        - m0: endpoint mass (usually fixed)
        - c: value or (min, max) or (init, min, max) - shape parameter
        - p: value or (min, max) or (init, min, max) - power parameter
    """
    
    PARAMETERS = {
        "m0": None,  # Required
        "c": (-20, -100, -0.01),
        "p": (0.5, 0, 1),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        workspace.factory(
            f"ArgusBG::{pdf_name}({var_name}, {params['m0']}, {params['c']}, {params['p']})"
        )
        
        return pdf_name


class ExponentialBuilder(PDFBuilder):
    """
    Build exponential background PDF.
    
    Config parameters (RooFit factory style):
        - tau: value or (min, max) or (init, min, max) - decay constant
    """
    
    PARAMETERS = {
        "tau": (-1.0, -10, 10),
    }
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        workspace.factory(f"Exponential::{pdf_name}({var_name}, {params['tau']})")
        
        return pdf_name


class FlatBuilder(PDFBuilder):
    """
    Build flat (uniform) background PDF.
    
    No configuration needed.
    """
    
    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        workspace.factory(f"Uniform::{pdf_name}({var_name})")
        
        return pdf_name


# template fit 
class TemplateFitBuilder(PDFBuilder):
    """
    Build template PDF from MC data using either RooKeysPdf (unbinned) or RooHistPdf (binned).
    
    For unbinned fits: Uses RooKeysPdf to create a smooth kernel density estimation from MC data.
    For binned fits: Creates a histogram template from MC data and builds RooHistPdf.
    
    Config parameters:
        - template_file: Path to ROOT file containing MC template data (Required)
        - tree_name: Name of TTree in template file (default: "event")
        - var_name: Name of the variable in the tree (default: from var_name parameter)
        - weight_branch: Branch name for event weights (optional)
        - binned: Use binned template (RooHistPdf) instead of unbinned (RooKeysPdf) (default: False)
        - nbins: Number of bins for histogram template (default: 100)
        - range: Tuple (min, max) for variable range (default: from workspace variable)
    """
    # suppose the var name in mc is same as in workspace(real data) 
    PARAMETERS = {
        "template_file": None,  # Required
        "tree_name": "event",
        "weight_branch": None,
        "binned": False,
        "nbins": 100,
        "range" : None,
    }

    def build(self, workspace: ROOT.RooWorkspace, var_name: str,
              config: Dict[str, Any], pdf_name: str) -> str:
        params = self.get_params(config, pdf_name)
        
        # Open template file and get MC tree
        template_file = ROOT.TFile.Open(params["template_file"])
        if not template_file or template_file.IsZombie():
            raise RuntimeError(f"Cannot open template file: {params['template_file']}")
        
        tree = template_file.Get(params["tree_name"])
        if not tree:
            raise RuntimeError(f"Cannot find tree '{params['tree_name']}' in {params['template_file']}")
        
        # Get or create the observable variable in workspace
        var = workspace.var(var_name)
        if var is None:
            # If variable doesn't exist, try to get range from config
            var_range = config.get("range") 
            if var_range and len(var_range) == 2:
                workspace.factory(f"{var_name}[{var_range[0]}, {var_range[1]}]")
                var = workspace.var(var_name)
            else:
                raise RuntimeError(f"Variable '{var_name}' not found in workspace and no range provided in config")
        
        # Create a temporary workspace for template dataset creation to avoid conflicts
        temp_ws = ROOT.RooWorkspace("temp_ws_template", "Temporary workspace for template")
        
        # Create FIT_UTILS instance and use handle_dataset to create MC dataset
        from .fit_tools import FIT_UTILS
        var_config = [(var_name, var.getMin(), var.getMax())]
        tools = FIT_UTILS(log_file=None, var_config=var_config)
        
        # Determine if we need binned dataset
        binned = config.get("binned", self.PARAMETERS["binned"])
        
        # Use handle_dataset to create MC dataset in temporary workspace
        mc_dataset = tools.handle_dataset(
            input_tree=tree,
            workspace=temp_ws,  # Use temporary workspace to avoid conflicts
            branches_name=[var_name],  # Assume MC tree has same variable name
            binned_fit=binned,
            hist_bins=config.get("nbins", self.PARAMETERS["nbins"]),
            weight_branch=params["weight_branch"],
            save_rootFile=False
        )
        
        # Import dataset to workspace
        workspace.Import(mc_dataset, ROOT.RooFit.Rename(f"mc_dataset_{pdf_name}"))
        
        # Create template PDF based on binned or unbinned mode
        if binned:
            # For binned mode, mc_dataset is already a RooDataHist
            # Create RooHistPdf directly
            workspace.factory(f"RooHistPdf::{pdf_name}({var_name}, mc_dataset_{pdf_name})")
        else:
            # For unbinned mode, use RooKeysPdf (kernel density estimation)
            # RooKeysPdf::pdf_name(var, dataset, mirrorOption, rho)
            # mirrorOption: RooKeysPdf::NoMirror, Mirror, MirrorBoth, MirrorAsymBoth
            # rho: kernel bandwidth as fraction of data range (typical: 1.0-2.0)
            workspace.factory(
                f"RooKeysPdf::{pdf_name}({var_name}, mc_dataset_{pdf_name}, RooKeysPdf::MirrorBoth, 2.0)"
            )
        return pdf_name

        


# ============================================================================
# PDF Builder Registry
# ============================================================================

class PDFBuilderRegistry:
    """
    Registry for managing available PDF builders.
    
    No distinction between signal and background - they're all just PDFs!
    """
    
    def __init__(self):
        self._builders = {}
        self._register_default_builders()
    
    def _register_default_builders(self):
        """Register all default PDF builders."""
        self.register("bw_gauss", BreitWignerGaussBuilder())
        self.register("double_gauss", DoubleGaussianBuilder())
        self.register("double_gauss_bw", DoubleGaussianBreitWignerBuilder())
        self.register("crystal_ball", CrystalBallBuilder())
        self.register("voigtian", VoigtianBuilder())
        self.register("gaussian", GaussianBuilder())
        self.register("polynomial", PolynomialBuilder())
        self.register("chebychev", ChebychevBuilder())
        self.register("argus", ArgusBGBuilder())
        self.register("exponential", ExponentialBuilder())
        self.register("flat", FlatBuilder())
        self.register("template", TemplateFitBuilder())
    
    def register(self, name: str, builder: PDFBuilder):
        """Register a PDF builder."""
        self._builders[name] = builder
    
    def get_builder(self, name: str) -> PDFBuilder:
        """Get a registered PDF builder."""
        if name not in self._builders:
            raise ValueError(
                f"Unknown PDF type: {name}. "
                f"Available: {list(self._builders.keys())}"
            )
        return self._builders[name]
    
    def list_types(self):
        """List all available PDF types."""
        return list(self._builders.keys())


# Global registry instance
PDF_REGISTRY = PDFBuilderRegistry()

"""
version history:

version: 2.1
- Added TemplateFitBuilder for template-based PDFs using RooKeysPdf or RooHistPdf.
- Improved parameter parsing and formatting in PDFBuilder base class.
date   : 2026-02-04
author : wang zheng
"""