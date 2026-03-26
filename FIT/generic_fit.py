"""
Generic fit class that works for all fit scenarios.

This is the most flexible approach - a single GenericFit class where users specify:
1. Variables and their ranges
2. PDFs to build (no signal/background distinction)
3. Model structure with operations (convolution, product, sum)
4. Plotting configuration
"""

import ROOT
from ROOT import RooFit as rf
from ROOT import RooStats
import time
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple
from .utils.handle_fit_io import FIT_IO
from .pdf_builders import PDF_REGISTRY
from .model_parser import ModelParser


"""
generic fit framework
version : 2.1.7
Date    : 2026-03-23
Author  : wangzheng
"""


# ============================================================================
# Configuration Classes
# ============================================================================

@dataclass
class Variable:
    """Single variable definition."""
    name: str
    min: float
    max: float
    nbin: int = 60  # Number of bins for this variable


@dataclass
class PDFSpec:
    """PDF specification."""
    name: str                                  # Unique PDF name
    var: str                                   # Variable name this PDF depends on
    type: str                                  # PDF type (from registry)
    config: Dict[str, Any] = field(default_factory=dict)  # PDF builder configuration


@dataclass
class FitDefinition:
    """Core fit definition - the physics content."""
    variables: List[Variable]                  # Variable definitions
    pdfs: List[PDFSpec]                        # PDF specifications
    model: str                                  # Model formula
    
    def __post_init__(self):
        """Validate the fit definition."""
        # Check that all PDF variables exist
        var_names = {v.name for v in self.variables}
        for pdf in self.pdfs:
            if pdf.var not in var_names:
                raise ValueError(
                    f"PDF '{pdf.name}' references unknown variable '{pdf.var}'. "
                    f"Available variables: {list(var_names)}"
                )
        
        # Check for duplicate PDF names
        pdf_names = [pdf.name for pdf in self.pdfs]
        duplicates = [name for name in pdf_names if pdf_names.count(name) > 1]
        if duplicates:
            raise ValueError(f"Duplicate PDF names found: {set(duplicates)}")
        
        # Check for duplicate variable names
        var_name_list = [v.name for v in self.variables]
        var_duplicates = [name for name in var_name_list if var_name_list.count(name) > 1]
        if var_duplicates:
            raise ValueError(f"Duplicate variable names found: {set(var_duplicates)}")
    
    def get_variable_tuples(self) -> List[Tuple[str, float, float]]:
        """Convert to legacy tuple format for backward compatibility."""
        return [(v.name, v.min, v.max) for v in self.variables]


@dataclass
class PlotConfiguration:
    """
    Plotting configuration.

    Attributes:
        plot_vars:
            Variables to plot. If None, all fit variables are plotted.

        plot_config:
            Optional dict controlling plotting behavior. Supported keys:

            - show_pull (bool, default: True)
                Whether to draw the pull panel.

            - show_legend (bool, default: True)
                Whether to draw legend.

            - xlabel (dict[str, str])
                Per-variable x-axis labels, e.g. {"Ks_M": "M_{#pi^{+}#pi^{-}} (GeV/c^{2})"}.

            - ylabel (dict[str, str])
                Per-variable y-axis labels.

            - components (dict)
                Plot style for total model and components.
                "model" controls the total fit line;
                other keys should match PDF/component names.
                Each component style supports:
                    {"label": str, "color": int, "style": int, "width": int}

            - legend (dict)
                Legend options:
                    x1, x2, y2 (float): legend box position
                    show_chi2 (bool, default: True)
                    show_yields (bool, default: True)
                    extra_text (list[str])

            - logy (bool | dict[str, bool], default: False)
                Control log-scale y-axis plotting.
                - bool: apply to all plotted variables
                - dict: per-variable override, e.g. {"Ks_M": True}

            - y_range (tuple[float, float] | dict)
                Control y-axis range.
                - tuple/list: (ymin, ymax) for all variables
                - dict[str, tuple/list]: per-variable range, e.g. {"Ks_M": (1, 5000)}
                - dict[str, dict]: per-variable min/max, e.g. {"Ks_M": {"min": 1, "max": 5000}}
                Note: when `logy=True` and `y_range` is not set, ymax defaults to 10x of frame maximum.
            
            - no-pull layout overrides (used when show_pull=False)
                y_title_offset_no_pull (float, default: 1.2)
                x_label_size_no_pull (float, default: 0.045)
                x_title_size_no_pull (float, default: 0.045)
                x_title_offset_no_pull (float, default: 1.0)
                legend_entry_height_no_pull (float, default: 0.035)

    Example:
        PlotConfiguration(plot_config={
            "show_pull": False,
            "xlabel": {"Ks_M": "M_{#pi^{+}#pi^{-}} (GeV/c^{2})"},
            "logy": {"Ks_M": True},
            "y_range": {"Ks_M": (1, 8000)},
            "components": {
                "model": {"label": "Total Fit", "color": 4, "style": 1, "width": 2},
                "bkg": {"label": "Background", "color": 2, "style": 2, "width": 2},
            },
            "legend": {"extra_text": ["Example"]},
        })
    """
    plot_vars: Optional[List[str]] = None      # Variables to plot (None = all)
    plot_config: Dict[str, Any] = field(default_factory=dict)  # Detailed plot settings


@dataclass
class FitterConfig:
    """Fitter control parameters."""
    num_cpu: int = 4
    print_level: int = 3
    strategy: int = 2
    minimizer: str = "Minuit2"
    algorithm: str = "migrad"
    two_step_fit: bool = True# Enable two-step fit: 1) quick fit without Hesse, 2) precise error calculation
    use_minos: bool = False# Use Minos for asymmetric errors (slow but accurate)


@dataclass
class DatasetConfig:
    """Dataset creation and processing configuration."""
    binned_fit: bool = False
    weight_branch: Optional[str] = None
    target_branch: Optional[List[str]] = None
    perform_splot: bool = False
    
    def __post_init__(self):
        """Validate dataset configuration."""
        if self.binned_fit and self.perform_splot:
            raise ValueError(
                "Cannot perform sPlot with binned fit. "
                "sPlot requires unbinned data. "
                "Set binned_fit=False to use sPlot."
            )


class GenericFit:
    """
    Universal fit class that handles all fitting scenarios.
    
    Key insight: All fits share the same workflow, only PDF configurations differ.
    
    For detailed examples and usage, see REFACTORING_DESIGN.md
    
    Quick example:
        fit_def = FitDefinition(
            variables=[Variable("phi_M", 1.0, 1.06, nbin=100)],
            pdfs=[
                PDFSpec(
                    name="sig",
                    var="phi_M",
                    type="bw_gauss",
                    config={"mass": 1.0195, "width": 0.004249}
                ),
                PDFSpec(
                    name="bkg",
                    var="phi_M",
                    type="polynomial",
                    config={"order": 1}
                )
            ],
            model="nsig[100,0,10000]*sig + nbkg[50,0,10000]*bkg"
        )
        fit = GenericFit(
            tree=tree,
            output_dir="output/",
            fit_definition=fit_def
        )
        fit.run()
    """
    
    def __init__(self,
                 # Core inputs (required)
                 tree: ROOT.TTree,
                 output_dir: str,
                 fit_definition: FitDefinition,
                 
                 # Optional configurations
                 plot_config: Optional[PlotConfiguration] = None,
                 fitter_config: Optional[FitterConfig] = None,
                 dataset_config: Optional[DatasetConfig] = None,
                 
                 # MC constraints (for fixing parameters from MC fit)
                 mc_constrains: Optional[Tuple[Any, List[str]]] = None,
                 
                 # Other
                 log_file: Optional[str] = None,
                 ):
        """
        Initialize generic fit.
        
        Args:
            tree: Input ROOT TTree
            output_dir: Output directory for plots
            fit_definition: FitDefinition with variables, pdfs, and model
            plot_config: PlotConfiguration (optional)
            fitter_config: FitterConfig (optional)
            dataset_config: DatasetConfig (optional)
            log_file: Optional log file path
        """
        self.tree = tree
        self.output_dir = output_dir
        self.fit_def = fit_definition
        self.log_file = log_file
        
        # Apply configurations with defaults
        self.plot_cfg = plot_config or PlotConfiguration()
        self.fitter_cfg = fitter_config or FitterConfig()
        self.dataset_cfg = dataset_config or DatasetConfig()
        
        # MC constraints: (result, param_names)
        self.mc_constrains = mc_constrains
        
        # Derived attributes
        self.variables = self.fit_def.variables
        self.pdfs = self.fit_def.pdfs
        self.model_str = self.fit_def.model
        self.plot_vars = self.plot_cfg.plot_vars or [v.name for v in self.variables]
        
        # Create variable lookup dict for easy access
        self._var_dict = {v.name: v for v in self.variables}
        
        # Internal state
        self.workspace = None
        self.dataset = None
        self.model = None
        self.result = None
        self.tools = None
        self.fit_results = {}
        
        # PDF registry
        self.pdf_registry = PDF_REGISTRY
    
    def initialize_workspace(self):
        """Initialize workspace and tools."""
        # Convert Variable objects to tuples for FIT_UTILS
        var_tuples = self.fit_def.get_variable_tuples()
        self.tools = FIT_IO(log_file=self.log_file, var_config=var_tuples)
        self.workspace = ROOT.RooWorkspace("w", "workspace")
    
    def create_dataset(self):
        """Create dataset from tree."""
        # Use nbin from first variable (or could aggregate differently)
        nbin = self.variables[0].nbin if self.variables else 60
        
        self.dataset = self.tools.handle_dataset(
            self.tree,
            self.workspace,
            self.dataset_cfg.target_branch,
            self.dataset_cfg.binned_fit,
            nbin,
            weight_branch=self.dataset_cfg.weight_branch,
        )
        
        self.workspace.Import(self.dataset, ROOT.RooFit.Rename("dataset"))
    
    def build_pdfs(self):
        """Build all PDFs."""
        for pdf_spec in self.pdfs:
            try:
                builder = self.pdf_registry.get_builder(pdf_spec.type)
                builder.build(self.workspace, pdf_spec.var, pdf_spec.config, pdf_spec.name)
            except Exception as e:
                raise RuntimeError(
                    f"Failed to build PDF '{pdf_spec.name}' of type '{pdf_spec.type}': {e}"
                ) from e
    
    def build_model(self):
        """Build model: resolve intermediate ops, then retrieve the final PDF."""
        parser = ModelParser(self.workspace)
        self.yield_vars = parser.parse_model(self.model_str, "model")
        self.model = self.workspace.pdf("model")
    
    def fix_parameters_from_result(self, result: Any, param_names: List[str]):
        """Fix parameters from a previous fit result (e.g., MC fit)."""
        if not self.workspace:
            raise RuntimeError("Workspace not initialized.")
        
        print("\n=== Fixing parameters from MC fit ===")
        fixed_count = 0
        missing_params = []
        
        for param_name in param_names:
            param_from_result = result.floatParsFinal().find(param_name)
            
            if not param_from_result:
                missing_params.append(param_name)
                continue
            
            value = param_from_result.getVal()
            error = param_from_result.getError()
            
            param_in_workspace = self.workspace.var(param_name)
            
            if param_in_workspace:
                param_in_workspace.setVal(value)
                param_in_workspace.setConstant(True)
                print(f"  Fixed {param_name} = {value:.6f} ± {error:.6f}")
                fixed_count += 1
            else:
                missing_params.append(param_name)
        
        if missing_params:
            print(f"  Warning: Parameters not found: {missing_params}")
        
        print(f"=== Fixed {fixed_count}/{len(param_names)} parameters ===\n")
        return fixed_count
    
    def perform_fit(self):
        """Perform the fit."""
        if not self.dataset or not self.model:
            print("Error: dataset or model is None")
            return None
        
        print(f"Dataset entries: {self.dataset.numEntries()}")
        if self.dataset.numEntries() == 0:
            print("Error: dataset is empty")
            return None
        
        # Common fit options
        fit_options = [
            rf.Save(True),
            rf.NumCPU(self.fitter_cfg.num_cpu),
            rf.PrintLevel(self.fitter_cfg.print_level),
            rf.Strategy(self.fitter_cfg.strategy),
            rf.Minimizer(self.fitter_cfg.minimizer, self.fitter_cfg.algorithm),
            rf.SumW2Error(not self.fitter_cfg.use_minos)  # correctly calculate errors with weights
        ]
        
        if self.fitter_cfg.two_step_fit:
            # Step 1
            print("\n=== Step 1: Finding minimum (Hesse=False) ===")
            step1_options = fit_options + [rf.Hesse(False), rf.Minos(False)]
            self.model.fitTo(self.dataset, *step1_options)
            
            # Step 2
            print("\n=== Step 2: Calculating errors (Hesse=True) ===")
            step2_options = fit_options + [rf.Hesse(True), rf.Minos(self.fitter_cfg.use_minos)]
            self.result = self.model.fitTo(self.dataset, *step2_options)
        else:
            # Single-step fit (default behavior)
            fit_options += [
                rf.Hesse(True),  # Always calculate Hesse in single-step mode
                rf.Minos(self.fitter_cfg.use_minos)
            ]
            self.result = self.model.fitTo(self.dataset, *fit_options)
        
        # Check fit quality
        self._check_fit_quality()
        
        return self.result
    
    def _check_fit_quality(self):
        """Check and report fit quality metrics."""
        if not self.result:
            return
        
        status_codes = {
            0: "successful fit",
            1: "covariance was made positive definite",
            2: "Hesse is invalid",
            3: "EDM is above max",
            4: "Reached call limit",
            5: "other failure"
        }
        
        print("*" * 30)
        print(f"Fit status code: {self.result.status()} ({status_codes.get(self.result.status(), 'unknown status code')})")
        print("Covariance quality:", self.result.covQual())
        print("Estimated distance to minimum (EDM):", self.result.edm())
        
        # Check if fit is acceptable
        ok = (self.result.status() == 0) and (self.result.covQual() >= 2) and (self.result.edm() < 1e-3)
        print("Fit OK?", ok)
        print("*" * 30)
    
    def perform_splot(self, yield_vars: Optional[List[str]] = None):
        """
        Perform sPlot.
        
        Args:
            yield_vars: List of yield variable names (default: auto-detect from model)
        """
        if yield_vars is None:
            # Auto-detect yield variables (nsig, nbkg, nbkg1, etc.)
            yield_vars = []
            for var_name in ["nsig", "nbkg", "nbkg1", "nbkg2", "nbkg3"]:
                if self.workspace.var(var_name):
                    yield_vars.append(var_name)
        
        if not yield_vars:
            print("No yield variables found for sPlot")
            return None
        
        yield_list = ROOT.RooArgList()
        for var_name in yield_vars:
            var = self.workspace.var(var_name)
            if var:
                yield_list.add(var)
        
        sData = RooStats.SPlot(
            "sData",
            "sPlot",
            self.dataset,
            self.model,
            yield_list
        )

        sWeighted_data = ROOT.RooDataSet(
            "sWeighted_data",
            "Data with sWeights",
            self.dataset,
            self.dataset.get(),
            "",
            "nsig_sw"  # Assuming nsig is the signal yield variable
        )

        out_file = ROOT.TFile(self.output_dir + "_splot_output.root", "RECREATE")
        sWeighted_data.Write("signal_weighted_data")
        out_file.Close()
        
        return sData
    
    def plot_results(self):
        """
        Plot fit results for all specified variables.
        
        Uses plot_config for customization or falls back to defaults.
        """
        ROOT.gStyle.SetLabelSize(0.04,"xyz")
        ROOT.gStyle.SetPadTopMargin(.1)
        ROOT.gStyle.SetPadLeftMargin(.14)
        ROOT.gStyle.SetPadRightMargin(.07)
        ROOT.gStyle.SetPadBottomMargin(.14)
        ROOT.gStyle.SetTitleSize(0.04,"xyz")
        ROOT.gStyle.SetOptTitle(0)
        ROOT.gStyle.SetMarkerSize(0.5)
        ROOT.gStyle.SetLabelFont(22,"XYZ")
        ROOT.gStyle.SetTitleFont(22,"XYZ")
        ROOT.gStyle.SetCanvasDefH(1080)
        ROOT.gStyle.SetCanvasDefW(1600)
        ROOT.gStyle.SetCanvasColor(ROOT.kWhite)
        ROOT.gStyle.SetPadTickX(1)
        ROOT.gStyle.SetPadTickY(1)
        ROOT.gStyle.SetPadGridX(1)
        ROOT.gStyle.SetPadGridY(1)

        for var_name in self.plot_vars:
            self._plot_single_variable(var_name)
    
    def _plot_single_variable(self, var_name: str):
        """Plot fit result for a single variable."""
        # Get nbin from the variable definition
        var_obj = self._var_dict.get(var_name)
        nbin = var_obj.nbin if var_obj else 60
        
        plot_dict = self.plot_cfg.plot_config
        var = self.workspace.var(var_name)
        x_min = var.getMin()
        x_max = var.getMax()
        
        # Get axis labels
        xlabel = plot_dict.get("xlabel", {}).get(var_name, var_name)
        ylabel = plot_dict.get("ylabel", {}).get(
            var_name, 
            f"Candidates / ({((x_max - x_min) / nbin * 1000):.3f} MeV/c^{{2}})"
        )
        show_pull = plot_dict.get("show_pull", True)

        # Log-scale control: bool for global, or dict for per-variable
        logy_cfg = plot_dict.get("logy", False)
        if isinstance(logy_cfg, dict):
            use_logy = bool(logy_cfg.get(var_name, False))
        else:
            use_logy = bool(logy_cfg)

        # Dedicated defaults when pull panel is hidden (can be overridden in plot_config)
        y_title_offset_no_pull = plot_dict.get("y_title_offset_no_pull", 1.2)
        x_label_size_no_pull = plot_dict.get("x_label_size_no_pull", 0.045)
        x_title_size_no_pull = plot_dict.get("x_title_size_no_pull", 0.045)
        x_title_offset_no_pull = plot_dict.get("x_title_offset_no_pull", 1.0)
        legend_entry_height_no_pull = plot_dict.get("legend_entry_height_no_pull", 0.035)
        
        # Setup canvas
        canvas = ROOT.TCanvas("c", "c", 1600, 1080)
        if show_pull:
            canvas.Divide(1, 2)
        
        # Keep reference to legend to prevent garbage collection
        legend = None
        
        # Upper pad
        pad1 = canvas.cd(1)
        if show_pull:
            pad1.SetPad(0, 0.3, 1, 1)
            pad1.SetBottomMargin(0.01)
        else:
            pad1.SetPad(0, 0, 1, 1)
            pad1.SetBottomMargin(0.12)
        pad1.SetLeftMargin(0.15)
        pad1.SetRightMargin(0.05)
        pad1.SetLogy(1 if use_logy else 0)
        pad1.Draw()
        
        # Create frame and plot
        frame = var.frame()
        self.dataset.plotOn(
            frame,
            rf.Name("data"),
            rf.MarkerColor(ROOT.kBlack),
            rf.MarkerStyle(20),
            rf.Binning(nbin)
        )
        
        # Plot total model with user config if provided
        components = plot_dict.get("components", {})
        model_config = components.get("model", {})
        
        self.model.plotOn(
            frame, 
            rf.Name("sum"), 
            rf.LineColor(model_config.get("color", 4)),
            rf.LineStyle(model_config.get("style", 1)),
            rf.LineWidth(model_config.get("width", 2))
        )
        
        # Plot components if specified (skip "model" as it's already plotted)
        for comp_name, comp_config in components.items():
            if comp_name == "model":  # Skip model, already plotted as total fit
                continue
            
            self.model.plotOn(
                frame,
                rf.Components(comp_name),
                rf.Name(comp_name),
                rf.LineColor(comp_config.get("color", ROOT.kRed)),
                rf.LineStyle(comp_config.get("style", 1)),
                rf.LineWidth(comp_config.get("width", 2))
            )

        # Y-axis range control
        y_range_cfg = plot_dict.get("y_range", None)
        y_min = None
        y_max = None
        if y_range_cfg is not None:
            if isinstance(y_range_cfg, dict):
                var_range = y_range_cfg.get(var_name)
            else:
                var_range = y_range_cfg

            if isinstance(var_range, (tuple, list)) and len(var_range) == 2:
                y_min, y_max = var_range
            elif isinstance(var_range, dict):
                y_min = var_range.get("min")
                y_max = var_range.get("max")

        frame_max = frame.GetMaximum()
        if frame_max <= 0:
            frame_max = 1.0

        # For log-scale, default ymax is 10x maximum when user does not provide y_range
        if y_max is not None:
            frame.SetMaximum(float(y_max))
        elif use_logy:
            frame.SetMaximum(frame_max * 5.0)

        if y_min is not None:
            frame.SetMinimum(float(y_min))
        elif use_logy:
            frame.SetMinimum(min(frame_max * 1e-3, 3))
            #frame.SetMinimum(10)

        frame.SetTitle("")
        frame.GetYaxis().SetTitle(ylabel)
        frame.GetYaxis().SetTitleOffset(1.0 if show_pull else y_title_offset_no_pull)

        if show_pull:
            frame.GetXaxis().SetLabelSize(0)
            frame.GetXaxis().SetTitle("")
        else:
            frame.GetXaxis().SetTitle(xlabel)
            frame.GetXaxis().CenterTitle()
            frame.GetXaxis().SetLabelSize(x_label_size_no_pull)
            frame.GetXaxis().SetTitleSize(x_title_size_no_pull)
            frame.GetXaxis().SetTitleOffset(x_title_offset_no_pull)
        frame.Draw()
        
        # Draw legend on pad1
        if plot_dict.get("show_legend", True):
            legend_entry_h = 0.05 if show_pull else legend_entry_height_no_pull
            legend = self._draw_legend(frame, var_name, nbin, entry_height=legend_entry_h)

        if show_pull:
            # Lower pad: pull distribution
            pad2 = canvas.cd(2)
            pad2.SetPad(0, 0, 1, 0.3)
            pad2.SetTopMargin(0.01)
            pad2.SetBottomMargin(0.3)
            pad2.SetLeftMargin(0.15)
            pad2.SetRightMargin(0.05)
            pad2.Draw()

            pull_frame = var.frame()
            pull_frame.SetTitle("")
            pull_frame.GetYaxis().SetTitle("Pull")
            pull_frame.GetYaxis().SetTitleOffset(0.35)
            pull_frame.GetYaxis().SetTitleSize(0.1)
            pull_frame.GetYaxis().CenterTitle()
            pull_frame.GetYaxis().SetRangeUser(-5, 5)
            pull_frame.GetYaxis().SetLabelSize(0.1)
            pull_frame.GetXaxis().SetLabelSize(0.13)
            pull_frame.GetXaxis().SetTitle(xlabel)
            pull_frame.GetXaxis().CenterTitle()
            pull_frame.GetXaxis().SetTitleSize(0.12)
            pull_frame.GetXaxis().SetTitleOffset(1.1)

            pullhist = frame.pullHist("data", "sum")
            pull_frame.addObject(pullhist, "P")
            pull_frame.Draw()
        
        # Update canvas
        canvas.cd()
        canvas.Modified()
        canvas.Update()
        
        # Save (keep legend reference alive)
        output_path = f"{self.output_dir}_{var_name}.png"
        canvas.SaveAs(output_path)
        print(f"Plot saved: {output_path}")
    
    def _draw_legend(self, frame, var_name: str, nbin: int, entry_height: float = 0.05):
        """Draw legend on the plot and return the legend object."""
        # Get legend configuration
        plot_dict = self.plot_cfg.plot_config
        legend_config = plot_dict.get("legend", {})
        components = plot_dict.get("components", {})
        extra_text = legend_config.get("extra_text", [])
        
        # Calculate legend size automatically
        base_entries = 2  # Data + Total fit
        component_entries = len(components)
        yield_entries = len(self.fit_results) // 2 if legend_config.get("show_yields", True) else 0
        chi2_entries = 1 if (self.result and legend_config.get("show_chi2", True)) else 0
        extra_text_entries = len(extra_text)
        total_entries = base_entries + component_entries + yield_entries + chi2_entries + extra_text_entries
        
        # Legend position
        x1 = legend_config.get("x1", 0.7)
        x2 = legend_config.get("x2", 0.95)
        y2 = legend_config.get("y2", 0.9)
        y1 = y2 - entry_height * total_entries
        
        leg = ROOT.TLegend(x1, y1, x2, y2)
        leg.SetBorderSize(0)
        leg.SetFillStyle(0)
        leg.SetFillColor(0)
        leg.SetTextFont(22)
        
        # Get objects from frame to avoid garbage collection issues
        # Data entry - use actual data histogram from frame
        data_hist = frame.findObject("data")
        if data_hist:
            leg.AddEntry(data_hist, "Data", "pe")
        
        # Total fit entry - use actual curve from frame
        total_curve = frame.findObject("sum")
        if total_curve:
            # Use model label if provided in config, otherwise default
            model_config = components.get("model", {})
            model_label = model_config.get("label", "Total fit")
            leg.AddEntry(total_curve, model_label, "l")
        
        # Component entries - use actual curves from frame (skip "model" as it's the total fit)
        for comp_name, comp_config in components.items():
            if comp_name == "model":  # Skip model, already added as total fit
                continue
            comp_curve = frame.findObject(comp_name)
            if comp_curve:
                label = comp_config.get("label", comp_name)
                leg.AddEntry(comp_curve, label, "l")
        
        # Chi-square/ndf
        if self.result and legend_config.get("show_chi2", True):
            n_float = self.result.floatParsFinal().getSize()
            chi2_val = None
            ndf = None

            # For unbinned input, build a temporary RooDataHist with user binning.
            try:
                var = self.workspace.var(var_name)
                if var:
                    var.setBins(int(nbin))
                    data_for_chi2 = self.dataset
                    if hasattr(self.dataset, "InheritsFrom") and self.dataset.InheritsFrom("RooDataSet"):
                        data_for_chi2 = ROOT.RooDataHist(
                            f"datahist_{var_name}",
                            f"datahist_{var_name}",
                            ROOT.RooArgSet(var),
                            self.dataset,
                        )

                    chi2_var = ROOT.RooChi2Var(
                        f"chi2_{var_name}",
                        f"chi2_{var_name}",
                        self.model,
                        data_for_chi2,
                        rf.DataError(ROOT.RooAbsData.SumW2),
                    )
                    chi2_val = float(chi2_var.getVal())
                    ndf = max(int(nbin) - int(n_float), 1)
            except Exception:
                chi2_val = None
                ndf = None

            # Fallback to frame-based chi2 if RooChi2Var construction fails.
            """
            if chi2_val is None or ndf is None:
                chi2_ndf = frame.chiSquare("sum", "data", n_float)
                data_hist = frame.getHist("data")
                n_bins_eff = data_hist.GetN() if data_hist else nbin
                ndf = max(int(n_bins_eff) - int(n_float), 1)
                chi2_val = chi2_ndf * ndf
            """
            leg.AddEntry(0, f"#chi^{{2}}/ndf = {chi2_val:.1f}/{ndf}", "")

        
        # Yield values - only show nsig (signal yield)
        if legend_config.get("show_yields", True):
            var = self.workspace.var("nsig")
            #nsig = var.getVal() 
            #nsig_err = var.getError()
            nsig = self.fit_results.get("nsig", 0)
            nsig_err = self.fit_results.get("nsig_err", 0)
            leg.AddEntry(0, f"N_{{sig}} = {nsig:.1f} #pm {nsig_err:.1f}", "")
        
        # Extra text entries
        extra_text = legend_config.get("extra_text", [])
        for text in extra_text:
            leg.AddEntry(0, text, "")
        
        leg.Draw()
        return leg
    
    def save_results(self):
        """Save fit results."""
        # Auto-detect and save all yield variables
        for var_name in ["nsig", "nbkg", "nbkg1", "nbkg2", "nbkg3"]:
            var = self.workspace.var(var_name)
            if var:
                self.fit_results[var_name] = var.getVal()
                self.fit_results[f"{var_name}_err"] = var.getError()
    
    def print_summary(self):
        """Print fit summary."""
        print(f"\n=== Fit Complete ===")
        for key, value in self.fit_results.items():
            if not key.endswith("_err"):
                err_key = f"{key}_err"
                if err_key in self.fit_results:
                    print(f"{key}: {value:.2f} ± {self.fit_results[err_key]:.2f}")
        print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"====================\n")
    
    def run(self) -> Tuple[Any, Dict[str, float]]:
        """
        Execute the complete fit workflow.
        
        Returns:
            Tuple: (fit_result, fit_results_dict)
        """
        self.initialize_workspace()

        with self.tools.redirect_output():
            print(f"=== Starting Generic Fit ===")
            print(f"Variables: {[v.name for v in self.variables]}")
            print(f"Output: {self.output_dir}")
            print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print(f"============================")
            
            # Workflow
            self.create_dataset()
            self.build_pdfs()
            self.build_model()
            
            # Fix parameters from MC if provided
            if self.mc_constrains:
                mc_result, param_names = self.mc_constrains
                self.fix_parameters_from_result(mc_result, param_names)
            
            self.perform_fit()
            
            if self.dataset_cfg.perform_splot:
                self.perform_splot()
            
            self.save_results()
            self.plot_results()
            self.print_summary()
        
        return self.result, self.fit_results
    
    def _null_context(self):
        """Null context manager."""
        from contextlib import contextmanager
        @contextmanager
        def _null():
            yield
        return _null()


"""
v2.0.0 
initial version

v2.1.0
- add weight branch support and sumw2error in fit
date : 2026-01-23

v2.1.1
- add 2-step fit option ,
- add para fix from MC fit result
date : 2026-03-09

v2.1.2
- optimize model parser
date : 2026-03-12

v2.1.3
- add more plot config options and show pull control
date : 2026-03-13

v2.1.4
- add y-axis range control in plotting
- add log-scale control in plotting
date : 2026-03-16

v2.1.5
- fix little bug in chi2 calculation in legend (wrong ndf)
date : 2026-03-18

v2.1.6
- fix chi2 calculation 's not compatible with unbinned dataset (RooChi2Var only accepts RooDataHist)
date : 2026-03-23

v2.1.7
- update the use of model parser
date : 2026-03-26
"""



