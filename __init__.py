"""
ANA_TOOLS - Analysis Tools Package

A collection of tools for Belle II analysis including:
- DRAW: Plotting and visualization utilities
- FIT: Fitting tools for data analysis
- STEERING_TOOLS: Analysis workflow helpers
"""

__version__ = '0.1.0'

# Optional: Import key components to make them available directly from the package
# This allows: from ANA_TOOLS import style_draw
from .DRAW import style_draw, HistStyle, graph_draw, Brush
from .STEERING_TOOLS import BelleAnalysisBase
from .OFFLINE_PROCESS import RDF_process, gMC_topoana, find_decay_indices, sampling_flat_dist, general_sampling, get_lineshape_weight
from .PHY_CALCULATOR import PhysicsCalculator
from .FIT import FIT_IO, QUICK_FIT, perform_2dfit, perform_resonance_fit, perform_chisq_fit,get_effCurve, fit_rho00, fit_rerho1m1,TreeSplitter
from .bin.belle_run_manager import BelleRunManager, RunEntry

# Define what gets imported with "from ANA_TOOLS import *"
__all__ = [
    # Drawing tools
    'style_draw', 'HistStyle', 'graph_draw', 'Brush',
    
    # Analysis tools
    'BelleAnalysisBase',
    
    # Fitting tools
    'FIT_IO', 'QUICK_FIT', 
    'perform_2dfit', 'perform_resonance_fit', 'perform_chisq_fit',
    'get_effCurve', 'fit_rho00', 'fit_rerho1m1',

    # Offline processing
    'RDF_process', 'gMC_topoana', 'find_decay_indices',
    'sampling_flat_dist', 'general_sampling', 'get_lineshape_weight',

    # Physics calculator
    'PhysicsCalculator',

    # Run manager
    'BelleRunManager', 'RunEntry',

    "TreeSplitter"
]