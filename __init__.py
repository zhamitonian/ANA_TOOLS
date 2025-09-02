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
from .STEERING_TOOLS import BelleAnalysisBase, ISRAnalysisTools
from .OFFLINE_PROCESS import RDF_process, gMC_topoana, find_decay_indices
from .PHY_CALCULATOR import PhysicsCalculator
from .FIT import FIT_UTILS, QUICK_FIT, perform_2dfit, perform_resonance_fit, perform_chisq_fit,get_effCurve

# Define what gets imported with "from ANA_TOOLS import *"
__all__ = [
    # Drawing tools
    'style_draw', 'HistStyle', 'graph_draw', 'Brush',
    
    # Analysis tools
    'BelleAnalysisBase', 'ISRAnalysisTools',
    
    # Fitting tools
    'FIT_UTILS', 'QUICK_FIT', 
    'perform_2dfit', 'perform_resonance_fit', 'perform_chisq_fit',
    'get_effCurve',

    # Offline processing
    'RDF_process', 'gMC_topoana', 'find_decay_indices',

    # Physics calculator
    'PhysicsCalculator'
]