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
from .DRAW import style_draw, HistStyle, Brush
from .STEERING_TOOLS import BelleAnalysisBase, ISRAnalysisTools
from .OFFLINE_PROCESS import RDF_process, gMC_topoana
from .PHY_CALCULATOR import PhysicsCalculator
from .FIT.fit_tools import FIT_UTILS, QUICK_FIT

# Define what gets imported with "from ANA_TOOLS import *"
__all__ = [
    # Drawing tools
    'style_draw', 'HistStyle', 'Brush',
    
    # Analysis tools
    'BelleAnalysisBase', 'ISRAnalysisTools',
    
    # Fitting tools
    'FIT_UTILS', 'QUICK_FIT',

    # Offline processing
    'RDF_process', 'gMC_topoana',

    # Physics calculator
    'PhysicsCalculator'
]