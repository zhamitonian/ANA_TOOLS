# Enable relative imports within the package
from .RDataFrame_process import RDF_process

# Define what gets imported with "from OFFLINE_PROCESS import *"
__all__ = ['RDF_process']