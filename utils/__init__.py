"""
Utility functions for the Copy-Move Forgery Detection project.
This module provides access to dataset handling, image processing,
metrics computation, and MacOS-specific optimizations.
"""

# Import core utility functions
from utils.transforms import get_transforms, convert_to_tensor
from utils.dataset import CMFDataset, is_forged
from utils.heatmap import generate_heatmap, save_heatmap_visualization
from utils.metrics import compute_metrics, compute_classification_metrics
from utils.mac_utils import (
    get_device, 
    optimize_memory, 
    recommend_batch_size, 
    set_pytorch_threads, 
    get_mac_info
)

# Module exports
__all__ = [
    # Transforms
    'get_transforms', 
    'convert_to_tensor',
    
    # Dataset
    'CMFDataset', 
    'is_forged',
    
    # Heatmap
    'generate_heatmap', 
    'save_heatmap_visualization',
    
    # Metrics
    'compute_metrics', 
    'compute_classification_metrics',
    
    # Mac utilities
    'get_device', 
    'optimize_memory', 
    'recommend_batch_size',
    'set_pytorch_threads', 
    'get_mac_info'
]

# Package version
__version__ = '1.0.0'
