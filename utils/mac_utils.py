# 📁 utils/mac_utils.py

# Add these imports
import torch
import platform
import os
import psutil
import gc
from accelerate import Accelerator

def get_device():
    """Get the appropriate device for PyTorch on macOS"""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def optimize_memory():
    """Free memory cache to optimize performance"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()

def set_accelerate_config():
    """Set optimal Accelerate configurations for MacBook Pro"""
    # Check if running on Apple Silicon
    is_apple_silicon = platform.processor() == '' and platform.machine() == 'arm64'
    
    # Determine appropriate mixed precision based on hardware
    if is_apple_silicon:
        # Apple Silicon M-series supports fp16 but let's be safe
        mixed_precision = "fp16"
        cpu_offload = False
    else:
        # Older Intel Macs - better to use no mixed precision
        mixed_precision = "no"
        cpu_offload = True
        
    # Recommended gradient accumulation steps based on memory
    memory_gb = psutil.virtual_memory().available / (1024**3)
    if memory_gb > 12:
        grad_acc_steps = 2
    else:
        grad_acc_steps = 4
        
    return {
        "mixed_precision": mixed_precision,
        "gradient_accumulation_steps": grad_acc_steps,
        "cpu_offload": cpu_offload
    }
