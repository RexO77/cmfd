import torch
import platform
import os
import psutil
import gc

def get_device():
    """
    Get the appropriate device for PyTorch on macOS.
    Prioritizes MPS (Apple Silicon GPU) for M-series Macs.
    """
    if torch.backends.mps.is_available():
        print("Using MPS (Metal Performance Shaders) device")
        return torch.device("mps")
    elif torch.cuda.is_available():
        print("Using CUDA device")
        return torch.device("cuda")
    else:
        print("Using CPU device")
        return torch.device("cpu")

def optimize_memory():
    """Free memory cache to optimize performance"""
    gc.collect()
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    if torch.backends.mps.is_available():
        # Clear MPS cache if possible
        torch.mps.empty_cache()

def optimize_memory_for_m_series():
    """Specific optimizations for M-series Macs"""
    if torch.backends.mps.is_available():
        # Empty cache to free up memory
        torch.mps.empty_cache()
        
        try:
            # Set memory fraction to avoid OOM errors
            # Adjust this value based on your specific workload
            torch.mps.set_per_process_memory_fraction(0.7)
        except AttributeError:
            # This function might not be available in all PyTorch versions
            print("Warning: Could not set memory fraction for MPS")

def get_mac_info():
    """Return information about the Mac system"""
    mac_info = {
        "system": platform.system(),
        "version": platform.version(),
        "processor": platform.processor(),
        "architecture": platform.machine(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "memory_total": round(psutil.virtual_memory().total / (1024**3), 2),  # GB
        "memory_available": round(psutil.virtual_memory().available / (1024**3), 2),  # GB
        "cpu_count": os.cpu_count()
    }
    
    # Add Apple Silicon specific info
    if platform.machine() == 'arm64' and platform.system() == 'Darwin':
        mac_info["apple_silicon"] = True
        if torch.backends.mps.is_available():
            try:
                mac_info["mps_current_allocated"] = round(torch.mps.current_allocated_memory() / (1024**3), 2)  # GB
                mac_info["mps_driver_allocated"] = round(torch.mps.driver_allocated_memory() / (1024**3), 2)  # GB
            except:
                mac_info["mps_memory_info"] = "Not available"
    else:
        mac_info["apple_silicon"] = False
    
    return mac_info

def recommend_batch_size():
    """Recommend a batch size based on available memory and device"""
    if torch.backends.mps.is_available():
        # For Apple Silicon - use dynamic sizing based on available memory
        try:
            total_memory = torch.mps.current_allocated_memory() + torch.mps.driver_allocated_memory()
            memory_gb = total_memory / (1024**3)
            
            # If memory info is available, use it to determine batch size
            if memory_gb > 0:
                # Base batch size on available memory
                base_batch_size = 32
                memory_factor = psutil.virtual_memory().available / (1024**3) / 8  # Divide by 8 to be conservative
                return max(4, min(32, int(base_batch_size * memory_factor)))
        except:
            pass
        
        # Fallback based on total system memory for M-series
        memory_gb = psutil.virtual_memory().available / (1024**3)
        if memory_gb > 24:  # High-end M2/M3/M4 Pro/Max/Ultra
            return 24
        elif memory_gb > 16:  # Mid-range M-series
            return 16
        elif memory_gb > 8:  # Base M-series
            return 8
        else:
            return 4
    else:
        # CPU-based batch size recommendation
        memory_gb = psutil.virtual_memory().available / (1024**3)
        if memory_gb > 16:
            return 8
        elif memory_gb > 8:
            return 4
        else:
            return 2

def set_pytorch_threads():
    """Set PyTorch CPU threads to optimize performance"""
    cpu_count = os.cpu_count()
    if cpu_count:
        # For M-series, use more threads as they're more efficient
        if platform.machine() == 'arm64' and platform.system() == 'Darwin':
            # Use most cores but leave 1-2 for system
            optimal_threads = max(1, cpu_count - 1)
        else:
            # For Intel, leave more headroom
            optimal_threads = max(1, cpu_count - 2)
            
        torch.set_num_threads(optimal_threads)
        return optimal_threads
    return None

def get_accelerate_config():
    """Get optimal Accelerate configuration for the current device"""
    # Check if running on Apple Silicon
    is_apple_silicon = platform.machine() == 'arm64' and platform.system() == 'Darwin'
    
    config = {
        # Use "no" for mixed precision since fp16 isn't supported on MPS
        "mixed_precision": "no",
        "gradient_accumulation_steps": 4,
        "cpu_offload": False,
        "enable_accelerate": True
    }
    
    # Adjust gradient accumulation based on available memory
    memory_gb = psutil.virtual_memory().available / (1024**3)
    if memory_gb > 24:
        config["gradient_accumulation_steps"] = 2
    elif memory_gb < 8:
        config["gradient_accumulation_steps"] = 8
    
    return config
