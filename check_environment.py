#!/usr/bin/env python3
"""
Check environment for CMFD app prerequisites and ensure all dependencies are correctly installed.
This script helps diagnose installation issues before running the main application.
"""

import os
import sys
import platform
import subprocess
import importlib.util

def check_dependency(module_name, minimum_version=None, pip_name=None):
    """Check if a Python module is installed and has minimum version if specified"""
    if pip_name is None:
        pip_name = module_name
        
    print(f"Checking {module_name}... ", end="")
    
    # Special case for OpenCV which has a different module name
    if module_name == "opencv-python":
        try:
            import cv2
            print(f"✅ Found (cv2 version {cv2.__version__})")
            return True
        except ImportError:
            print(f"Not found. Installing {pip_name}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
                try:
                    import cv2
                    print(f"✅ Installed successfully (cv2 version {cv2.__version__})")
                    return True
                except ImportError:
                    print(f"❌ Failed to install OpenCV properly")
                    return False
            except Exception as e:
                print(f"❌ Failed to install: {e}")
                return False
        return True  # OpenCV is available

    # For other modules
    spec = importlib.util.find_spec(module_name)
    
    if spec is None:
        print(f"Not found. Installing {pip_name}...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", pip_name])
            spec = importlib.util.find_spec(module_name)
            if spec is None:
                print(f"❌ Failed to install {module_name}")
                return False
            else:
                print(f"✅ Installed successfully")
                return True
        except Exception as e:
            print(f"❌ Failed to install: {e}")
            return False
    
    if minimum_version:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"Found version {version}")
            # TODO: Add version comparison if needed
            return True
        except:
            print(f"✅ Found but couldn't determine version")
            return True
    else:
        print(f"✅ Found")
        return True

def check_pytorch():
    """Check PyTorch specifically, with MPS support detection for Apple Silicon"""
    print("\nChecking PyTorch installation...")
    
    if not check_dependency("torch"):
        return False
    
    import torch
    print(f"PyTorch version: {torch.__version__}")
    
    # Check CUDA
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        print(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
    else:
        print("❌ CUDA not available")
    
    # Check MPS (Apple Silicon)
    mps_available = False
    try:
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            mps_available = True
            print("✅ MPS (Apple Silicon acceleration) available")
            print("⚠️  Using CPU fallback due to known MPS tensor type issues")
            os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'
    except:
        pass
    
    if not mps_available and not cuda_available:
        print("Using CPU for computation")
    
    return True

def check_system():
    """Check system information"""
    print("\nSystem Information:")
    print(f"Operating System: {platform.system()} {platform.version()}")
    print(f"Python Version: {platform.python_version()}")
    
    # Check if Apple Silicon
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        print("✅ Apple Silicon detected")
        
    # Check memory
    try:
        import psutil
        memory = psutil.virtual_memory()
        print(f"Memory: {memory.total / (1024**3):.1f} GB total, {memory.available / (1024**3):.1f} GB available")
    except:
        print("Could not determine memory information")
    
    return True

def check_model_files():
    """Check if model files exist"""
    print("\nChecking for model files...")
    
    model_paths = [
        "outputs/checkpoints/most_accuracy_model.pt",
        "outputs/checkpoints/best_model.pt"
    ]
    
    found_any = False
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ Found model: {path}")
            found_any = True
        else:
            print(f"❌ Model not found: {path}")
    
    if not found_any:
        print("\n⚠️  No pretrained models found. You'll need to either:")
        print("  1. Train your own model, or")
        print("  2. Download pretrained models to the outputs/checkpoints directory")
    
    return True

def main():
    """Run all environment checks"""
    print("=" * 50)
    print("CMFD Environment Check")
    print("=" * 50)
    
    # Check system
    check_system()
    
    # Check dependencies
    print("\nChecking dependencies...")
    dependencies = [
        ("numpy", None),
        ("opencv-python", None),  # Will check for cv2
        ("PIL", None, "Pillow"),
        ("streamlit", None),
        ("plotly", None),
        ("tqdm", None),
    ]
    
    for dep in dependencies:
        if len(dep) == 2:
            check_dependency(dep[0], dep[1])
        else:
            check_dependency(dep[0], dep[1], dep[2])
    
    # Check PyTorch
    check_pytorch()
    
    # Check model files
    check_model_files()
    
    # Final report
    print("\n" + "=" * 50)
    print("Environment check complete.")
    print("\nTo run the application:")
    print("  1. Run the optimized launcher: python run_app.py")
    print("  2. Or run directly with: streamlit run ui/app.py --server.fileWatcherType none")
    print("=" * 50)
    
    return 0

if __name__ == "__main__":
    sys.exit(main()) 