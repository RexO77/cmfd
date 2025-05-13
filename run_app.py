#!/usr/bin/env python3
"""
Helper script to run the AI Forgery Detective app with optimal settings
to avoid PyTorch-Streamlit conflicts.
"""

import os
import sys
import subprocess
import platform

# Set environment variables to prevent PyTorch-Streamlit conflicts
os.environ.update({
    'STREAMLIT_SERVER_FILEWATCH_POLL_SECS': '999999',
    'STREAMLIT_SERVER_ENABLE_STATIC_SERVING': 'false',
    'STREAMLIT_SERVER_HEADLESS': 'true',
    'KMP_DUPLICATE_LIB_OK': 'TRUE',
    'OMP_NUM_THREADS': '1',
    'PYTORCH_CUDA_ALLOC_CONF': 'max_split_size_mb:128',
    'PYTORCH_ENABLE_MPS_FALLBACK': '1'  # Force CPU on Apple Silicon for better compatibility
})

def main():
    """Run the Streamlit app with optimal settings"""
    app_path = os.path.join('ui', 'app.py')
    
    # Verify app exists
    if not os.path.exists(app_path):
        print(f"Error: Could not find {app_path}")
        print(f"Current directory: {os.getcwd()}")
        print("Please run this script from the project root directory")
        return 1
    
    # Check for required model file
    model_path = os.path.join('outputs', 'checkpoints', 'most_accuracy_model.pt')
    colab_model_path = 'final_casia_finetuned_model.pth'
    
    if not os.path.exists(model_path):
        print("\n⚠️ Model file not found at:", model_path)
        
        if os.path.exists(colab_model_path):
            print(f"Found Google Colab model at: {colab_model_path}")
            setup_model = input("Would you like to set up the model now? (y/n): ").strip().lower()
            
            if setup_model == 'y':
                try:
                    from setup_model import setup_model as setup_fn
                    success = setup_fn(colab_model_path, model_path)
                    if not success:
                        print("Model setup failed. Please check the error messages above.")
                        return 1
                except Exception as e:
                    print(f"Error setting up model: {e}")
                    print("Please run 'python setup_model.py' manually.")
                    return 1
            else:
                print("Please set up the model by running 'python setup_model.py' before starting the app.")
                print("Or update the model path in the app settings.")
        else:
            print("No model file found. Please download the model and place it in the project directory.")
            print("Then run 'python setup_model.py' to set up the model.")
            return 1
    
    # Check for Apple Silicon and optimize accordingly
    apple_silicon = False
    if platform.system() == "Darwin" and platform.machine() == "arm64":
        apple_silicon = True
        print("\n⚠️ Apple Silicon detected - using CPU rather than MPS for PyTorch")
        print("This avoids tensor type mismatches that can cause crashes")
    
    print("\n=== AI Forgery Detective ===")
    print("Starting Streamlit with optimized settings to avoid PyTorch conflicts...\n")
    
    # Streamlit command with optimized flags
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.fileWatcherType", "none",      # Disable file watching completely
        "--server.runOnSave", "false",           # Don't reload on save
        "--client.toolbarMode", "minimal",       # Minimal UI
        "--logger.level", "error",               # Reduce logging noise
        "--server.maxUploadSize", "10",          # Limit upload size to 10MB
        "--server.maxMessageSize", "50",         # Limit message size
        "--server.enableCORS", "false",          # Disable CORS
        "--server.enableXsrfProtection", "false" # Disable XSRF for local testing
    ]
    
    try:
        # Check if PyTorch and other key dependencies are installed
        try:
            import torch
            import streamlit
            import cv2
            import numpy
            from PIL import Image
        except ImportError as e:
            print(f"\nError: Missing required dependency: {e}")
            print("Installing missing dependencies...")
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
            print("Dependencies installed. Starting app...")
            
        # Start the Streamlit app
        subprocess.run(cmd)
        return 0
    except KeyboardInterrupt:
        print("\nShutting down...")
        return 0
    except Exception as e:
        print(f"\nError running Streamlit: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())