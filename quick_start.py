#!/usr/bin/env python3
"""
Ultra-simple Streamlit launcher that bypasses all problematic components
"""
import os
import subprocess
import sys

if __name__ == "__main__":
    # Kill any existing Streamlit processes
    os.system("pkill -f streamlit")
    
    # Set absolute minimum environment variables
    os.environ["PYTHONPATH"] = os.path.dirname(os.path.abspath(__file__))
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    
    print("Starting Streamlit app...")
    
    try:
        # Run Streamlit with the simplest possible command
        subprocess.run([
            sys.executable,
            "-m",
            "streamlit",
            "run",
            os.path.join(os.path.dirname(os.path.abspath(__file__)), "ui", "app.py"),
            "--server.port", "8510",
            "--server.fileWatcherType", "none"
        ], check=True)
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)