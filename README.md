# AI Forgery Detective

An advanced Copy-Move Forgery Detection system using Vision Transformer (ViT) and Siamese networks.

## Overview

This project implements a state-of-the-art forgery detection system for images, specifically focusing on copy-move forgeries - where part of an image is copied and pasted elsewhere in the same image, often with the goal of hiding or duplicating objects.

The system uses a combination of:
- Vision Transformer (ViT) backbone for feature extraction
- Siamese network architecture for similarity detection
- Segmentation head for precise localization
- Both classification and localization outputs for comprehensive forgery detection

## System Requirements

- Python 3.8+ 
- PyTorch 2.0+
- Timm library for Vision Transformer models
- 4GB+ RAM recommended
- CUDA-compatible GPU (optional, but recommended for faster processing)
- A Mac with macOS 12.0+ (for Apple Silicon support)

## Quick Start

### 1. Install Dependencies

Install all required dependencies:

```bash
pip install -r requirements.txt
```

This will install PyTorch, Timm, and other necessary libraries.

### 2. Run the Application

Use the optimized launcher script to avoid compatibility issues between PyTorch and Streamlit:

```bash
python run_app.py
```

This runs Streamlit with optimized settings to prevent conflicts.

### 3. Alternative Launch Method

If you want more control over Streamlit settings, run:

```bash
streamlit run ui/app.py --server.fileWatcherType none
```

## For Apple Silicon Mac Users

This application has been optimized for Apple Silicon Macs by using CPU fallback for tensor operations that would cause type compatibility issues on the MPS (Metal Performance Shaders) backend.

If you experience any issues:
1. Make sure to run using `run_app.py` which properly sets the `PYTORCH_ENABLE_MPS_FALLBACK` environment variable
2. Check that your PyTorch version is 2.0+ with MPS support

## Model Architecture

The system uses a state-of-the-art architecture based on:

### Vision Transformer Backbone

A pre-trained Vision Transformer (ViT) forms the backbone for feature extraction. The ViT processes the input image in patches and leverages self-attention mechanisms to capture both local and global relationships.

### Siamese Architecture

A Siamese branch processes the ViT features to compute patch-to-patch similarities. This helps detect duplicated regions within the image by comparing every patch with every other patch.

### Segmentation Head

The segmentation head provides pixel-level localization of potential forgery regions, enabling precise visualization of where the forgery is located.

### Classification Head

The classification head provides an overall forgery probability for the entire image, which helps in binary decision-making (forged vs. authentic).

## Troubleshooting

If you encounter issues:

1. Make sure all dependencies are installed: `pip install -r requirements.txt`
2. Verify model files exist in the outputs/checkpoints directory 
3. Try running with debug mode: `STREAMLIT_LOGGER_LEVEL=debug python run_app.py`
4. Make sure you've renamed the downloaded model to match the expected filename

## Acknowledgments

- This project uses the timm library for Vision Transformer implementation
- Training was performed on the CASIA dataset for copy-move forgery detection
   