# Copy-Move Forgery Detection (CMFD) with Vision Transformers

This repository contains a state-of-the-art Copy-Move Forgery Detection system that uses Vision Transformers (ViT) and a Siamese Network to detect manipulated images. The project is specifically optimized for MacOS, especially Apple Silicon (M-series) devices.

## Overview

Copy-move forgery is a common image manipulation technique where a part of an image is copied and pasted elsewhere in the same image. This tool can:

- Detect if an image has been manipulated using copy-move techniques
- Generate heatmaps highlighting the likely manipulated regions
- Compare two image regions to determine their similarity
- Process images efficiently on Mac devices using Metal Performance Shaders (MPS)

## Features

- **Advanced Architecture**: Combines Vision Transformer (ViT) for feature extraction with a Siamese network for patch similarity comparison
- **Forgery Localization**: Generates detailed heatmaps showing potential manipulated regions
- **Mac Optimization**: Special optimizations for Apple Silicon (M1/M2/M3/M4) processors
- **User Interface**: Streamlit-based interface for easy interaction with the model
- **Accelerated Training**: Uses HuggingFace Accelerate library for faster, memory-efficient training

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- A Mac with macOS 12.0+ (for MPS support)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/cmfd.git
   cd cmfd
