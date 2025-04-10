import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def generate_heatmap(image, patch_scores, patch_coords, patch_size):
    """
    Generate a heatmap overlay for forgery detection.
    
    Args:
        image: Input image (OpenCV format, BGR)
        patch_scores: List of scores for each patch (higher = more likely forged)
        patch_coords: List of (x, y) coordinates for each patch
        patch_size: Size of each square patch
        
    Returns:
        overlay: Image with heatmap overlay
    """
    # Create empty heatmap of same size as input image
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    
    # Add scores to heatmap
    for (x, y), score in zip(patch_coords, patch_scores):
        heatmap[y:y+patch_size, x:x+patch_size] += score
    
    # Normalize heatmap to [0, 1]
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Apply color mapping and create overlay
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay, heatmap

def save_heatmap_visualization(image, heatmap, save_path):
    """Save heatmap visualization to disk"""
    plt.figure(figsize=(12, 6))
    
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(heatmap, cmap='jet')
    plt.title("Forgery Heatmap")
    plt.axis('off')
    plt.colorbar(label='Forgery Probability')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
