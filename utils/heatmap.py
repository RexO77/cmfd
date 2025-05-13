import numpy as np
import cv2
import matplotlib.pyplot as plt
import torch

def generate_heatmap(image, patch_scores, patch_coords, patch_size, scale_factor=(1,1)):
    """
    Generate a heatmap overlay for forgery detection.
    
    Args:
        image: Input image (OpenCV format, BGR) - Should be the ORIGINAL size image
        patch_scores: List of scores for each patch (higher = more likely forged)
        patch_coords: List of (x, y) coordinates for each patch (relative to potentially RESIZED image)
        patch_size: Size of each square patch (relative to potentially RESIZED image)
        scale_factor: Tuple (scale_w, scale_h) to scale patch coords/size back to original image size. Default is (1,1).
        
    Returns:
        overlay: Image with heatmap overlay
        heatmap_norm: Normalized heatmap (for visualization)
    """
    # Create empty heatmap of same size as ORIGINAL input image
    heatmap = np.zeros(image.shape[:2], dtype=np.float32)
    
    scale_w, scale_h = scale_factor
    
    # Add scores to heatmap, scaling coordinates and patch size back to original
    scaled_patch_size_w = int(patch_size * scale_w)
    scaled_patch_size_h = int(patch_size * scale_h)

    for (x, y), score in zip(patch_coords, patch_scores):
        # Scale coordinates back to original image space
        orig_x = int(x * scale_w)
        orig_y = int(y * scale_h)
        
        # Ensure coordinates + patch size don't exceed original image bounds
        end_y = min(image.shape[0], orig_y + scaled_patch_size_h)
        end_x = min(image.shape[1], orig_x + scaled_patch_size_w)
        
        # Add score to the corresponding region in the full-size heatmap
        heatmap[orig_y:end_y, orig_x:end_x] += score
    
    # Normalize heatmap to [0, 1]
    heatmap_norm = heatmap.copy() # Keep a normalized version for visualization
    if np.max(heatmap_norm) > 0:
        heatmap_norm = heatmap_norm / np.max(heatmap_norm)
    
    # Apply color mapping and create overlay
    heatmap_colored = cv2.applyColorMap((heatmap_norm * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(image, 0.6, heatmap_colored, 0.4, 0)
    
    return overlay, heatmap_norm

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
