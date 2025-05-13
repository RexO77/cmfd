"""
Copy-Move Forgery Detection inference module.
This module provides functions to detect copy-move forgery in images using the ViT+Siamese model.
"""

import os
import sys
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import math
from PIL import Image
import matplotlib.pyplot as plt
from torchvision import transforms
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Import model
from models.vit_siamese_model import ViTSiameseModel

# Device configuration
def get_device():
    """Get the best available device (CUDA, MPS, or CPU)"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        # On Mac with Apple Silicon, use CPU for better stability
        print("MPS (Apple Silicon) detected - using CPU for better compatibility")
        return torch.device("cpu")
    else:
        return torch.device("cpu")

# Image preprocessing
def preprocess_image(image_path, target_size=224):
    """Preprocess an image for inference
    
    Args:
        image_path: Path to the image
        target_size: Size to resize the image to
        
    Returns:
        preprocessed_image: Tensor ready for the model
        original_image: Original image in BGR format
    """
    # Read image
    if isinstance(image_path, str) or isinstance(image_path, Path):
        # Read from path
        original_image = cv2.imread(str(image_path))
        if original_image is None:
            raise ValueError(f"Could not read image from {image_path}")
        
        # Convert BGR to RGB
        original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(original_image_rgb)
    else:
        # Assume it's already a PIL image or similar
        pil_image = image_path
        # Convert to numpy for OpenCV processing
        original_image = np.array(pil_image)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2BGR)
    
    # Define transformations
    transform = transforms.Compose([
        transforms.Resize((target_size, target_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transformations
    preprocessed_image = transform(pil_image).unsqueeze(0)  # Add batch dimension
    
    return preprocessed_image, original_image

# Load model
def load_model(model_path):
    """Load the ViT+Siamese model
    
    Args:
        model_path: Path to the trained model weights
        
    Returns:
        model: Loaded model
    """
    device = get_device()
    
    # Initialize model
    model = ViTSiameseModel(pretrained=False)
    
    # Load weights
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    
    return model, device

# Generate heatmap from model output
def generate_heatmap_visualization(image, forgery_map, threshold=0.3, alpha=0.4):
    """Generate heatmap visualization
    
    Args:
        image: Original image (BGR format)
        forgery_map: Forgery probability map from model
        threshold: Threshold for forgery detection
        alpha: Transparency for heatmap overlay
        
    Returns:
        overlay: Image with heatmap overlay
        binary_mask: Binary mask of detected forgery regions
    """
    # Make sure forgery_map is 2D
    if len(forgery_map.shape) > 2:
        forgery_map = forgery_map.squeeze()
    
    # Resize forgery map to match image size if needed
    if forgery_map.shape != (image.shape[0], image.shape[1]):
        forgery_map = cv2.resize(forgery_map, (image.shape[1], image.shape[0]))
    
    # Create binary mask based on threshold
    binary_mask = (forgery_map > threshold).astype(np.uint8) * 255
    
    # Apply color mapping to forgery map
    heatmap = cv2.applyColorMap((forgery_map * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Create overlay
    overlay = image.copy()
    mask = forgery_map > threshold
    overlay[mask] = cv2.addWeighted(image[mask], 1-alpha, heatmap[mask], alpha, 0)
    
    # Draw contours around detected regions for better visibility
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
    
    return overlay, binary_mask

# Extract suspicious regions from binary mask
def extract_suspicious_regions(binary_mask, min_area=100):
    """Extract suspicious regions from binary mask
    
    Args:
        binary_mask: Binary mask of detected forgery
        min_area: Minimum area for a region to be considered
        
    Returns:
        regions: List of (x, y, w, h) tuples representing suspicious regions
    """
    # Find contours
    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter contours by area and extract bounding boxes
    regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= min_area:
            x, y, w, h = cv2.boundingRect(contour)
            regions.append((x, y, w, h))
    
    return regions

# Calculate forgery probability
def calculate_forgery_probability(cls_output, final_map):
    """Calculate overall forgery probability
    
    Args:
        cls_output: Classification output from model
        final_map: Final forgery map from model
        
    Returns:
        probability: Forgery probability
        is_forged: Boolean indicating if forgery is detected
    """
    # Get class probabilities from classification output
    probs = F.softmax(cls_output, dim=1)
    forgery_prob_cls = probs[0, 1].item()  # Probability of class 1 (forged)
    
    # Get average forgery probability from map (regions > 0.3)
    map_binary = final_map > 0.3
    if map_binary.sum() > 0:
        # Average probability in suspicious regions
        forgery_prob_map = final_map[map_binary].mean().item()
        # How much of the image is flagged as suspicious
        coverage_ratio = map_binary.sum().item() / final_map.numel()
    else:
        forgery_prob_map = 0.0
        coverage_ratio = 0.0
    
    # Combine probabilities, weighing classification more if coverage is low
    combined_prob = forgery_prob_cls * 0.7 + forgery_prob_map * 0.3
    
    # Adjust based on coverage - higher coverage might indicate more confidence
    # but also penalize if too much of the image is flagged (possible false positive)
    if coverage_ratio > 0.5:
        # Penalize if more than half the image is flagged
        coverage_adjustment = -0.2 * (coverage_ratio - 0.5)
    else:
        # Boost if reasonable amount is flagged
        coverage_adjustment = 0.2 * coverage_ratio
    
    # Calculate final probability with coverage adjustment
    final_probability = min(1.0, max(0.0, combined_prob + coverage_adjustment))
    
    # Determine if image is forged
    is_forged = final_probability >= 0.5
    
    return final_probability, is_forged

# Main detection function
def detect_forgery(image_path, model_path, output_dir=None):
    """Detect copy-move forgery in an image
    
    Args:
        image_path: Path to the image
        model_path: Path to the model weights
        output_dir: Directory to save output visualizations
        
    Returns:
        result: Dictionary with detection results
    """
    # Load model
    model, device = load_model(model_path)
    
    # Preprocess image
    image_tensor, original_image = preprocess_image(image_path)
    image_tensor = image_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs = model(image_tensor)
        
        # Get outputs
        cls_output = outputs['cls_output']
        final_map = outputs['final_map'].squeeze().cpu().numpy()
        seg_output = outputs['seg_output'].squeeze().cpu().numpy()
        sim_heatmap = outputs['sim_heatmap'].squeeze().cpu().numpy()
    
    # Generate heatmap visualization
    overlay, binary_mask = generate_heatmap_visualization(original_image, final_map)
    
    # Extract suspicious regions
    suspicious_regions = extract_suspicious_regions(binary_mask)
    
    # Calculate forgery probability
    forgery_probability, is_forged = calculate_forgery_probability(cls_output, torch.tensor(final_map))
    
    # Save results if output directory is specified
    heatmap_path = None
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        
        # Get image name without extension
        if isinstance(image_path, str):
            image_name = os.path.basename(image_path).split('.')[0]
        else:
            image_name = "result"
        
        # Save heatmap overlay
        heatmap_path = os.path.join(output_dir, f"{image_name}_heatmap.jpg")
        cv2.imwrite(heatmap_path, overlay)
        
        # Create comprehensive visualization
        plt.figure(figsize=(16, 8))
        
        # Original image
        plt.subplot(2, 3, 1)
        plt.imshow(cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Original Image\nForged: {is_forged} ({forgery_probability:.2f})")
        plt.axis('off')
        
        # Similarity heatmap
        plt.subplot(2, 3, 2)
        plt.imshow(sim_heatmap, cmap='jet')
        plt.title("Similarity Heatmap")
        plt.axis('off')
        
        # Segmentation output
        plt.subplot(2, 3, 3)
        plt.imshow(seg_output, cmap='jet')
        plt.title("Segmentation Output")
        plt.axis('off')
        
        # Final forgery map
        plt.subplot(2, 3, 4)
        plt.imshow(final_map, cmap='jet')
        plt.title("Final Forgery Map")
        plt.axis('off')
        
        # Binary mask
        plt.subplot(2, 3, 5)
        plt.imshow(binary_mask, cmap='gray')
        plt.title(f"Binary Mask\n{len(suspicious_regions)} Suspicious Regions")
        plt.axis('off')
        
        # Overlay
        plt.subplot(2, 3, 6)
        plt.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        plt.title("Forgery Overlay")
        plt.axis('off')
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"{image_name}_analysis.jpg"), dpi=300)
        plt.close()
    
    # Return results
    result = {
        "forgery_probability": forgery_probability,
        "forgery_detected": is_forged,
        "suspicious_regions": [(x, y, w, h, 1) for x, y, w, h in suspicious_regions],
        "heatmap_path": heatmap_path
    }
    
    return result

# Function to compare two images for similarity
def predict(image1_path, image2_path, model_path):
    """Compare two images for similarity
    
    Args:
        image1_path: Path to the first image
        image2_path: Path to the second image
        model_path: Path to the model weights
        
    Returns:
        similarity_score: Similarity score between the two images
    """
    # Load model
    model, device = load_model(model_path)
    
    # Preprocess images
    image1_tensor, _ = preprocess_image(image1_path)
    image2_tensor, _ = preprocess_image(image2_path)
    
    image1_tensor = image1_tensor.to(device)
    image2_tensor = image2_tensor.to(device)
    
    # Run inference
    with torch.no_grad():
        outputs1 = model(image1_tensor)
        outputs2 = model(image2_tensor)
        
        # Get features (CLS tokens) from both images
        cls1 = outputs1['cls_output']
        cls2 = outputs2['cls_output']
        
        # Get similarity heat maps
        sim1 = outputs1['sim_heatmap']
        sim2 = outputs2['sim_heatmap']
        
        # Calculate feature similarity
        feature_sim = F.cosine_similarity(cls1, cls2).item()
        
        # Calculate heatmap similarity
        sim1_flat = sim1.view(sim1.size(0), -1)
        sim2_flat = sim2.view(sim2.size(0), -1)
        heatmap_sim = F.cosine_similarity(sim1_flat, sim2_flat).item()
        
        # Combined similarity score
        similarity_score = 0.7 * feature_sim + 0.3 * heatmap_sim
    
    # Normalize to 0-1 range
    similarity_score = (similarity_score + 1) / 2
    
    return similarity_score

# Command line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy-Move Forgery Detection")
    parser.add_argument('--mode', type=str, default='detect', choices=['detect', 'predict'], 
                        help="Mode: 'detect' for single image forgery detection, 'predict' for comparing two images.")
    parser.add_argument('--image', type=str, help="Path to the image (for detect mode)")
    parser.add_argument('--image1', type=str, help="Path to the first image (for predict mode)")
    parser.add_argument('--image2', type=str, help="Path to the second image (for predict mode)")
    parser.add_argument('--model', type=str, default='outputs/checkpoints/most_accuracy_model.pt',
                        help="Path to the model weights")
    parser.add_argument('--output', type=str, default='outputs/predictions',
                        help="Directory to save output visualizations")
    
    args = parser.parse_args()
    
    if args.mode == 'predict':
        if not args.image1 or not args.image2:
            parser.error("--image1 and --image2 are required for predict mode.")
        
        similarity_score = predict(
            args.image1,
            args.image2,
            args.model
        )
        print(f"\nComparing '{os.path.basename(args.image1)}' and '{os.path.basename(args.image2)}':")
        print(f"  Similarity Score: {similarity_score:.4f}")
        if (similarity_score > 0.5): 
             print("  Result: Likely a FORGED pair (score > 0.5)")
        else:
             print("  Result: Likely an ORIGINAL pair (score <= 0.5)")

    elif args.mode == 'detect':
        if not args.image:
            parser.error("--image is required for detect mode.")
            
        result = detect_forgery(
            args.image, 
            args.model, 
            args.output
        )
        
        print(f"\nDetecting forgery within '{os.path.basename(args.image)}':")
        print(f"  Forgery probability: {result['forgery_probability']:.4f}")
        print(f"  Forgery detected: {result['forgery_detected']}")
        print(f"  Number of suspicious regions: {len(result['suspicious_regions'])}")
        if 'heatmap_path' in result and result['heatmap_path']:
            print(f"  Heatmap saved to: {result['heatmap_path']}")
