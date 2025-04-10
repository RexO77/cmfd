import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from models.vit_encoder import ViTEncoder
from models.siamese import SiameseNetwork
from utils.transforms import convert_to_tensor
from utils.heatmap import generate_heatmap, save_heatmap_visualization
from utils.mac_utils import get_device, optimize_memory

def load_models(model_path):
    """Load saved models"""
    device = get_device()
    
    # Initialize models
    vit = ViTEncoder(pretrained=False)
    siamese = SiameseNetwork()
    
    # Move models to device
    vit.to(device)
    siamese.to(device)
    
    # Load saved model
    if os.path.exists(model_path):
        checkpoint = torch.load(model_path, map_location=device)
        vit.load_state_dict(checkpoint['vit_state_dict'])
        siamese.load_state_dict(checkpoint['siamese_state_dict'])
        print(f"Model loaded from {model_path}")
    else:
        print(f"Warning: Model not found at {model_path}, using untrained model")
    
    # Set models to evaluation mode
    vit.eval()
    siamese.eval()
    
    return vit, siamese, device

def predict(image_path1, image_path2, model_path="outputs/checkpoints/best_model.pt"):
    """
    Predict similarity between two image patches
    
    Args:
        image_path1: Path to first image
        image_path2: Path to second image
        model_path: Path to saved model
        
    Returns:
        similarity_score: Probability of forgery (0-1)
    """
    # Load model
    vit, siamese, device = load_models(model_path)
    
    # Load and preprocess images
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)
    
    if img1 is None:
        print(f"Error: Could not load image {image_path1}")
        return 0.0
    if img2 is None:
        print(f"Error: Could not load image {image_path2}")
        return 0.0
    
    # Convert to RGB and tensor
    img1_rgb = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    img2_rgb = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    img1_tensor = convert_to_tensor(img1_rgb).unsqueeze(0).to(device)
    img2_tensor = convert_to_tensor(img2_rgb).unsqueeze(0).to(device)
    
    # Forward pass
    with torch.no_grad():
        feat1 = vit(img1_tensor)
        feat2 = vit(img2_tensor)
        score = siamese(feat1, feat2)
    
    # Clean up memory
    optimize_memory()
    
    return float(score.item())

def detect_forgery(image_path, model_path="outputs/checkpoints/best_model.pt", 
                  patch_size=64, stride=32, threshold=0.7, 
                  output_dir="outputs/predictions"):
    """
    Detect copy-move forgery in a single image by comparing all patches
    
    Args:
        image_path: Path to the image
        model_path: Path to the saved model
        patch_size: Size of patches to compare
        stride: Stride between patches
        threshold: Threshold for forgery detection
        output_dir: Directory to save the output heatmap
        
    Returns:
        result: Dict with forgery probability and heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    vit, siamese, device = load_models(model_path)
    
    # Load image
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return {"forgery_probability": 0.0, "forgery_detected": False}
    
    img_height, img_width = img.shape[:2]
    
    # Create patch coordinates
    patch_coords = []
    for y in range(0, img_height - patch_size + 1, stride):
        for x in range(0, img_width - patch_size + 1, stride):
            patch_coords.append((x, y))
    
    # Extract patches
    patches = []
    for x, y in patch_coords:
        patch = img[y:y+patch_size, x:x+patch_size]
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patch_tensor = convert_to_tensor(patch_rgb)
        patches.append(patch_tensor)
    
    # Get features for all patches
    patch_features = []
    with torch.no_grad():
        for i in range(0, len(patches), 8):  # Process in batches to avoid memory issues
            batch = torch.stack(patches[i:i+8]).to(device)
            features = vit(batch)
            patch_features.append(features.cpu())
            # Free up memory after each batch
            optimize_memory()
    
    patch_features = torch.cat(patch_features, dim=0)
    
    # Compare all pairs of patches
    suspicious_pairs = []
    similarity_scores = []
    n_patches = len(patch_features)
    
    # Calculate similarity matrix in a memory-efficient way
    for i in range(n_patches):
        for j in range(i+1, n_patches):
            # Skip patches that are too close to each other
            xi, yi = patch_coords[i]
            xj, yj = patch_coords[j]
            
            # Skip adjacent patches (likely to be similar but not forgery)
            distance = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            if distance < patch_size * 2:
                continue
                
            # Calculate similarity
            with torch.no_grad():
                feat_i = patch_features[i].unsqueeze(0).to(device)
                feat_j = patch_features[j].unsqueeze(0).to(device)
                similarity = siamese(feat_i, feat_j).item()
            
            # Check if similarity is above threshold
            if similarity > threshold:
                suspicious_pairs.append((i, j))
                similarity_scores.append(similarity)
    
    # Create heatmap from suspicious pairs
    patch_scores = np.zeros(len(patch_coords))
    for (i, j), score in zip(suspicious_pairs, similarity_scores):
        patch_scores[i] += score
        patch_scores[j] += score
    
    # Normalize scores
    if len(suspicious_pairs) > 0:
        patch_scores = patch_scores / max(1, np.max(patch_scores))
    
    # Generate heatmap
    heatmap_overlay, heatmap = generate_heatmap(img.copy(), patch_scores, patch_coords, patch_size)
    
    # Save results
    image_name = os.path.basename(image_path).split('.')[0]
    cv2.imwrite(os.path.join(output_dir, f"{image_name}_heatmap.jpg"), heatmap_overlay)
    save_heatmap_visualization(img, heatmap, os.path.join(output_dir, f"{image_name}_visualization.jpg"))
    
    # Determine forgery probability from suspicious pairs
    forgery_probability = len(suspicious_pairs) / max(1, (n_patches * (n_patches - 1) / 2))
    forgery_detected = forgery_probability > 0.01  # Very small threshold
    
    # Free up memory
    optimize_memory()
    
    return {
        "forgery_probability": forgery_probability,
        "forgery_detected": forgery_detected,
        "suspicious_pairs": suspicious_pairs,
        "heatmap_path": os.path.join(output_dir, f"{image_name}_heatmap.jpg")
    }

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy-Move Forgery Detection")
    parser.add_argument('--image', type=str, required=True, help="Path to the image")
    parser.add_argument('--model', type=str, default="outputs/checkpoints/best_model.pt", help="Path to the model")
    parser.add_argument('--output', type=str, default="outputs/predictions", help="Output directory")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch size")
    parser.add_argument('--stride', type=int, default=32, help="Stride between patches")
    parser.add_argument('--threshold', type=float, default=0.7, help="Threshold for forgery detection")
    
    args = parser.parse_args()
    
    result = detect_forgery(
        args.image, 
        args.model, 
        args.patch_size, 
        args.stride, 
        args.threshold, 
        args.output
    )
    
    print(f"Forgery probability: {result['forgery_probability']:.4f}")
    print(f"Forgery detected: {result['forgery_detected']}")
    print(f"Number of suspicious pairs: {len(result['suspicious_pairs'])}")
    print(f"Heatmap saved to: {result['heatmap_path']}")
