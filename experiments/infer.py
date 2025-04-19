import sys # Move sys import to the top
from pathlib import Path # Move Path import to the top

# Add project root to Python path (MUST be before other project imports)
sys.path.append(str(Path(__file__).parent.parent))

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm # Import tqdm
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

def process_window(window, vit, siamese, device, patch_size, stride, threshold, offset):
    """Process a single window and return results with original coordinates"""
    x_offset, y_offset = offset
    window_h, window_w = window.shape[:2]
    
    # Patch extraction with original coordinates
    patch_coords = []
    for y in range(0, window_h - patch_size + 1, stride):
        for x in range(0, window_w - patch_size + 1, stride):
            patch_coords.append((x + x_offset, y + y_offset))
    
    if not patch_coords:
        return {'suspicious_pairs': [], 'similarity_scores': [], 'patch_coords': patch_coords}
    
    # Feature extraction
    patches = []
    for i, (x, y) in enumerate(patch_coords):
        # Use local coordinates for extraction, but store global coordinates
        patch = window[y-y_offset:y-y_offset+patch_size, x-x_offset:x-x_offset+patch_size]
        patch_rgb = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
        patches.append(convert_to_tensor(patch_rgb))
    
    # Extract features in batches
    patch_features = []
    feature_batch_size = 16
    with torch.no_grad():
        for i in range(0, len(patches), feature_batch_size):
            batch = torch.stack(patches[i:i+feature_batch_size]).to(device)
            features = vit(batch)
            patch_features.append(features.cpu())
            optimize_memory()
    
    patch_features = torch.cat(patch_features, dim=0)
    
    # Compare all pairs of patches
    suspicious_pairs = []
    similarity_scores = []
    window_indices = []  # To store window-local indices
    n_patches = len(patch_features)
    
    # Move features to GPU for comparison
    patch_features_gpu = patch_features.to(device)
    
    with torch.no_grad():
        for i in range(n_patches):
            batch_indices_j = []
            feat_i = patch_features_gpu[i]

            for j in range(i + 1, n_patches):
                xi, yi = patch_coords[i][0] - x_offset, patch_coords[i][1] - y_offset
                xj, yj = patch_coords[j][0] - x_offset, patch_coords[j][1] - y_offset
                distance = np.sqrt((xi - xj)**2 + (yi - yj)**2)
                
                # Skip adjacent patches
                if distance < patch_size * 1.5:
                    continue
                
                batch_indices_j.append(j)
                
                # Process batch when full or at the end
                if len(batch_indices_j) == 512 or j == n_patches - 1:
                    if batch_indices_j:
                        feat_j_batch = patch_features_gpu[batch_indices_j]
                        feat_i_batch = feat_i.unsqueeze(0).repeat(len(batch_indices_j), 1)
                        
                        similarities_batch = siamese(feat_i_batch, feat_j_batch).squeeze().cpu().numpy()
                        
                        # Handle case where batch size is 1
                        if similarities_batch.ndim == 0:
                            similarities_batch = np.array([similarities_batch])
                        
                        # Check threshold and store suspicious pairs
                        for k, similarity in enumerate(similarities_batch):
                            if similarity > threshold:
                                j_index = batch_indices_j[k]
                                # Store window-local indices for this window's processing
                                window_indices.append((i, j_index))
                                suspicious_pairs.append((patch_coords[i], patch_coords[j_index]))
                                similarity_scores.append(similarity)
                        
                        batch_indices_j = []
    
    # Clean up GPU memory
    if 'patch_features_gpu' in locals():
        del patch_features_gpu
        torch.mps.empty_cache() if torch.backends.mps.is_available() else None
    
    return {
        'suspicious_pairs': suspicious_pairs, 
        'similarity_scores': similarity_scores, 
        'patch_coords': patch_coords,
        'window_indices': window_indices
    }

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

def detect_forgery_window(image_path, model_path="outputs/checkpoints/best_model.pt",
                  patch_size=64, stride=32, threshold=0.85,
                  output_dir="outputs/predictions",
                  max_dim=512):
    """
    Detect copy-move forgery using sliding window approach
    
    Args:
        image_path: Path to the image
        model_path: Path to the model
        patch_size: Size of patches for comparison
        stride: Stride between patches
        threshold: Threshold for suspicious patch similarity
        output_dir: Directory to save output
        max_dim: Maximum dimension for resizing (None for no resizing)
        
    Returns:
        result: Dict with forgery probability and heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    vit, siamese, device = load_models(model_path)
    
    # Load image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not load image {image_path}")
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [], "heatmap_path": None}
    
    # Use sliding window approach for large images
    window_size = (512, 384)  # Your model's optimal working resolution
    all_suspicious_pairs = []
    all_similarity_scores = []
    all_patch_coords = []
    
    h, w = img_orig.shape[:2]
    scale_factor = (1, 1)  # No scaling needed
    
    # Check if image is small enough to process directly
    if h <= window_size[1] and w <= window_size[0]:
        # For small images, process the whole image at once
        window_results = process_window(
            img_orig, vit, siamese, device, patch_size, stride, threshold, (0, 0)
        )
        all_suspicious_pairs = window_results['suspicious_pairs']
        all_similarity_scores = window_results['similarity_scores']
        all_patch_coords = window_results['patch_coords']
    else:
        # For large images, use sliding window approach with overlap
        # Add 50% overlap between windows to catch forgeries at window boundaries
        overlap_factor = 0.5
        y_step = int(window_size[1] * (1 - overlap_factor))
        x_step = int(window_size[0] * (1 - overlap_factor))
        
        total_windows = ((h - 1) // y_step + 1) * ((w - 1) // x_step + 1)
        window_count = 0
        
        for y_start in range(0, h, y_step):
            for x_start in range(0, w, x_step):
                window_count += 1
                print(f"Processing window {window_count}/{total_windows} at ({x_start}, {y_start})")
                
                y_end = min(y_start + window_size[1], h)
                x_end = min(x_start + window_size[0], w)
                window = img_orig[y_start:y_end, x_start:x_end]
                
                # Process each window
                window_results = process_window(
                    window, vit, siamese, device, patch_size, stride, threshold,
                    (x_start, y_start)  # Pass offset coordinates
                )
                
                # Accumulate results
                all_suspicious_pairs.extend(window_results['suspicious_pairs'])
                all_similarity_scores.extend(window_results['similarity_scores'])
                all_patch_coords.extend(window_results['patch_coords'])
                
                # Free memory after processing each window
                optimize_memory()
    
    # If no suspicious pairs found, return early
    if not all_suspicious_pairs:
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [], "heatmap_path": None}
    
    # --- Apply spatial clustering to group suspicious pairs ---
    clusters = cluster_suspicious_pairs(all_suspicious_pairs, all_similarity_scores, patch_size)
    
    # Filter out small clusters (likely false positives)
    min_cluster_size = 3  # Require at least 3 pairs in a cluster to be considered a forgery
    significant_clusters = [c for c in clusters if len(c['pairs']) >= min_cluster_size]
    
    # --- Apply geometric consistency check to validate each cluster ---
    validated_clusters = []
    for cluster in significant_clusters:
        # Check if cluster has geometric consistency (indicating true copy-move)
        is_valid, consistency_score = verify_geometric_consistency(cluster, patch_size)
        
        # Only add clusters that pass geometric validation
        if is_valid:
            cluster['consistency_score'] = consistency_score
            validated_clusters.append(cluster)
    
    # --- If no geometrically valid clusters remain, likely no forgery ---
    if not validated_clusters:
        print("No geometrically consistent forgery clusters found")
        return {
            "forgery_probability": 0.0,
            "forgery_detected": False,
            "suspicious_pairs": all_suspicious_pairs,  # Keep original for debugging
            "heatmap_path": None
        }
    
    # Re-compute suspicious pairs based on validated clusters
    filtered_pairs = []
    filtered_scores = []
    
    for cluster in validated_clusters:
        filtered_pairs.extend(cluster['pairs'])
        filtered_scores.extend(cluster['scores'])
    
    # Convert the filtered pairs to the format needed for heatmap
    processed_pairs = []
    for i, ((x1, y1), (x2, y2)) in enumerate(filtered_pairs):
        # Find indices in all_patch_coords
        idx1 = all_patch_coords.index((x1, y1)) if (x1, y1) in all_patch_coords else -1
        idx2 = all_patch_coords.index((x2, y2)) if (x2, y2) in all_patch_coords else -1
        
        if idx1 >= 0 and idx2 >= 0:
            processed_pairs.append((idx1, idx2))
    
    # Create heatmap from filtered pairs
    patch_scores = np.zeros(len(all_patch_coords))
    for i, ((i1, i2), score) in enumerate(zip(processed_pairs, filtered_scores)):
        patch_scores[i1] += score
        patch_scores[i2] += score
    
    # Normalize scores
    if np.max(patch_scores) > 0:
        patch_scores = patch_scores / np.max(patch_scores)
    
    # Generate heatmap
    heatmap_overlay, heatmap = generate_heatmap(
        img_orig.copy(),
        filtered_scores,
        [pair[0] for pair in filtered_pairs] + [pair[1] for pair in filtered_pairs],
        patch_size,
        scale_factor=scale_factor
    )
    
    # Save results
    image_name = os.path.basename(image_path).split('.')[0]
    heatmap_overlay_path = os.path.join(output_dir, f"{image_name}_heatmap.jpg")
    visualization_path = os.path.join(output_dir, f"{image_name}_visualization.jpg")
    cv2.imwrite(heatmap_overlay_path, heatmap_overlay)
    save_heatmap_visualization(img_orig, heatmap, visualization_path)
    
    # Calculate forgery probability based on validated clusters
    avg_consistency = np.mean([c.get('consistency_score', 0) for c in validated_clusters])
    cluster_quality = np.mean([c['avg_score'] for c in validated_clusters])
    cluster_size_factor = min(1.0, len(validated_clusters) / 2)
    
    # More sophisticated formula combining cluster quality, consistency and coverage
    forgery_probability = cluster_quality * cluster_size_factor * avg_consistency
    
    # Fixed threshold for forgery detection based on quality of clusters
    final_decision_threshold = 0.3
    forgery_detected = forgery_probability > final_decision_threshold
    
    print(f"  (Debug: Valid Clusters={len(validated_clusters)}, Consistency={avg_consistency:.4f}, " 
          f"Quality={cluster_quality:.4f}, Probability={forgery_probability:.4f}, Threshold={final_decision_threshold})")
    
    # Free up memory
    optimize_memory()
    
    return {
        "forgery_probability": forgery_probability,
        "forgery_detected": forgery_detected,
        "suspicious_pairs": filtered_pairs,
        "heatmap_path": heatmap_overlay_path
    }

def detect_forgery_dense(image_path, model_path="outputs/checkpoints/best_model.pt",
                  patch_size=64, stride=32, threshold=0.85,
                  output_dir="outputs/predictions",
                  max_dim=None):
    """
    Detect copy-move forgery in a single image using dense feature matching
    
    Args:
        image_path: Path to the image
        model_path: Path to the model
        patch_size: Size of patches for comparison
        stride: Stride between patches
        threshold: Threshold for suspicious patch similarity
        output_dir: Directory to save output
        max_dim: Maximum dimension for resizing (None for no resizing)
        
    Returns:
        result: Dict with forgery probability and heatmap
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load models
    vit, siamese, device = load_models(model_path)
    
    # Load image
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not load image {image_path}")
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [],
        "heatmap_path": None}
    
    h_orig, w_orig = img_orig.shape[:2]
    
    # Optional resize (for large images)
    if max_dim and (h_orig > max_dim or w_orig > max_dim):
        scale = max_dim / max(h_orig, w_orig)
        w_resized = int(w_orig * scale)
        h_resized = int(h_orig * scale)
        img_resized = cv2.resize(img_orig, (w_resized, h_resized), interpolation=cv2.INTER_AREA)
        print(f"Resized image from {w_orig}x{h_orig} to {w_resized}x{h_resized} for processing")
        
        # Save scale factor for visualization
        scale_factor = (w_orig / w_resized, h_orig / h_resized)
    else:
        img_resized = img_orig.copy()
        scale_factor = (1, 1)
    
    h, w = img_resized.shape[:2]
    
    # Extract patches more efficiently
    patches = []
    patch_coords = []
    
    # Use RGB patches for feature extraction
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    
    for y in range(0, h - patch_size + 1, stride):
        for x in range(0, w - patch_size + 1, stride):
            patch = img_rgb[y:y+patch_size, x:x+patch_size]
            patches.append(convert_to_tensor(patch))
            patch_coords.append((x, y))
    
    if not patches:
        print(f"Error: No patches could be extracted. Image dimensions too small.")
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [],
        "heatmap_path": None}
    
    print(f"Extracting features from {len(patches)} patches...")
    
    # Extract features in batches
    batch_size = 32  # Adjust based on GPU memory
    num_batches = (len(patches) + batch_size - 1) // batch_size
    patch_features = []
    
    with torch.no_grad():
        for i in tqdm(range(num_batches), desc="Extracting Features"):
            start_idx = i * batch_size
            end_idx = min(start_idx + batch_size, len(patches))
            
            batch = torch.stack(patches[start_idx:end_idx]).to(device)
            features = vit(batch)
            patch_features.append(features.cpu())
            
            # Free up memory
            del batch
            if torch.backends.mps.is_available():
                torch.mps.empty_cache()
            elif torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    # Concatenate all features
    patch_features = torch.cat(patch_features, dim=0)
    
    print(f"Computing patch similarities...")
    
    # Compute pairwise similarities in batches
    suspicious_pairs = []
    similarity_scores = []
    n_patches = len(patch_features)
    
    # Move features to device for computation
    patch_features = patch_features.to(device)
    
    compare_batch_size = 256  # Size of comparison batches
    
    with torch.no_grad():
        for i in tqdm(range(0, n_patches, compare_batch_size), desc="Comparing Patches"):
            end_i = min(i + compare_batch_size, n_patches)
            features_i = patch_features[i:end_i]
            
            # Pre-compute minimum allowed distance
            min_spatial_dist = patch_size * 1.5  # Minimum distance between patches
            
            for j in range(0, n_patches, compare_batch_size):
                end_j = min(j + compare_batch_size, n_patches)
                
                # Skip if i == j (same batch) and j < i (already compared)
                if end_j <= i:
                    continue
                
                features_j = patch_features[j:end_j]
                
                # Compute batch similarity
                batch_i = features_i.repeat_interleave(features_j.size(0), dim=0)
                batch_j = features_j.repeat(features_i.size(0), 1)
                
                # Get similarities
                similarities = siamese(batch_i, batch_j).squeeze()
                
                # Process results
                for idx, sim in enumerate(similarities):
                    # Calculate original indices
                    idx_i = i + (idx // features_j.size(0))
                    idx_j = j + (idx % features_j.size(0))
                    
                    # Skip self-comparisons and patches that are too close
                    if idx_i >= idx_j:
                        continue
                    
                    # Get spatial coordinates
                    x_i, y_i = patch_coords[idx_i]
                    x_j, y_j = patch_coords[idx_j]
                    
                    # Skip patches that are too close spatially
                    spatial_dist = np.sqrt((x_i - x_j)**2 + (y_i - y_j)**2)
                    if spatial_dist < min_spatial_dist:
                        continue
                    
                    # Add to suspicious pairs if similarity is high enough
                    sim_value = sim.item()
                    if sim_value > threshold:
                        suspicious_pairs.append(((x_i, y_i), (x_j, y_j)))
                        similarity_scores.append(sim_value)
    
    # Free up memory
    del patch_features
    if torch.backends.mps.is_available():
        torch.mps.empty_cache()
    elif torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Apply RANSAC to filter out geometric inconsistency
    if len(suspicious_pairs) > 10:
        print(f"Filtering {len(suspicious_pairs)} suspicious pairs...")
        # Convert to numpy for RANSAC
        src_points = np.array([pair[0] for pair in suspicious_pairs])
        dst_points = np.array([pair[1] for pair in suspicious_pairs])
        
        # Use RANSAC to find consistent transform
        try:
            # Find homography matrix (easier than expected transform)
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 5.0)
            
            # Update suspicious pairs and scores
            filtered_pairs = []
            filtered_scores = []
            
            for i, (pair, score) in enumerate(zip(suspicious_pairs, similarity_scores)):
                if mask[i][0] == 1:  # inlier
                    filtered_pairs.append(pair)
                    filtered_scores.append(score)
            
            suspicious_pairs = filtered_pairs
            similarity_scores = filtered_scores
            
            print(f"RANSAC filtered down to {len(suspicious_pairs)} consistent pairs")
        except Exception as e:
            print(f"RANSAC filtering error: {e}, using all pairs")
    
    # Create heatmap
    print(f"Generating heatmap from {len(suspicious_pairs)} suspicious pairs...")
    
    # Generate heatmap indicating suspicious regions
    heatmap = np.zeros(img_resized.shape[:2], dtype=np.float32)
    
    for (x1, y1), (x2, y2) in suspicious_pairs:
        # Add heat to both patches in the pair
        heatmap[y1:y1+patch_size, x1:x1+patch_size] += 1
        heatmap[y2:y2+patch_size, x2:x2+patch_size] += 1
    
    # Normalize heatmap
    if np.max(heatmap) > 0:
        heatmap = heatmap / np.max(heatmap)
    
    # Apply colormap and overlay
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(img_resized, 0.6, heatmap_colored, 0.4, 0)
    
    # Calculate forgery probability based on spatial coherence of suspicious pairs
    if len(suspicious_pairs) > 0:
        # Calculate potential non-adjacent pairs
        total_possible_pairs = (n_patches * (n_patches - 1)) / 2
        
        # Calculate raw probability
        raw_probability = len(suspicious_pairs) / total_possible_pairs
        
        # Apply additional weighting based on spatial coherence
        coherence_score = compute_spatial_coherence(suspicious_pairs, patch_size)
        forgery_probability = min(1.0, raw_probability * 10 * coherence_score)
    else:
        forgery_probability = 0.0
        coherence_score = 0.0
    
    # Decision threshold (can be adjusted based on testing)
    decision_threshold = 0.3
    forgery_detected = forgery_probability > decision_threshold
    
    # Print debug information
    print(f"  (Debug: Raw Pairs Ratio={len(suspicious_pairs)/max(1, (n_patches*(n_patches-1)/2)):.6f}, "
          f"Coherence={coherence_score:.4f}, Final Probability={forgery_probability:.4f})")
    
    # Save results
    image_name = os.path.basename(image_path).split('.')[0]
    heatmap_overlay_path = os.path.join(output_dir, f"{image_name}_heatmap.jpg")
    visualization_path = os.path.join(output_dir, f"{image_name}_visualization.jpg")
    cv2.imwrite(heatmap_overlay_path, overlay)
    save_heatmap_visualization(img_orig, cv2.resize(heatmap, (img_orig.shape[1], img_orig.shape[0])), visualization_path)
    
    return {
        "forgery_probability": forgery_probability,
        "forgery_detected": forgery_detected,
        "suspicious_pairs": suspicious_pairs,
        "heatmap_path": heatmap_overlay_path
    }

def detect_forgery(image_path, model_path="outputs/checkpoints/best_model.pt",
                  patch_size=64, stride=32, threshold=0.85,
                  output_dir="outputs/predictions",
                  max_dim=512):
    """
    Detect copy-move forgery in a single image - chooses best algorithm based on image size
    
    Args:
        image_path: Path to the image
        model_path: Path to the model
        patch_size: Size of patches for comparison
        stride: Stride between patches
        threshold: Threshold for suspicious patch similarity
        output_dir: Directory to save output
        max_dim: Maximum dimension for image resizing (for window approach)
        
    Returns:
        result: Dict with forgery probability and heatmap
    """
    # Check image size to pick best approach
    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Could not load image {image_path}")
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [], "heatmap_path": None}
    
    h, w = img.shape[:2]
    pixel_count = h * w
    
    # For very large images (>2MP), use sliding window approach
    if pixel_count > 2000000:
        print(f"Using sliding window approach for large image ({w}x{h})")
        return detect_forgery_window(image_path, model_path, patch_size, stride, threshold, output_dir, max_dim)
    else:
        # For smaller images, use dense feature matching
        print(f"Using dense feature matching for small image ({w}x{h})")
        return detect_forgery_dense(image_path, model_path, patch_size, stride, threshold, output_dir, 
                                  None if max_dim == 0 else max_dim)

def cluster_suspicious_pairs(suspicious_pairs, similarity_scores, patch_size):
    """
    Group suspicious pairs into spatially coherent clusters
    
    Args:
        suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinates for suspicious pairs
        similarity_scores: List of similarity scores for each pair
        patch_size: Size of patches
        
    Returns:
        clusters: List of dictionaries containing clustered pairs and their statistics
    """
    if not suspicious_pairs:
        return []
        
    # Initialize clusters
    clusters = []
    visited = set()
    
    # Distance threshold for cluster membership - patches should be close to existing cluster
    distance_threshold = patch_size * 2
    
    # Process each suspicious pair
    for i, ((x1, y1), (x2, y2)) in enumerate(suspicious_pairs):
        if i in visited:
            continue
            
        # Start a new cluster
        current_cluster = {
            'pairs': [((x1, y1), (x2, y2))],
            'scores': [similarity_scores[i]],
            'source_coords': [(x1, y1)],
            'target_coords': [(x2, y2)]
        }
        visited.add(i)
        
        # Find other pairs that belong to this cluster
        cluster_changed = True
        while cluster_changed:
            cluster_changed = False
            for j, ((xj1, yj1), (xj2, yj2)) in enumerate(suspicious_pairs):
                if j in visited:
                    continue
                    
                # Check if this pair connects to any point in the current cluster
                for sx, sy in current_cluster['source_coords']:
                    dist1 = np.sqrt((sx - xj1)**2 + (sy - yj1)**2)
                    dist2 = np.sqrt((sx - xj2)**2 + (sy - yj2)**2)
                    if min(dist1, dist2) < distance_threshold:
                        current_cluster['pairs'].append(((xj1, yj1), (xj2, yj2)))
                        current_cluster['scores'].append(similarity_scores[j])
                        current_cluster['source_coords'].append((xj1, yj1))
                        current_cluster['target_coords'].append((xj2, yj2))
                        visited.add(j)
                        cluster_changed = True
                        break
                        
                if cluster_changed:
                    break
                    
                # Also check target coords
                for tx, ty in current_cluster['target_coords']:
                    dist1 = np.sqrt((tx - xj1)**2 + (ty - yj1)**2)
                    dist2 = np.sqrt((tx - xj2)**2 + (ty - yj2)**2)
                    if min(dist1, dist2) < distance_threshold:
                        current_cluster['pairs'].append(((xj1, yj1), (xj2, yj2)))
                        current_cluster['scores'].append(similarity_scores[j])
                        current_cluster['source_coords'].append((xj1, yj1))
                        current_cluster['target_coords'].append((xj2, yj2))
                        visited.add(j)
                        cluster_changed = True
                        break
        
        # Compute cluster statistics
        current_cluster['size'] = len(current_cluster['pairs'])
        current_cluster['avg_score'] = np.mean(current_cluster['scores'])
        current_cluster['max_score'] = np.max(current_cluster['scores'])
        
        # Add cluster to results
        clusters.append(current_cluster)
    
    # Sort clusters by size (largest first)
    clusters.sort(key=lambda c: c['size'], reverse=True)
    
    return clusters

def verify_geometric_consistency(cluster, patch_size):
    """
    Verify if a cluster shows geometric consistency, indicating a true copy-move operation
    
    Args:
        cluster: Dictionary containing cluster information
        patch_size: Size of patches
        
    Returns:
        is_valid: Boolean indicating if cluster is geometrically consistent
        consistency_score: Score representing degree of consistency (0-1)
    """
    # Need at least 3 pairs to check consistency
    if len(cluster['pairs']) < 3:
        return False, 0.0
    
    # Extract source and target points
    sources = np.array(cluster['source_coords'])
    targets = np.array(cluster['target_coords'])
    
    # Check if they form a coherent group by analyzing offset vectors
    offsets = targets - sources
    
    # Calculate mean and standard deviation of offsets
    mean_offset = np.mean(offsets, axis=0)
    std_offset = np.std(offsets, axis=0)
    
    # If standard deviation is very small relative to patch size,
    # this indicates a consistent translation (typical copy-move)
    offset_consistency = np.mean(std_offset) / patch_size
    
    # A true copy-move typically has very consistent offsets
    # Further relaxed threshold to detect subtle forgeries
    if offset_consistency < 0.5:  
        # Calculate consistency score (1 = perfect consistency)
        consistency_score = 1.0 - min(1.0, offset_consistency * 2)
        
        # More relaxed area coverage factor - even smaller regions could be forgeries
        area_coverage = min(1.0, len(cluster['pairs']) / 8)
        
        # Much more relaxed validation threshold
        is_valid = consistency_score * area_coverage > 0.2
        return is_valid, consistency_score
    else:
        # Also check if it might be a rotation or scaling type forgery
        src_dists = []
        tgt_dists = []
        
        # Calculate point-to-point distances in source and target
        for i in range(min(10, len(sources))):
            for j in range(i+1, min(10, len(sources))):
                src_dist = np.sqrt(np.sum((sources[i] - sources[j]) ** 2))
                tgt_dist = np.sqrt(np.sum((targets[i] - targets[j]) ** 2))
                src_dists.append(src_dist)
                tgt_dists.append(tgt_dist)
        
        # Convert to numpy arrays
        src_dists = np.array(src_dists)
        tgt_dists = np.array(tgt_dists)
        
        # If distances are preserved (ratio is consistent), it might be a rigid transformation
        if len(src_dists) > 0:
            distance_ratios = tgt_dists / (src_dists + 1e-6)
            ratio_std = np.std(distance_ratios)
            ratio_consistency = ratio_std / np.mean(distance_ratios)
            
            # Further relaxed threshold
            if ratio_consistency < 0.4:  
                consistency_score = 1.0 - min(1.0, ratio_consistency * 2.5)
                area_coverage = min(1.0, len(cluster['pairs']) / 8)
                # More relaxed validation threshold
                is_valid = consistency_score * area_coverage > 0.15
                return is_valid, consistency_score
    
    # If we get here, cluster did not pass geometric validation
    return False, 0.0

def compute_spatial_coherence(pairs, patch_size):
    """
    Compute spatial coherence score for suspicious pairs
    Higher coherence = more likely to be an actual forgery
    """
    if len(pairs) < 3:
        return 0.5  # Not enough pairs to determine coherence
    
    # Calculate offset vectors between pairs
    offsets = []
    for (x1, y1), (x2, y2) in pairs:
        offsets.append((x2 - x1, y2 - y1))
    
    offsets = np.array(offsets)
    
    # Calculate mean and standard deviation of offsets
    mean_offset = np.mean(offsets, axis=0)
    std_offset = np.std(offsets, axis=0)
    
    # Calculate normalized standard deviation (lower = more coherent)
    norm_std = np.mean(std_offset) / patch_size
    
    # Convert to coherence score (1 = perfect coherence, 0 = no coherence)
    coherence = 1.0 - min(1.0, norm_std)
    
    return coherence

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Copy-Move Forgery Detection Inference")
    # --- Add arguments for predict mode ---
    parser.add_argument('--mode', type=str, default='detect', choices=['detect', 'predict'], 
                        help="Mode: 'detect' for single image forgery detection, 'predict' for comparing two images.")
    parser.add_argument('--image1', type=str, help="Path to the first image (for predict mode)")
    parser.add_argument('--image2', type=str, help="Path to the second image (for predict mode)")
    # --- Keep existing arguments for detect mode ---
    parser.add_argument('--image', type=str, help="Path to the image (for detect mode)")
    parser.add_argument('--model', type=str, default="outputs/checkpoints/best_model.pt", help="Path to the model")
    parser.add_argument('--output', type=str, default="outputs/predictions", help="Output directory (for detect mode)")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch size (for detect mode)")
    parser.add_argument('--stride', type=int, default=32, help="Stride between patches (for detect mode)")
    parser.add_argument('--threshold', type=float, default=0.85, help="Threshold for similarity (for detect mode)") 
    parser.add_argument('--max_dim', type=int, default=512, help="Maximum dimension for image resizing (for detect mode, use 0 or None to disable)")
    parser.add_argument('--use_dense', action='store_true', help="Use dense feature matching instead of sliding window")
    
    args = parser.parse_args()
    
    # --- Add logic to switch between modes ---
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
        if similarity_score > 0.5: # Using 0.5 as the threshold based on training/eval
             print("  Result: Likely a FORGED pair (score > 0.5)")
        else:
             print("  Result: Likely an ORIGINAL pair (score <= 0.5)")

    elif args.mode == 'detect':
        if not args.image:
            parser.error("--image is required for detect mode.")
            
        # Handle max_dim=0 or None
        max_dim_value = args.max_dim if args.max_dim and args.max_dim > 0 else None

        result = detect_forgery(
            args.image, 
            args.model, 
            args.patch_size, 
            args.stride, 
            args.threshold, 
            args.output,
            max_dim=max_dim_value # <-- Pass max_dim
        )
        
        print(f"\nDetecting forgery within '{os.path.basename(args.image)}':")
        print(f"  Forgery probability: {result['forgery_probability']:.4f}")
        print(f"  Forgery detected: {result['forgery_detected']}")
        print(f"  Number of suspicious pairs: {len(result.get('suspicious_pairs',[]))}")
        if 'heatmap_path' in result and result['heatmap_path']:
            print(f"  Heatmap saved to: {result['heatmap_path']}")
