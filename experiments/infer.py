import sys
from pathlib import Path

# Add project root to Python path (MUST be before other project imports)
sys.path.append(str(Path(__file__).parent.parent))

import os
import cv2
import torch
import numpy as np
from tqdm import tqdm
from models.vit_encoder import ViTEncoder
from models.siamese import SiameseNetwork
from utils.transforms import convert_to_tensor
from utils.heatmap import generate_heatmap, save_heatmap_visualization
from utils.mac_utils import get_device, optimize_memory
from utils.frequency_analysis import FrequencyAnalyzer
from utils.confidence_scoring import calculate_forgery_probability
from utils.region_analysis import cluster_suspicious_pairs, verify_geometric_consistency, compute_spatial_coherence

def load_models(model_path):
    """Load saved models"""
    device = get_device()
    
    # Initialize models
    vit = ViTEncoder(pretrained=False)
    siamese = SiameseNetwork()
    
    # Load saved model
    if os.path.exists(model_path):
        try:
            # First load to CPU to avoid data type issues
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Ensure all tensors have consistent types (float32)
            for key in checkpoint['vit_state_dict']:
                if isinstance(checkpoint['vit_state_dict'][key], torch.Tensor):
                    checkpoint['vit_state_dict'][key] = checkpoint['vit_state_dict'][key].float()
            
            for key in checkpoint['siamese_state_dict']:
                if isinstance(checkpoint['siamese_state_dict'][key], torch.Tensor):
                    checkpoint['siamese_state_dict'][key] = checkpoint['siamese_state_dict'][key].float()
            
            # Load state dictionaries
            vit.load_state_dict(checkpoint['vit_state_dict'])
            siamese.load_state_dict(checkpoint['siamese_state_dict'])
            
            print(f"Model loaded successfully from {model_path}")
        except Exception as e:
            print(f"Error loading model: {e}. Using untrained model.")
    else:
        print(f"Warning: Model not found at {model_path}, using untrained model")
    
    # Set models to evaluation mode
    vit.eval()
    siamese.eval()
    
    # Move models to device AFTER loading weights
    vit.to(device)
    siamese.to(device)
    
    return vit, siamese, device

def process_window(window, vit, siamese, device, patch_size, stride, threshold, offset=(0, 0)):
    """Process a single window and return results with original coordinates"""
    x_offset, y_offset = offset
    window_h, window_w = window.shape[:2]
    
    # Patch extraction with original coordinates
    patch_coords = []
    patches = []
    
    # Convert window to RGB once for efficiency
    window_rgb = cv2.cvtColor(window, cv2.COLOR_BGR2RGB)
    
    # Extract all patches at once
    for y in range(0, window_h - patch_size + 1, stride):
        for x in range(0, window_w - patch_size + 1, stride):
            # Extract patch from window (using local coordinates)
            patch = window_rgb[y:y+patch_size, x:x+patch_size]
            patches.append(convert_to_tensor(patch))
            # Store global coordinates for the patch
            patch_coords.append((x + x_offset, y + y_offset))
    
    if not patches:
        return {'suspicious_pairs': [], 'similarity_scores': [], 'patch_coords': patch_coords,
                'pair_frequency_metrics': []}
    
    # Extract features in batches
    feature_batch_size = 32  # Increase for better efficiency, decrease if OOM errors occur
    patch_features = []
    
    with torch.no_grad():
        for i in range(0, len(patches), feature_batch_size):
            # Move batch to device, process, then back to CPU to save memory
            batch = torch.stack(patches[i:i+feature_batch_size]).to(device)
            features = vit(batch)
            patch_features.append(features.cpu())  # Move back to CPU after processing
            # Clean up batch to free memory
            del batch
            torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Clean up early to free memory
    del patches
    
    # Concatenate features on CPU first, then move to device once for all comparisons
    patch_features = torch.cat(patch_features, dim=0)
    
    # Compare all pairs of patches with vectorized operations where possible
    suspicious_pairs = []
    similarity_scores = []
    n_patches = len(patch_features)
    
    # Use smaller comparison batch size if memory is limited
    comparison_batch_size = 64 if n_patches > 500 else 128
    
    with torch.no_grad():
        for i in range(n_patches):
            # Get current patch feature and coordinates
            feat_i = patch_features[i].to(device)  # Move single feature to device
            # Get local coordinates for distance calculation
            xi, yi = patch_coords[i][0] - x_offset, patch_coords[i][1] - y_offset
            
            # Process in batches to avoid memory issues
            for j_start in range(i + 1, n_patches, comparison_batch_size):
                j_end = min(j_start + comparison_batch_size, n_patches)
                if j_end <= j_start:
                    continue
                
                # Filter patches by spatial distance first (avoid unnecessary comparisons)
                valid_indices = []
                for j in range(j_start, j_end):
                    # Get local coordinates for distance calculation
                    xj, yj = patch_coords[j][0] - x_offset, patch_coords[j][1] - y_offset
                    # Skip adjacent patches
                    if np.sqrt((xi - xj)**2 + (yi - yj)**2) < patch_size * 1.5:
                        continue
                    valid_indices.append(j)
                
                if not valid_indices:
                    continue
                
                # Compute similarities for valid patches
                feat_j_batch = patch_features[valid_indices].to(device)
                batch_size = feat_j_batch.size(0)
                feat_i_expanded = feat_i.unsqueeze(0).expand(batch_size, -1)
                
                similarities = siamese(feat_i_expanded, feat_j_batch).squeeze().cpu().numpy()
                
                # Handle case where only one comparison is made
                if batch_size == 1:
                    similarities = np.array([similarities])
                
                # Store suspicious pairs
                for k, j in enumerate(valid_indices):
                    if similarities[k] > threshold:
                        suspicious_pairs.append((patch_coords[i], patch_coords[j]))
                        similarity_scores.append(float(similarities[k]))
            
            # Clean up the current feature to free memory
            del feat_i
            torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Clean up GPU memory
    del patch_features
    torch.cuda.empty_cache() if device.type == 'cuda' else None
    
    # Use frequency domain analysis only if we have suspicious pairs
    pair_frequency_metrics = [] 
    if suspicious_pairs:
        try:
            # Initialize frequency analyzer
            frequency_analyzer = FrequencyAnalyzer()
            
            # Prepare pairs with local coordinates for analysis
            local_pairs_for_analysis = [
                ((x1-x_offset, y1-y_offset), (x2-x_offset, y2-y_offset)) 
                for (x1, y1), (x2, y2) in suspicious_pairs
            ]
            
            # Run frequency domain analysis
            freq_region_results = frequency_analyzer.analyze_regions(
                window, local_pairs_for_analysis, patch_size)
            
            # Analyze frequency bands for each pair individually
            freq_bands_results = frequency_analyzer.analyze_frequency_bands(
                window, local_pairs_for_analysis, patch_size)
            
            # Get band similarities
            band_similarities_per_pair = freq_bands_results.get('band_similarities', [])
            
            # Create a metric for each suspicious pair
            for i in range(len(suspicious_pairs)):
                avg_band_similarity = 0.0
                if i < len(band_similarities_per_pair) and band_similarities_per_pair[i]:
                    if isinstance(band_similarities_per_pair[i], (list, np.ndarray)):
                        avg_band_similarity = np.mean(band_similarities_per_pair[i])
                    else:
                        avg_band_similarity = band_similarities_per_pair[i]
                
                pair_metrics = {
                    'avg_band_similarity': avg_band_similarity,
                    'block_strength': freq_region_results.get('block_strength', 0),
                    'block_periodicity': freq_region_results.get('block_periodicity', 0)
                }
                pair_frequency_metrics.append(pair_metrics)
        except Exception as e:
            print(f"Warning: Frequency analysis failed: {e}")
            # Fill with default values
            pair_frequency_metrics = [{'avg_band_similarity': 0.0, 'block_strength': 0, 'block_periodicity': 0}] * len(suspicious_pairs)
    
    # Ensure the returned list has the same length as suspicious_pairs
    if len(pair_frequency_metrics) != len(suspicious_pairs):
        pair_frequency_metrics = [{'avg_band_similarity': 0.0, 'block_strength': 0, 'block_periodicity': 0}] * len(suspicious_pairs)

    return {
        'suspicious_pairs': suspicious_pairs, 
        'similarity_scores': similarity_scores, 
        'patch_coords': patch_coords,
        'pair_frequency_metrics': pair_frequency_metrics
    }

def predict(image_path1, image_path2, model_path="outputs/checkpoints/most_accuracy_model.pt"):
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

def detect_forgery(image_path, model_path="outputs/checkpoints/most_accuracy_model.pt",
                  patch_size=64, stride=32, threshold=0.55,
                  output_dir="outputs/predictions",
                  max_dim=1024):
    """
    Detect copy-move forgery in a single image - chooses best algorithm based on image size
    
    Args:
        image_path: Path to the image
        model_path: Path to the model
        patch_size: Size of patches for comparison
        stride: Stride between patches
        threshold: Threshold for suspicious patch similarity
        output_dir: Directory to save output
        max_dim: Maximum dimension for image resizing
        
    Returns:
        result: Dict with forgery probability and heatmap
    """
    # Check if we're in an interactive environment that expects progress updates
    show_progress = os.environ.get("CMFD_SHOW_PROGRESS", "0") == "1"
    if show_progress:
        print("Progress reporting enabled")
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Check image exists and can be loaded
    img_orig = cv2.imread(image_path)
    if img_orig is None:
        print(f"Error: Could not load image {image_path}")
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [], "heatmap_path": None}
    
    h, w = img_orig.shape[:2]
    pixel_count = h * w
    print(f"Processing image of size {w}x{h} ({pixel_count} pixels)")
    
    # Load models
    try:
        vit, siamese, device = load_models(model_path)
        print(f"Using device: {device}")
    except Exception as e:
        print(f"Error loading models: {e}")
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [], "heatmap_path": None}
    
    # Initialize result accumulators
    all_suspicious_pairs = []
    all_similarity_scores = []
    all_patch_coords = []
    all_pair_frequency_metrics = []
    scale_factor = (1, 1)  # Default value
    
    # Set smaller stride for smaller images to get more detail
    adaptive_stride = max(16, stride if pixel_count > 1000000 else stride // 2)
    print(f"Using stride: {adaptive_stride}")
    
    # For very large images (>1MP) or limited memory systems, use sliding window
    if pixel_count > 1000000 or os.environ.get("CMFD_FORCE_WINDOWS", "0") == "1":
        print(f"Using sliding window approach for large image")
        # Use smaller window size on lower-spec machines
        window_size = (512, 384)
        
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
                    window, vit, siamese, device, patch_size, adaptive_stride, threshold,
                    (x_start, y_start)  # Pass offset coordinates
                )
                
                # Accumulate results
                all_suspicious_pairs.extend(window_results['suspicious_pairs'])
                all_similarity_scores.extend(window_results['similarity_scores'])
                all_patch_coords.extend(window_results['patch_coords'])
                all_pair_frequency_metrics.extend(window_results['pair_frequency_metrics'])
                
                # Free memory after processing each window
                if device.type == 'cuda':
                    torch.cuda.empty_cache()
                
                # Show progress if enabled
                if show_progress and total_windows > 1:
                    progress_pct = int(30 + (window_count / total_windows) * 40)
                    print(f"Progress: {progress_pct}%")
    else:
        # For smaller images, use direct processing with optional resize
        print(f"Using direct processing for smaller image")
        img_resized = img_orig
        scale_factor = (1, 1)
        
        # Optional resize for large images but below 1MP
        if max_dim and (h > max_dim or w > max_dim):
            scale = max_dim / max(h, w)
            w_resized = int(w * scale)
            h_resized = int(h * scale)
            img_resized = cv2.resize(img_orig, (w_resized, h_resized), interpolation=cv2.INTER_AREA)
            print(f"Resized image from {w}x{h} to {w_resized}x{h_resized} for processing")
            scale_factor = (w / w_resized, h / h_resized)
        
        # Process using process_window function but with the whole image
        window_results = process_window(
            img_resized, vit, siamese, device, patch_size, adaptive_stride, threshold, (0, 0)
        )
        
        all_suspicious_pairs = window_results['suspicious_pairs']
        all_similarity_scores = window_results['similarity_scores']
        all_patch_coords = window_results['patch_coords']
        all_pair_frequency_metrics = window_results['pair_frequency_metrics']
        
        # Update progress
        if show_progress:
            print("Progress: 60%")
    
    # If no suspicious pairs found, return early
    if not all_suspicious_pairs:
        print("No suspicious pairs found")
        if show_progress:
            print("Progress: 100%")
        return {"forgery_probability": 0.0, "forgery_detected": False, "suspicious_pairs": [], "heatmap_path": None}
    
    print(f"Found {len(all_suspicious_pairs)} suspicious pairs")
    
    # Filter out geometric inconsistency using RANSAC 
    filtered_pairs = all_suspicious_pairs
    filtered_scores = all_similarity_scores
    filtered_freq_metrics = all_pair_frequency_metrics
    
    # Update progress for RANSAC filtering
    if show_progress:
        print("Progress: 70%")
    
    if len(all_suspicious_pairs) > 4:
        print(f"Filtering {len(all_suspicious_pairs)} suspicious pairs using RANSAC...")
        try:
            # Convert to numpy for RANSAC
            src_points = np.array([pair[0] for pair in all_suspicious_pairs])
            dst_points = np.array([pair[1] for pair in all_suspicious_pairs])
            
            # Use a lower threshold for RANSAC to be more permissive
            H, mask = cv2.findHomography(src_points, dst_points, cv2.RANSAC, 10.0)
            
            if mask is not None:
                # Update suspicious pairs and scores
                filtered_pairs = []
                filtered_scores = []
                filtered_freq_metrics = []
                
                for i, (pair, score) in enumerate(zip(all_suspicious_pairs, all_similarity_scores)):
                    if mask[i][0] == 1:  # inlier
                        filtered_pairs.append(pair)
                        filtered_scores.append(score)
                        if i < len(all_pair_frequency_metrics):
                            filtered_freq_metrics.append(all_pair_frequency_metrics[i])
                
                print(f"RANSAC filtered down to {len(filtered_pairs)} consistent pairs")
            else:
                print("RANSAC did not yield valid mask, using all pairs")
                
        except Exception as e:
            print(f"RANSAC filtering error: {e}, using all pairs")
            import traceback
            traceback.print_exc()
    
    # Update progress for cluster analysis
    if show_progress:
        print("Progress: 80%")
    
    # Apply spatial clustering to group suspicious pairs
    try:
        clusters = cluster_suspicious_pairs(filtered_pairs, filtered_scores, patch_size)
        
        # Filter out small clusters (likely false positives)
        min_cluster_size = 2  # Minimum pairs needed in a cluster
        significant_clusters = [c for c in clusters if len(c['pairs']) >= min_cluster_size]
        
        # Apply geometric consistency check to validate each cluster
        validated_clusters = []
        for cluster in significant_clusters:
            # Check if cluster has geometric consistency (indicating true copy-move)
            is_valid, consistency_score = verify_geometric_consistency(cluster, patch_size)
            
            # Only add clusters that pass geometric validation
            if is_valid:
                cluster['consistency_score'] = consistency_score
                validated_clusters.append(cluster)
    except Exception as e:
        print(f"Error in cluster analysis: {e}")
        import traceback
        traceback.print_exc()
        validated_clusters = []
    
    # Update progress for heatmap generation
    if show_progress:
        print("Progress: 90%")
    
    # If no geometrically valid clusters remain, likely no forgery
    if not validated_clusters and len(filtered_pairs) > 0:
        print("No geometrically consistent forgery clusters found, but suspicious pairs exist")
        # Just use filtered pairs directly
        forgery_probability = min(0.3, len(filtered_pairs) / 100)  # Low confidence
        forgery_detected = forgery_probability > 0.2
    elif not validated_clusters:
        print("No suspicious pairs or valid clusters found")
        if show_progress:
            print("Progress: 100%")
        return {
            "forgery_probability": 0.0,
            "forgery_detected": False,
            "suspicious_pairs": filtered_pairs,
            "heatmap_path": None
        }
    else:
        # Use validated clusters to determine forgery probability
        avg_consistency = np.mean([c.get('consistency_score', 0) for c in validated_clusters])
        cluster_quality = np.mean([c['avg_score'] for c in validated_clusters])
        cluster_size_factor = min(1.0, len(validated_clusters) / 2)
        
        # More sophisticated formula combining cluster quality, consistency and coverage
        forgery_probability = cluster_quality * cluster_size_factor * avg_consistency
        
        # Threshold for forgery detection based on quality of clusters
        final_decision_threshold = 0.3
        forgery_detected = forgery_probability > final_decision_threshold
        
        print(f"  (Debug: Valid Clusters={len(validated_clusters)}, Consistency={avg_consistency:.4f}, " 
              f"Quality={cluster_quality:.4f}, Probability={forgery_probability:.4f})")
    
    # Generate heatmap from filtered pairs
    try:
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
    except Exception as e:
        print(f"Error generating heatmap: {e}")
        import traceback
        traceback.print_exc()
        heatmap_overlay_path = None
    
    # Free up memory
    if device.type == 'cuda':
        torch.cuda.empty_cache()
    
    # Complete progress
    if show_progress:
        print("Progress: 100%")
    
    return {
        "forgery_probability": forgery_probability,
        "forgery_detected": forgery_detected,
        "suspicious_pairs": filtered_pairs,
        "heatmap_path": heatmap_overlay_path
    }

def calculate_offset_distribution(suspicious_pairs):
    """
    Calculate statistics about the distribution of offset vectors in suspicious pairs
    
    Args:
        suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
        
    Returns:
        dict: Statistical metrics about offset distribution
    """
    if len(suspicious_pairs) < 3:
        return {"entropy": 0, "mode_strength": 0, "spread": 0}
    
    # Calculate offset vectors
    offsets = []
    for (x1, y1), (x2, y2) in suspicious_pairs:
        offsets.append((x2-x1, y2-y1))
    
    # Round offsets to integers for better clustering
    offset_tuples = [(int(round(x)), int(round(y))) for x, y in offsets]
    
    # Count frequency of each offset
    from collections import Counter
    offset_counts = Counter(offset_tuples)
    
    # Calculate mode (most common offset)
    most_common_offset, most_common_count = offset_counts.most_common(1)[0]
    mode_strength = most_common_count / len(offsets)
    
    # Calculate spread (number of unique offsets relative to total)
    spread = len(offset_counts) / len(offsets)
    
    # Calculate entropy (higher means more random distribution)
    probs = [count/len(offsets) for count in offset_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs)
    
    return {
        "entropy": entropy,
        "mode_strength": mode_strength,
        "spread": spread,
        "most_common_offset": most_common_offset
    }

def calculate_forgery_probability(suspicious_pairs, similarity_scores, n_patches, patch_size, frequency_results=None):
    """
    Calculate forgery probability using multiple metrics with special handling for different forgery types,
    optionally incorporating frequency analysis results.
    
    Args:
        suspicious_pairs: List of coordinate pairs that are suspicious
        similarity_scores: List of similarity scores for each pair
        n_patches: Total number of patches in the image
        patch_size: Size of each patch
        frequency_results (dict, optional): Results from frequency analysis. Defaults to None.
        
    Returns:
        probability: Final forgery probability
        metrics: Dictionary of metrics used in calculation
    """
    if not suspicious_pairs or len(suspicious_pairs) == 0:
        # Return default metrics structure even if no pairs
        return 0.0, {
            "raw_ratio": 0, "num_pairs": 0, "coherence": 0, "entropy": 0, 
            "mode_strength": 0, "spread": 0, "structured_score": 0, 
            "complex_score": 0, "base_score": 0, "frequency_score": 0, 
            "probability": 0
        }
    
    # 1. Calculate basic metrics
    total_possible_pairs = (n_patches * (n_patches - 1)) / 2
    # Avoid division by zero if n_patches is 1 or 0
    raw_probability = len(suspicious_pairs) / total_possible_pairs if total_possible_pairs > 0 else 0 
    coherence_score = compute_spatial_coherence(suspicious_pairs, patch_size)
    
    # 2. Calculate offset distribution metrics
    offset_metrics = calculate_offset_distribution(suspicious_pairs)
    
    # Calculate raw pair ratio score (scaled for better weighting)
    # This gives a rough measure of how many suspicious pairs were found relative to possible pairs
    raw_ratio_score = min(0.3, raw_probability * 25.0)  # Cap at 0.3
    
    # Calculate pair count score
    # More pairs = more likely to be a forgery, with diminishing returns
    num_pairs = len(suspicious_pairs)
    pair_count_score = min(0.6, 0.1 * np.log10(1 + num_pairs))  # Log scale with max 0.6
    
    # Calculate average similarity score
    avg_similarity = np.mean(similarity_scores) if similarity_scores else 0
    similarity_score = avg_similarity * 0.2  # Scale to roughly 0.1-0.2 range
    
    # Combine base scores that apply to all forgery types 
    base_score = raw_ratio_score + pair_count_score + similarity_score
    
    # Now consider two possible forgery scenarios:
    
    # Scenario 1: Structured copy-move with coherent offsets (classic copy-move)
    # - Low entropy, high mode strength, low spread
    structured_indicators = [
        offset_metrics["mode_strength"] * 1.2,      # Strong primary direction
        (1 - offset_metrics["spread"]) * 0.8,       # Few different offsets
        max(0, 1 - (offset_metrics["entropy"] / 4)) * 0.7  # Low entropy
    ]
    structured_score = sum(structured_indicators) / (1.2 + 0.8 + 0.7)  # Normalize by sum of weights
    
    # Scenario 2: Multiple small forgeries or complex manipulations
    # - Many suspicious pairs but possibly higher entropy and spread
    # - We rely more on coherence and raw number of suspicious pairs
    complex_indicators = [
        coherence_score * 1.5,                # Spatial coherence
        min(0.5, num_pairs / 300) * 1.0,      # More pairs = more suspicious
        min(0.5, raw_probability * 50) * 0.8  # Higher proportion of suspicious pairs
    ]
    complex_score = sum(complex_indicators) / (1.5 + 1.0 + 0.8)  # Normalize by sum of weights
    
    # Take the maximum score from either forgery scenario 
    # This lets us detect both classic copy-move and more complex manipulations
    scenario_score = max(structured_score, complex_score) * 0.6  # Scale to roughly 0-0.6 range

    # --- Add Frequency Analysis Score ---
    frequency_score = 0.0
    if frequency_results and isinstance(frequency_results, dict):
        # Example: Use average band similarity deviation
        band_similarities = frequency_results.get('band_similarities', [])
        
        # Use block artifact consistency.
        block_strength = frequency_results.get('block_strength', 0)
        block_periodicity = frequency_results.get('block_periodicity', 0)
        
        # Simple score based on block artifacts (needs tuning)
        # Strong, periodic block artifacts might indicate inconsistent compression
        block_artifact_score = min(0.2, (block_strength * block_periodicity) * 5.0) 
        
        # Use band similarity consistency. If pairs are truly copied, their band similarities should be high.
        # Let's calculate the average similarity across all bands for each pair
        avg_pair_band_sims = []
        for pair_bands in band_similarities:
            if isinstance(pair_bands, (list, np.ndarray)) and len(pair_bands) > 0:
                avg_pair_band_sims.append(np.mean(pair_bands))
            elif isinstance(pair_bands, (int, float)): # Handle case where it's already an average
                avg_pair_band_sims.append(pair_bands)

        if avg_pair_band_sims:
             # Average similarity across all pairs' frequency bands
            overall_avg_band_sim = np.mean(avg_pair_band_sims)
            # Score increases as average band similarity increases (more evidence for copy-move)
            band_consistency_score = min(0.3, overall_avg_band_sim * 0.3) 
        else:
            band_consistency_score = 0.0

        frequency_score = block_artifact_score + band_consistency_score
        frequency_score = min(0.4, frequency_score) # Cap frequency contribution

    # Final probability = base score + best scenario score + frequency score
    forgery_probability = base_score + scenario_score + frequency_score # Add frequency_score
    forgery_probability = min(1.0, forgery_probability)  # Cap at 1.0
    
    metrics = {
        "raw_ratio": raw_probability,
        "num_pairs": num_pairs,
        "coherence": coherence_score,
        "entropy": offset_metrics["entropy"],
        "mode_strength": offset_metrics["mode_strength"],
        "spread": offset_metrics["spread"],
        "structured_score": structured_score,
        "complex_score": complex_score,
        "base_score": base_score,
        "frequency_score": frequency_score, # Add frequency score to metrics
        "probability": forgery_probability
    }
    
    return forgery_probability, metrics

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
    parser.add_argument('--model', type=str, default="outputs/checkpoints/most_accuracy_model.pt", help="Path to the trained model checkpoint")
    parser.add_argument('--output', type=str, default="outputs/predictions", help="Output directory (for detect mode)")
    parser.add_argument('--patch_size', type=int, default=64, help="Patch size (for detect mode)")
    parser.add_argument('--stride', type=int, default=32, help="Stride between patches (for detect mode)")
    parser.add_argument('--threshold', type=float, default=0.55, help="Threshold for similarity (for detect mode)") 
    parser.add_argument('--max_dim', type=int, default=1024, help="Maximum dimension for image resizing (for detect mode, use 0 or None to disable)")
    
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
        if (similarity_score > 0.5): # Using 0.5 as the threshold based on training/eval
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
            max_dim=max_dim_value
        )
        
        print(f"\nDetecting forgery within '{os.path.basename(args.image)}':")
        print(f"  Forgery probability: {result['forgery_probability']:.4f}")
        print(f"  Forgery detected: {result['forgery_detected']}")
        print(f"  Number of suspicious pairs: {len(result.get('suspicious_pairs',[]))}")
        if 'heatmap_path' in result and result['heatmap_path']:
            print(f"  Heatmap saved to: {result['heatmap_path']}")
