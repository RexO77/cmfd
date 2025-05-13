"""
Advanced confidence scoring system for copy-move forgery detection.
This module provides tools to calculate confidence scores, uncertainty ranges,
and detailed evidence metrics for copy-move forgery detection.
"""

import numpy as np
import torch
import cv2
from scipy import stats
from collections import defaultdict, Counter

def calculate_forgery_probability(suspicious_pairs, similarity_scores, image_size, threshold=0.7):
    """
    Calculate the probability that an image contains forgery based on suspicious pairs.
    
    Args:
        suspicious_pairs: List of coordinate pairs that are suspicious
        similarity_scores: List of similarity scores for each pair
        image_size: Tuple (width, height) of the image
        threshold: Confidence threshold
        
    Returns:
        probability: Probability score indicating likelihood of forgery (0-1)
        is_forged: Boolean indicating if forgery is detected
    """
    if not suspicious_pairs or len(suspicious_pairs) == 0:
        return 0.0, False
    
    # Use the more comprehensive function to calculate confidence
    n_patches = (image_size[0] // 64) * (image_size[1] // 64)  # Estimate patches using default 64 size
    patch_size = 64  # Default patch size
    
    # Basic image metrics for the confidence calculation
    image_metrics = {'image_size': image_size}
    
    confidence_score, confidence_interval, evidence_metrics = calculate_forgery_confidence(
        suspicious_pairs, similarity_scores, image_metrics, n_patches, patch_size
    )
    
    # Determine if the image is forged based on confidence score and threshold
    is_forged = confidence_score >= threshold
    
    return confidence_score, is_forged

def calculate_forgery_confidence(suspicious_pairs, similarity_scores, image_metrics, n_patches, patch_size):
    """
    Calculate forgery confidence score with uncertainty estimation
    
    Args:
        suspicious_pairs: List of coordinate pairs that are suspicious
        similarity_scores: List of similarity scores for each pair
        image_metrics: Dictionary with image analysis metrics (texture, contrast, etc.)
        n_patches: Total number of patches in the image
        patch_size: Size of each patch
        
    Returns:
        confidence_score: Final confidence score (0-1)
        uncertainty_range: Tuple (lower_bound, upper_bound) for 95% confidence interval
        evidence_metrics: Dictionary with detailed evidence metrics
    """
    if not suspicious_pairs or len(suspicious_pairs) == 0:
        return 0.0, (0.0, 0.0), {
            'suspicious_pairs': 0,
            'coherence': 0.0,
            'pattern_strength': 0.0,
            'regional_concentration': 0.0,
            'evidence_strength': 'none',
            'primary_offset': None,
            'uncertainty': 0.0
        }
    
    # --- Basic metrics calculation ---
    n_pairs = len(suspicious_pairs)
    total_possible_pairs = (n_patches * (n_patches - 1)) / 2
    raw_probability = n_pairs / total_possible_pairs
    
    # Calculate offset vector distribution
    offsets = []
    for (x1, y1), (x2, y2) in suspicious_pairs:
        offsets.append((x2 - x1, y2 - y1))
    
    # Calculate spatial coherence
    coherence_score = compute_advanced_coherence(suspicious_pairs, patch_size)
    
    # --- Analyze offset distribution ---
    offset_metrics = analyze_offset_distribution(offsets)
    
    # --- Regional concentration analysis ---
    regional_concentration = calculate_regional_concentration(suspicious_pairs, image_metrics.get('image_size', (512, 512)))
    
    # --- Monte Carlo sampling for uncertainty estimation ---
    # Perform multiple calculations with slight variations in thresholds
    confidence_samples = monte_carlo_confidence_sampling(
        suspicious_pairs,
        similarity_scores,
        n_patches,
        patch_size,
        num_samples=20
    )
    
    # Calculate mean confidence and 95% confidence interval
    confidence_mean = np.mean(confidence_samples)
    confidence_std = np.std(confidence_samples)
    confidence_interval = stats.norm.interval(0.95, loc=confidence_mean, scale=confidence_std)
    
    # Ensure interval is within [0, 1]
    confidence_interval = (
        max(0.0, confidence_interval[0]), 
        min(1.0, confidence_interval[1])
    )
    uncertainty = confidence_interval[1] - confidence_interval[0]
    
    # --- Calculate pattern strength ---
    pattern_strength = calculate_pattern_strength(
        offset_metrics,
        coherence_score,
        regional_concentration,
        n_pairs / max(1, n_patches)
    )
    
    # --- Determine evidence strength category ---
    evidence_strength = categorize_evidence_strength(pattern_strength, uncertainty)
    
    # --- Calculate final confidence score ---
    # Combine multiple factors with appropriate weighting
    base_score = raw_probability * 10  # Scale up the raw probability
    coherence_weight = 1.5 if coherence_score > 0.5 else 0.8
    pattern_weight = 2.0 if pattern_strength > 0.6 else 1.0
    
    # Weighted combination
    confidence_score = (
        0.2 * base_score +
        0.4 * coherence_score * coherence_weight +
        0.4 * pattern_strength * pattern_weight
    )
    
    # Cap at 1.0
    confidence_score = min(1.0, confidence_score)
    
    # --- Prepare evidence metrics for return ---
    evidence_metrics = {
        'suspicious_pairs': n_pairs,
        'suspicious_ratio': raw_probability,
        'coherence': coherence_score,
        'pattern_strength': pattern_strength,
        'regional_concentration': regional_concentration,
        'evidence_strength': evidence_strength,
        'primary_offset': offset_metrics.get('most_common_offset', None),
        'offset_entropy': offset_metrics.get('entropy', 0),
        'offset_mode_strength': offset_metrics.get('mode_strength', 0),
        'uncertainty': uncertainty
    }
    
    return confidence_score, confidence_interval, evidence_metrics


def monte_carlo_confidence_sampling(suspicious_pairs, similarity_scores, n_patches, patch_size, num_samples=20):
    """
    Perform Monte Carlo sampling to estimate confidence score uncertainty
    
    Args:
        suspicious_pairs: List of coordinate pairs that are suspicious
        similarity_scores: List of similarity scores for each pair
        n_patches: Total number of patches in the image
        patch_size: Size of each patch
        num_samples: Number of Monte Carlo samples
        
    Returns:
        samples: List of confidence scores from different samplings
    """
    samples = []
    
    # Base parameters
    base_coherence_weight = 0.4
    base_pattern_weight = 0.4
    
    for _ in range(num_samples):
        # Randomly sample parameters with small variations
        coherence_weight = base_coherence_weight * np.random.uniform(0.8, 1.2)
        pattern_weight = base_pattern_weight * np.random.uniform(0.8, 1.2)
        
        # Randomly subsample suspicious pairs (80-100%)
        n_pairs = len(suspicious_pairs)
        sample_size = int(n_pairs * np.random.uniform(0.8, 1.0))
        indices = np.random.choice(n_pairs, size=sample_size, replace=False)
        
        sampled_pairs = [suspicious_pairs[i] for i in indices]
        sampled_scores = [similarity_scores[i] for i in indices]
        
        # Calculate metrics for this sample
        n_pairs = len(sampled_pairs)
        total_possible_pairs = (n_patches * (n_patches - 1)) / 2
        raw_probability = n_pairs / total_possible_pairs
        
        # Calculate offset vectors
        offsets = []
        for (x1, y1), (x2, y2) in sampled_pairs:
            offsets.append((x2 - x1, y2 - y1))
        
        # Calculate spatial coherence
        coherence_score = compute_advanced_coherence(sampled_pairs, patch_size)
        
        # Analyze offset distribution
        offset_metrics = analyze_offset_distribution(offsets)
        
        # Calculate pattern strength
        pattern_strength = calculate_pattern_strength(
            offset_metrics,
            coherence_score,
            0.5,  # Use a default value since we don't compute regional concentration in MC samples
            n_pairs / max(1, n_patches)
        )
        
        # Calculate confidence for this sample
        base_score = raw_probability * 10
        confidence = (
            0.2 * base_score +
            coherence_weight * coherence_score +
            pattern_weight * pattern_strength
        )
        confidence = min(1.0, confidence)
        samples.append(confidence)
    
    return samples


def compute_advanced_coherence(pairs, patch_size):
    """
    Compute spatial coherence score with enhanced metrics
    
    Args:
        pairs: List of ((x1, y1), (x2, y2)) coordinates for suspicious pairs
        patch_size: Size of patches
        
    Returns:
        coherence_score: Enhanced coherence score (0-1)
    """
    if len(pairs) < 3:
        return 0.7  # Default coherence for small sets of pairs
    
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
    coherence = 1.0 - min(1.0, norm_std * 0.7)
    
    # Boost coherence score for larger sets of pairs
    pair_count_boost = min(0.3, len(pairs) / 100)
    coherence = min(1.0, coherence + pair_count_boost)
    
    # Calculate additional coherence metrics
    # 1. Directional consistency
    offset_angles = np.arctan2(offsets[:, 1], offsets[:, 0])
    angle_std = np.std(offset_angles)
    direction_coherence = 1.0 - min(1.0, angle_std / np.pi)
    
    # 2. Distance consistency
    offset_distances = np.sqrt(offsets[:, 0]**2 + offsets[:, 1]**2)
    distance_cv = np.std(offset_distances) / (np.mean(offset_distances) + 1e-5)
    distance_coherence = 1.0 - min(1.0, distance_cv)
    
    # Combine multiple coherence metrics
    final_coherence = (
        0.6 * coherence + 
        0.2 * direction_coherence + 
        0.2 * distance_coherence
    )
    
    return min(1.0, final_coherence)


def analyze_offset_distribution(offsets):
    """
    Calculate statistics about the distribution of offset vectors
    
    Args:
        offsets: List of (dx, dy) offset vectors
        
    Returns:
        dict: Statistical metrics about offset distribution
    """
    if len(offsets) < 3:
        return {"entropy": 0, "mode_strength": 0, "spread": 0, "most_common_offset": (0, 0)}
    
    # Round offsets to integers for better clustering
    offset_tuples = [(int(round(x)), int(round(y))) for x, y in offsets]
    
    # Count frequency of each offset
    offset_counts = Counter(offset_tuples)
    
    # Calculate mode (most common offset)
    most_common_offset, most_common_count = offset_counts.most_common(1)[0]
    mode_strength = most_common_count / len(offsets)
    
    # Calculate spread (number of unique offsets relative to total)
    spread = len(offset_counts) / len(offsets)
    
    # Calculate entropy (higher means more random distribution)
    probs = [count/len(offsets) for count in offset_counts.values()]
    entropy = -sum(p * np.log2(p) for p in probs)
    
    # Calculate directional statistics
    offset_array = np.array(offset_tuples)
    directions = np.arctan2(offset_array[:, 1], offset_array[:, 0])
    direction_counts = Counter(np.round(directions, 2))
    max_dir_count = max(direction_counts.values())
    direction_consistency = max_dir_count / len(directions)
    
    return {
        "entropy": entropy,
        "mode_strength": mode_strength,
        "spread": spread,
        "most_common_offset": most_common_offset,
        "direction_consistency": direction_consistency
    }


def calculate_regional_concentration(suspicious_pairs, image_size):
    """
    Calculate how concentrated the suspicious pairs are in specific image regions
    
    Args:
        suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinates for suspicious pairs
        image_size: Tuple (width, height) of the image size
        
    Returns:
        concentration_score: Score indicating regional concentration (0-1)
    """
    if len(suspicious_pairs) < 5:
        return 0.5  # Default for small number of pairs
    
    # Extract source and target points
    sources = []
    targets = []
    for (x1, y1), (x2, y2) in suspicious_pairs:
        sources.append((x1, y1))
        targets.append((x2, y2))
    
    # Divide image into 4x4 grid (16 regions)
    w, h = image_size
    region_w, region_h = w // 4, h // 4
    
    # Count pairs in each region
    source_regions = defaultdict(int)
    target_regions = defaultdict(int)
    
    for x, y in sources:
        region_x, region_y = min(3, x // region_w), min(3, y // region_h)
        source_regions[(region_x, region_y)] += 1
    
    for x, y in targets:
        region_x, region_y = min(3, x // region_w), min(3, y // region_h)
        target_regions[(region_x, region_y)] += 1
    
    # Calculate maximum concentration in any region
    max_source_conc = max(source_regions.values()) / len(sources) if sources else 0
    max_target_conc = max(target_regions.values()) / len(targets) if targets else 0
    
    # Calculate how many regions have significant number of points
    threshold = len(suspicious_pairs) * 0.1
    src_active_regions = sum(1 for count in source_regions.values() if count > threshold)
    tgt_active_regions = sum(1 for count in target_regions.values() if count > threshold)
    
    # Normalize regional activity (1 = very concentrated, 0 = widely dispersed)
    src_region_ratio = 1.0 - (src_active_regions / 16)
    tgt_region_ratio = 1.0 - (tgt_active_regions / 16)
    
    # Combine metrics
    concentration_score = (
        0.4 * max_source_conc +
        0.4 * max_target_conc +
        0.1 * src_region_ratio +
        0.1 * tgt_region_ratio
    )
    
    return concentration_score


def calculate_pattern_strength(offset_metrics, coherence, regional_concentration, pair_density):
    """
    Calculate a unified pattern strength score from multiple metrics
    
    Args:
        offset_metrics: Dictionary of offset distribution metrics
        coherence: Spatial coherence score
        regional_concentration: Regional concentration score
        pair_density: Ratio of suspicious pairs to patches
        
    Returns:
        pattern_strength: Unified pattern strength score (0-1)
    """
    # Extract relevant metrics
    entropy = offset_metrics.get('entropy', 0)
    mode_strength = offset_metrics.get('mode_strength', 0)
    spread = offset_metrics.get('spread', 1)
    direction_consistency = offset_metrics.get('direction_consistency', 0)
    
    # Calculate entropy factor (lower entropy -> stronger pattern)
    max_entropy = 10.0  # Approximate max possible entropy value
    entropy_factor = max(0, 1 - (entropy / max_entropy))
    
    # Calculate density factor
    density_factor = min(1.0, pair_density * 20)  # Scale up the density
    
    # Combined pattern strength
    pattern_strength = (
        0.30 * coherence +
        0.20 * mode_strength +
        0.15 * (1 - spread) +
        0.15 * direction_consistency +
        0.10 * regional_concentration +
        0.10 * density_factor
    )
    
    # Apply entropy penalty for patterns that are too random
    pattern_strength *= (0.7 + 0.3 * entropy_factor)
    
    return min(1.0, pattern_strength)


def categorize_evidence_strength(pattern_strength, uncertainty):
    """
    Categorize the evidence strength based on pattern strength and uncertainty
    
    Args:
        pattern_strength: Pattern strength score (0-1)
        uncertainty: Uncertainty in confidence score
        
    Returns:
        category: String category of evidence strength
    """
    if pattern_strength < 0.2:
        return 'none'
    elif pattern_strength < 0.4:
        return 'weak'
    elif pattern_strength < 0.6:
        if uncertainty > 0.3:
            return 'moderate-uncertain'
        else:
            return 'moderate'
    elif pattern_strength < 0.8:
        if uncertainty > 0.25:
            return 'strong-uncertain'
        else:
            return 'strong'
    else:
        if uncertainty > 0.2:
            return 'very-strong-uncertain'
        else:
            return 'very-strong'