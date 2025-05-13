"""
Region-aware spatial analysis for copy-move forgery detection.
This module provides tools for segmenting images into regions based on texture
and analyzing suspicious pairs in the context of these regions.
"""

import numpy as np
import cv2
from sklearn.cluster import KMeans
from skimage.segmentation import quickshift, felzenszwalb, slic
from skimage.feature import local_binary_pattern
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN # Added for clustering placeholder
from collections import Counter # Added for consistency placeholder


class RegionAnalyzer:
    """
    Analyzes suspicious pairs in the context of image regions
    """
    
    def __init__(self, segmentation_method='slic', n_segments=100):
        """
        Initialize region analyzer
        
        Args:
            segmentation_method: Method for image segmentation ('slic', 'quickshift', 'felzenszwalb')
            n_segments: Target number of segments for segmentation
        """
        self.segmentation_method = segmentation_method
        self.n_segments = n_segments
        
    def segment_image(self, image):
        """
        Segment image into regions based on texture and color
        
        Args:
            image: Input image (BGR format)
            
        Returns:
            segments: Segmentation mask with segment IDs
            region_stats: Dictionary with statistics for each segment
        """
        # Convert to RGB for skimage functions
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Perform segmentation based on method
        if self.segmentation_method == 'slic':
            segments = slic(rgb_image, n_segments=self.n_segments, compactness=10,
                          start_label=0)
        elif self.segmentation_method == 'quickshift':
            segments = quickshift(rgb_image, kernel_size=3, max_dist=6, ratio=0.5)
        elif self.segmentation_method == 'felzenszwalb':
            segments = felzenszwalb(rgb_image, scale=100, sigma=0.5, min_size=50)
        else:
            raise ValueError(f"Unknown segmentation method: {self.segmentation_method}")
            
        # Calculate region statistics
        region_stats = self._calculate_region_stats(image, segments)
        
        return segments, region_stats
        
    def _calculate_region_stats(self, image, segments):
        """
        Calculate statistics for each segment
        
        Args:
            image: Input image (BGR format)
            segments: Segmentation mask with segment IDs
            
        Returns:
            region_stats: Dictionary with statistics for each segment
        """
        # Get unique segment IDs
        segment_ids = np.unique(segments)
        
        # Calculate color statistics for each segment
        region_stats = {}
        
        for segment_id in segment_ids:
            # Create mask for this segment
            mask = segments == segment_id
            
            # Calculate number of pixels in this segment
            n_pixels = np.sum(mask)
            
            # Skip very small segments
            if n_pixels < 100:
                continue
                
            # Calculate color statistics
            color_mean = np.mean(image[mask], axis=0)
            color_std = np.std(image[mask], axis=0)
            
            # Calculate texture features using LBP
            if len(image.shape) == 3:
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray_image = image
                
            # Calculate LBP for texture analysis
            lbp = local_binary_pattern(gray_image, P=8, R=1, method='uniform')
            lbp_hist, _ = np.histogram(lbp[mask], bins=10, range=(0, 10), density=True)
            
            # Store statistics
            region_stats[segment_id] = {
                'n_pixels': n_pixels,
                'color_mean': color_mean,
                'color_std': color_std,
                'texture': lbp_hist,
                'centroid': self._calculate_centroid(mask)
            }
            
        return region_stats
        
    def _calculate_centroid(self, mask):
        """Calculate centroid of a binary mask"""
        # Get coordinates of mask pixels
        coords = np.where(mask)
        
        # Calculate centroid
        if len(coords[0]) > 0:
            centroid_y = np.mean(coords[0])
            centroid_x = np.mean(coords[1])
            return (centroid_x, centroid_y)
        else:
            return (0, 0)
            
    def analyze_suspicious_pairs(self, image, suspicious_pairs, similarity_scores=None, patch_size=64):
        """
        Analyze suspicious pairs in the context of image regions
        
        Args:
            image: Input image (BGR format)
            suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
            similarity_scores: Optional list of similarity scores for each pair
            patch_size: Size of each patch
            
        Returns:
            results: Dictionary with region-based analysis results
        """
        # Segment image into regions
        segments, region_stats = self.segment_image(image)
        
        # If no suspicious pairs, return empty results
        if not suspicious_pairs:
            return {
                'forgery_regions': [],
                'region_confidence_scores': {},
                'visualization': None
            }
        
        # Map patches to segments
        patch_segments = {}
        for i, ((x1, y1), (x2, y2)) in enumerate(suspicious_pairs):
            # Get segment IDs for source and destination patches
            source_patch_segments = self._get_patch_segments(segments, x1, y1, patch_size)
            dest_patch_segments = self._get_patch_segments(segments, x2, y2, patch_size)
            
            # Store segment mappings
            patch_segments[i] = {
                'source': source_patch_segments,
                'dest': dest_patch_segments
            }
        
        # Group suspicious pairs by segment pairs
        segment_pairs = self._group_by_segments(suspicious_pairs, patch_segments)
        
        # Analyze each segment pair
        forgery_regions = []
        region_confidence_scores = {}
        
        for (source_segment, dest_segment), pair_indices in segment_pairs.items():
            if source_segment == dest_segment:
                # Skip pairs within the same segment
                continue
                
            # Calculate confidence score for this segment pair
            if similarity_scores:
                # Use average similarity score for pairs in this segment pair
                confidence = np.mean([similarity_scores[i] for i in pair_indices])
            else:
                # Use number of pairs as a proxy for confidence
                confidence = min(1.0, len(pair_indices) / 10)
                
            # Only consider pairs with sufficient evidence
            if confidence > 0.5 and len(pair_indices) >= 3:
                forgery_regions.append((source_segment, dest_segment))
                region_confidence_scores[(source_segment, dest_segment)] = confidence
        
        # Create visualization
        visualization = self._create_visualization(image, segments, suspicious_pairs, 
                                               forgery_regions, region_confidence_scores, patch_size)
        
        return {
            'forgery_regions': forgery_regions,
            'region_confidence_scores': region_confidence_scores,
            'visualization': visualization
        }
        
    def _get_patch_segments(self, segments, x, y, patch_size):
        """Get segment IDs for a patch"""
        # Get patch region
        patch = segments[y:y+patch_size, x:x+patch_size]
        
        # Count segment IDs in patch
        unique, counts = np.unique(patch, return_counts=True)
        
        # Sort by count (descending)
        sorted_indices = np.argsort(-counts)
        sorted_segments = unique[sorted_indices]
        sorted_counts = counts[sorted_indices]
        
        # Return segment IDs and their proportions
        total_pixels = patch_size * patch_size
        segment_props = {s: c / total_pixels for s, c in zip(sorted_segments, sorted_counts)}
        
        return segment_props
        
    def _group_by_segments(self, suspicious_pairs, patch_segments):
        """Group suspicious pairs by segment pairs"""
        segment_pairs = {}
        
        for i, ((x1, y1), (x2, y2)) in enumerate(suspicious_pairs):
            # Get dominant segments for source and destination
            source_segments = patch_segments[i]['source']
            dest_segments = patch_segments[i]['dest']
            
            # Get dominant segment (with highest proportion)
            source_segment = max(source_segments.items(), key=lambda x: x[1])[0]
            dest_segment = max(dest_segments.items(), key=lambda x: x[1])[0]
            
            # Skip if both patches are in the same segment
            if source_segment == dest_segment:
                continue
                
            # Add to segment pairs
            segment_pair = (source_segment, dest_segment)
            if segment_pair not in segment_pairs:
                segment_pairs[segment_pair] = []
                
            segment_pairs[segment_pair].append(i)
            
        return segment_pairs
        
    def _create_visualization(self, image, segments, suspicious_pairs, forgery_regions, 
                           region_confidence_scores, patch_size):
        """Create visualization of region-based analysis"""
        # Create segmentation visualization with random colors
        n_segments = np.max(segments) + 1
        colors = np.random.randint(0, 255, (n_segments, 3), dtype=np.uint8)
        
        # Create segment overlay
        overlay = np.zeros_like(image)
        for segment_id in range(n_segments):
            mask = segments == segment_id
            overlay[mask] = colors[segment_id]
            
        # Blend with original image
        alpha = 0.3
        blended = cv2.addWeighted(image, 1 - alpha, overlay, alpha, 0)
        
        # Mark suspicious pairs
        for (x1, y1), (x2, y2) in suspicious_pairs:
            # Draw rectangles for patches
            cv2.rectangle(blended, (x1, y1), (x1 + patch_size, y1 + patch_size), (0, 0, 255), 1)
            cv2.rectangle(blended, (x2, y2), (x2 + patch_size, y2 + patch_size), (0, 0, 255), 1)
            
            # Draw line connecting patches
            cv2.line(blended, 
                  (x1 + patch_size // 2, y1 + patch_size // 2), 
                  (x2 + patch_size // 2, y2 + patch_size // 2),
                  (0, 0, 255), 1)
        
        # Highlight forgery regions
        for source_segment, dest_segment in forgery_regions:
            # Create masks for source and destination segments
            source_mask = segments == source_segment
            dest_mask = segments == dest_segment
            
            # Get confidence score
            confidence = region_confidence_scores.get((source_segment, dest_segment), 0.5)
            
            # Create highlight color based on confidence
            # Higher confidence = more red
            highlight_color = (0, int(255 * (1 - confidence)), int(255 * confidence))
            
            # Highlight segments
            blended[source_mask] = cv2.addWeighted(
                blended[source_mask], 0.5, 
                np.full_like(blended[source_mask], highlight_color), 0.5, 0
            )
            blended[dest_mask] = cv2.addWeighted(
                blended[dest_mask], 0.5, 
                np.full_like(blended[dest_mask], highlight_color), 0.5, 0
            )
            
        return blended
        
    def analyze_coherence(self, suspicious_pairs, similarity_scores=None, image_shape=None):
        """
        Analyze spatial coherence of suspicious pairs
        
        Args:
            suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
            similarity_scores: Optional list of similarity scores for each pair
            image_shape: Shape of the image (h, w) or (h, w, c)
            
        Returns:
            coherence: Spatial coherence score (0-1)
            pattern_strength: Pattern strength score (0-1)
            primary_offset: Primary offset vector (dx, dy)
        """
        if not suspicious_pairs:
            return 0.0, 0.0, (0, 0)
            
        # Calculate offset vectors
        offsets = []
        for (x1, y1), (x2, y2) in suspicious_pairs:
            offsets.append((x2 - x1, y2 - y1))
            
        # Use similarity scores as weights if provided
        weights = None
        if similarity_scores:
            weights = np.array(similarity_scores)
            weights = weights / np.sum(weights)  # Normalize weights
            
        # Find most common offset (primary offset)
        offset_counts = {}
        for i, (dx, dy) in enumerate(offsets):
            offset = (dx, dy)
            weight = weights[i] if weights is not None else 1.0
            
            if offset not in offset_counts:
                offset_counts[offset] = 0
                
            offset_counts[offset] += weight
            
        # Sort offsets by count (descending)
        sorted_offsets = sorted(offset_counts.items(), key=lambda x: x[1], reverse=True)
        
        # Get primary offset
        primary_offset = sorted_offsets[0][0] if sorted_offsets else (0, 0)
        
        # Calculate proportion of pairs with primary offset
        total_pairs = len(suspicious_pairs)
        primary_count = offset_counts.get(primary_offset, 0)
        proportion_primary = primary_count / total_pairs if total_pairs > 0 else 0
        
        # Calculate coherence based on consistency of offsets
        # Higher coherence means more consistent offsets
        coherence = proportion_primary
        
        # Calculate pattern strength based on spatial arrangement
        pattern_strength = self._calculate_pattern_strength(suspicious_pairs, image_shape)
        
        return coherence, pattern_strength, primary_offset
        
    def _calculate_pattern_strength(self, suspicious_pairs, image_shape):
        """
        Calculate pattern strength based on spatial arrangement of suspicious pairs
        
        Args:
            suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
            image_shape: Shape of the image (h, w) or (h, w, c)
            
        Returns:
            pattern_strength: Pattern strength score (0-1)
        """
        if not suspicious_pairs or not image_shape:
            return 0.0
            
        # Create binary maps for source and destination
        h, w = image_shape[:2] if len(image_shape) > 2 else image_shape
        src_map = np.zeros((h, w), dtype=np.uint8)
        dst_map = np.zeros((h, w), dtype=np.uint8)
        
        # Fill maps
        for (x1, y1), (x2, y2) in suspicious_pairs:
            # Skip pairs outside image bounds
            if (x1 < 0 or y1 < 0 or x1 >= w or y1 >= h or
                x2 < 0 or y2 < 0 or x2 >= w or y2 >= h):
                continue
                
            src_map[y1, x1] = 1
            dst_map[y2, x2] = 1
        
        # Calculate distances between neighboring points in source map
        src_points = np.argwhere(src_map > 0)
        dst_points = np.argwhere(dst_map > 0)
        
        # If not enough points, return low pattern strength
        if len(src_points) < 3 or len(dst_points) < 3:
            return 0.1
            
        # Calculate consistency of relative positions
        src_dists = self._calculate_point_distances(src_points)
        dst_dists = self._calculate_point_distances(dst_points)
        
        # If distances are similar between source and destination points,
        # this indicates a strong pattern (copy-move forgery)
        if len(src_dists) == 0 or len(dst_dists) == 0:
            return 0.0
            
        # Calculate correlation between distance matrices
        pattern_strength = self._calculate_distance_correlation(src_dists, dst_dists)
        
        return pattern_strength
        
    def _calculate_point_distances(self, points):
        """Calculate pairwise distances between points"""
        n_points = len(points)
        distances = []
        
        for i in range(n_points):
            for j in range(i + 1, n_points):
                # Calculate Euclidean distance
                dist = np.sqrt(np.sum((points[i] - points[j]) ** 2))
                distances.append(dist)
                
        return distances
        
    def _calculate_distance_correlation(self, dists1, dists2):
        """Calculate correlation between two distance arrays"""
        # If arrays have different lengths, use the smaller length
        min_length = min(len(dists1), len(dists2))
        
        # If not enough distances, return low correlation
        if min_length < 3:
            return 0.3
            
        # Sort distances
        dists1 = sorted(dists1)[:min_length]
        dists2 = sorted(dists2)[:min_length]
        
        # Convert to numpy arrays
        dists1 = np.array(dists1)
        dists2 = np.array(dists2)
        
        # Use safer correlation calculation to avoid NaN values
        return compute_correlation(dists1, dists2)

# Placeholder function for clustering suspicious pairs
def cluster_suspicious_pairs(suspicious_pairs, similarity_scores, patch_size, eps=50, min_samples=3):
    """
    Cluster suspicious pairs based on spatial proximity and offset consistency.
    Placeholder implementation using DBSCAN on offsets.
    """
    if not suspicious_pairs:
        return []

    offsets = np.array([(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in suspicious_pairs])
    
    # Cluster based on offset vectors
    try:
        clustering = DBSCAN(eps=eps, min_samples=min_samples).fit(offsets)
        labels = clustering.labels_
    except ValueError: # Handle case with too few samples for DBSCAN
        labels = np.zeros(len(suspicious_pairs), dtype=int) # Assign all to one cluster

    # Group pairs by cluster label
    clusters = {}
    for i, label in enumerate(labels):
        if label == -1:  # Skip noise points
            continue
        if label not in clusters:
            clusters[label] = {'pairs': [], 'scores': [], 'offsets': []}
        
        clusters[label]['pairs'].append(suspicious_pairs[i])
        clusters[label]['scores'].append(similarity_scores[i])
        clusters[label]['offsets'].append(offsets[i])

    # Format output
    output_clusters = []
    for label, data in clusters.items():
        if len(data['pairs']) >= min_samples: # Only keep clusters with enough samples
             output_clusters.append({
                 'pairs': data['pairs'],
                 'scores': data['scores'],
                 'avg_score': np.mean(data['scores']),
                 'offset_mode': Counter(map(tuple, data['offsets'])).most_common(1)[0][0] if data['offsets'] else (0,0)
             })
             
    print(f"  (Debug Clustering: Found {len(output_clusters)} clusters from {len(suspicious_pairs)} pairs)")
    return output_clusters

# Placeholder function for verifying geometric consistency
def verify_geometric_consistency(cluster, patch_size, threshold=0.8):
    """
    Verify geometric consistency within a cluster of suspicious pairs.
    Placeholder implementation checking offset consistency.
    """
    if not cluster or 'offsets' not in cluster or not cluster['offsets']:
         # If cluster is missing offsets (e.g., from placeholder clustering), calculate them
         offsets = [(p2[0] - p1[0], p2[1] - p1[1]) for p1, p2 in cluster.get('pairs', [])]
         if not offsets:
             return False, 0.0 # Cannot verify consistency without offsets or pairs
         cluster['offsets'] = offsets # Store for calculation

    offsets = np.array(cluster['offsets'])
    if len(offsets) < 2:
        return True, 1.0 # Single pair or no pairs are trivially consistent

    # Calculate standard deviation of offsets
    std_offset = np.std(offsets, axis=0)
    
    # Normalize deviation by patch size
    norm_std = np.mean(std_offset) / patch_size
    
    # Consistency score (lower deviation = higher consistency)
    consistency_score = max(0.0, 1.0 - norm_std) # Simple inverse relationship

    is_consistent = consistency_score >= threshold
    print(f"  (Debug Consistency: Cluster size={len(offsets)}, NormStd={norm_std:.4f}, Score={consistency_score:.4f}, Valid={is_consistent})")

    return is_consistent, consistency_score

# Function moved from infer.py
def compute_spatial_coherence(pairs, patch_size):
    """
    Compute spatial coherence score for suspicious pairs
    Higher coherence = more likely to be an actual forgery
    """
    if len(pairs) < 3:
        return 0.7  # Increase base coherence for small sets of pairs
    
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
    # Make more lenient by reducing the penalty for variation
    coherence = 1.0 - min(1.0, norm_std * 0.7)
    
    # Boost coherence score for larger sets of pairs
    pair_count_boost = min(0.3, len(pairs) / 100)
    coherence = min(1.0, coherence + pair_count_boost)
    
    return coherence

def compute_correlation(array1, array2):
    """Compute normalized correlation between two arrays with NaN protection"""
    # Flatten arrays
    array1 = array1.flatten()
    array2 = array2.flatten()
    
    # Calculate means
    mean1 = np.mean(array1)
    mean2 = np.mean(array2)
    
    # Calculate denominators with protection against zero
    denom1 = np.std(array1)
    denom2 = np.std(array2)
    
    # Handle zero or near-zero standard deviations
    if denom1 < 1e-10 or denom2 < 1e-10:
        return 0.0
    
    # Calculate correlation with protection against NaN
    try:
        corr = np.sum((array1 - mean1) * (array2 - mean2)) / (denom1 * denom2 * len(array1))
        if np.isnan(corr):
            return 0.0
        return corr
    except:
        return 0.0