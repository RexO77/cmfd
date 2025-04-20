"""
Enhanced visualization for copy-move forgery detection results.
This module provides improved visualization tools to display forgery detection
results with confidence levels and comprehensive explanations.
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as patches
from matplotlib.lines import Line2D
import io
from PIL import Image


def generate_heatmap(image, suspicious_pairs, similarity_scores=None, patch_size=64, alpha=0.6):
    """
    Generate a heat map visualizing potentially forged regions
    
    Args:
        image: Input image (BGR format)
        suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
        similarity_scores: Optional list of similarity scores for each pair
        patch_size: Size of each patch
        alpha: Transparency level for heatmap overlay
        
    Returns:
        heatmap_overlay: Image with heatmap overlay
    """
    # Create empty heatmap
    h, w = image.shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    
    # Use similarity scores if provided, otherwise use default value
    if similarity_scores is None:
        similarity_scores = [1.0] * len(suspicious_pairs)
    
    # Normalize similarity scores to [0, 1]
    if similarity_scores:
        min_score = min(similarity_scores)
        max_score = max(similarity_scores)
        if max_score > min_score:
            norm_scores = [(s - min_score) / (max_score - min_score) for s in similarity_scores]
        else:
            norm_scores = [1.0] * len(similarity_scores)
    else:
        norm_scores = []
    
    # Fill heatmap based on suspicious pairs and their similarity scores
    for (x1, y1), (x2, y2), score in zip(suspicious_pairs, [p[1] for p in suspicious_pairs], norm_scores):
        # Add to heatmap with intensity based on similarity score
        intensity = 0.4 + 0.6 * score  # Scale to [0.4, 1.0] to ensure visibility
        
        # Use a Gaussian blob for smoother visualization
        xc1, yc1 = x1 + patch_size // 2, y1 + patch_size // 2
        xc2, yc2 = x2 + patch_size // 2, y2 + patch_size // 2
        
        sigma = patch_size // 3
        
        # Create Gaussian kernel for smoother heatmap
        y, x = np.ogrid[-sigma:sigma+1, -sigma:sigma+1]
        gauss = np.exp(-(x*x + y*y) / (2*sigma*sigma))
        gauss = gauss / gauss.max()
        
        # Add to source and destination in heatmap
        for xc, yc in [(xc1, yc1), (xc2, yc2)]:
            x_min = max(0, xc - sigma)
            x_max = min(w, xc + sigma + 1)
            y_min = max(0, yc - sigma)
            y_max = min(h, yc + sigma + 1)
            
            gauss_x_min = max(0, sigma - xc)
            gauss_x_max = min(2*sigma+1, sigma + (w - xc))
            gauss_y_min = max(0, sigma - yc)
            gauss_y_max = min(2*sigma+1, sigma + (h - yc))
            
            heatmap_slice = heatmap[y_min:y_max, x_min:x_max]
            gauss_slice = gauss[gauss_y_min:gauss_y_max, gauss_x_min:gauss_x_max]
            
            if heatmap_slice.shape == gauss_slice.shape:
                heatmap_slice += intensity * gauss_slice
    
    # Normalize heatmap
    heatmap = np.clip(heatmap, 0, 1)
    
    # Apply colormap (using jet colormap)
    heatmap_colored = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    
    # Create mask for non-zero regions
    mask = heatmap > 0.01
    
    # Create overlay image
    overlay = image.copy()
    overlay[mask] = cv2.addWeighted(image, 1-alpha, heatmap_colored, alpha, 0)[mask]
    
    return overlay, heatmap


def create_detailed_visualization(image, results, patch_size=64, max_pairs=50):
    """
    Create detailed visualization of forgery detection results
    
    Args:
        image: Input image
        results: Dictionary with detection results
        patch_size: Patch size
        max_pairs: Maximum number of suspicious pairs to visualize
        
    Returns:
        visualization: Detailed visualization image
    """
    # Extract results
    is_forged = results.get('is_forged', False)
    confidence_score = results.get('confidence_score', 0)
    evidence_metrics = results.get('evidence_metrics', {})
    suspicious_pairs = results.get('suspicious_pairs', [])
    similarity_scores = results.get('similarity_scores', [])
    
    # Limit number of pairs to visualize
    if len(suspicious_pairs) > max_pairs:
        # Sort by similarity score if available, otherwise use the first pairs
        if similarity_scores:
            sorted_pairs = sorted(zip(suspicious_pairs, similarity_scores), 
                                key=lambda x: x[1], reverse=True)
            suspicious_pairs = [p for p, s in sorted_pairs[:max_pairs]]
            similarity_scores = [s for p, s in sorted_pairs[:max_pairs]]
        else:
            suspicious_pairs = suspicious_pairs[:max_pairs]
    
    # Create figure with subplots
    fig = plt.figure(figsize=(18, 12))
    
    # Define grid layout
    grid = plt.GridSpec(2, 3, height_ratios=[3, 1])
    
    # 1. Original Image
    ax_image = fig.add_subplot(grid[0, 0])
    ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_image.set_title("Original Image")
    ax_image.axis("off")
    
    # 2. Heatmap overlay
    ax_heatmap = fig.add_subplot(grid[0, 1])
    overlay, heatmap = generate_heatmap(image, suspicious_pairs, similarity_scores, patch_size)
    ax_heatmap.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_heatmap.set_title(f"Forgery Heatmap (Confidence: {confidence_score:.3f})")
    ax_heatmap.axis("off")
    
    # 3. Connection visualization
    ax_connections = fig.add_subplot(grid[0, 2])
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    ax_connections.imshow(rgb_image)
    
    # Draw connections between pairs with color based on similarity
    if suspicious_pairs:
        if similarity_scores:
            min_score = min(similarity_scores)
            max_score = max(similarity_scores)
            norm = plt.Normalize(min_score, max_score)
        else:
            norm = plt.Normalize(0, 1)
            similarity_scores = [1.0] * len(suspicious_pairs)
            
        # Create colormap for connections
        cmap = plt.cm.plasma
            
        for (x1, y1), (x2, y2), score in zip(
            [p[0] for p in suspicious_pairs],
            [p[1] for p in suspicious_pairs],
            similarity_scores
        ):
            # Calculate center of patches
            center1 = (x1 + patch_size // 2, y1 + patch_size // 2)
            center2 = (x2 + patch_size // 2, y2 + patch_size // 2)
            
            # Draw connection line with color and transparency based on similarity
            color = cmap(norm(score))
            linewidth = 1 + 2 * (score - min_score) / max(max_score - min_score, 0.001)
            alpha = 0.4 + 0.6 * (score - min_score) / max(max_score - min_score, 0.001)
            
            ax_connections.plot([center1[0], center2[0]], [center1[1], center2[1]], 
                             color=color, linewidth=linewidth, alpha=alpha)
            
            # Draw rectangles for matched regions
            rect1 = patches.Rectangle((x1, y1), patch_size, patch_size, 
                                   linewidth=1, edgecolor=color, facecolor='none', alpha=alpha)
            rect2 = patches.Rectangle((x2, y2), patch_size, patch_size, 
                                   linewidth=1, edgecolor=color, facecolor='none', alpha=alpha)
            
            ax_connections.add_patch(rect1)
            ax_connections.add_patch(rect2)
            
    ax_connections.set_title("Suspicious Region Connections")
    ax_connections.axis("off")
    
    # 4. Evidence summary
    ax_evidence = fig.add_subplot(grid[1, :])
    
    # Format decision text
    if is_forged:
        decision = "FORGED"
        color = 'red'
    else:
        decision = "AUTHENTIC"
        color = 'green'
        
    title_text = f"DECISION: {decision} (Confidence: {confidence_score:.3f})"
    
    # Get confidence interval if available
    if 'confidence_interval' in results:
        lower, upper = results['confidence_interval']
        title_text += f" - 95% CI: [{lower:.3f}, {upper:.3f}]"
        
    ax_evidence.text(0.5, 0.9, title_text, fontsize=14, ha='center', color=color, weight='bold')
    
    # Add evidence metrics
    evidence_text = []
    
    if evidence_metrics:
        # Format suspicious pairs info
        n_pairs = evidence_metrics.get('suspicious_pairs', 0)
        evidence_text.append(f"• Suspicious Matching Regions: {n_pairs}")
        
        # Format coherence info
        coherence = evidence_metrics.get('coherence', 0)
        evidence_text.append(f"• Spatial Coherence: {coherence:.3f}")
        
        # Format offset info
        primary_offset = evidence_metrics.get('primary_offset')
        if primary_offset:
            dx, dy = primary_offset
            evidence_text.append(f"• Primary Offset Vector: ({dx}, {dy})")
        
        # Format pattern strength
        pattern_strength = evidence_metrics.get('pattern_strength', 0)
        evidence_text.append(f"• Pattern Strength: {pattern_strength:.3f}")
        
        # Format uncertainty
        uncertainty = evidence_metrics.get('uncertainty', 0)
        evidence_text.append(f"• Uncertainty: ±{uncertainty:.3f}")
        
        # Format evidence strength category
        evidence_strength = evidence_metrics.get('evidence_strength', 'unknown')
        evidence_text.append(f"• Evidence Strength: {evidence_strength}")
    
    # Format threshold results if available
    if 'threshold_results' in results:
        threshold_results = results['threshold_results']
        threshold_text = ["Multiple Threshold Analysis:"]
        
        for threshold, result in threshold_results.items():
            confidence = result.get('confidence_score', 0)
            n_pairs = len(result.get('suspicious_pairs', []))
            threshold_text.append(f"  - Threshold {threshold:.2f}: {confidence:.3f} confidence, {n_pairs} pairs")
            
        evidence_text.append("\n" + "\n".join(threshold_text))
    
    # Add frequency domain results if available
    if 'frequency_results' in results:
        freq_results = results['frequency_results']
        freq_text = ["Frequency Domain Analysis:"]
        
        dct_similarity = freq_results.get('dct_similarity', 0)
        freq_text.append(f"  - DCT Coefficient Similarity: {dct_similarity:.3f}")
        
        has_artifacts = freq_results.get('has_compression_artifacts', False)
        freq_text.append(f"  - JPEG Compression Artifacts: {'Detected' if has_artifacts else 'Not Detected'}")
        
        n_inconsistent = len(freq_results.get('inconsistent_regions', []))
        freq_text.append(f"  - Inconsistent Noise Regions: {n_inconsistent}")
        
        evidence_text.append("\n" + "\n".join(freq_text))
    
    # Add region-based results if available
    if 'region_results' in results:
        region_results = results['region_results']
        region_text = ["Region-Based Analysis:"]
        
        n_forgery_regions = len(region_results.get('forgery_regions', []))
        region_text.append(f"  - Detected Forgery Regions: {n_forgery_regions}")
        
        evidence_text.append("\n" + "\n".join(region_text))
    
    # Display text
    ax_evidence.text(0.1, 0.7, "\n".join(evidence_text), fontsize=11, ha='left', va='top')
    
    # Add explanation of visualization
    ax_evidence.text(0.1, 0.1, 
                  "Visualization Guide:\n"
                  "1. Original image shown on the left\n"
                  "2. Heat map overlay in the middle (brighter regions indicate potential forgeries)\n"
                  "3. Suspicious region connections on the right (line color indicates similarity strength)",
                  fontsize=10, style='italic')
    
    ax_evidence.axis('off')
    
    # Finalize figure
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def create_report_visualization(image_path, results, include_frequency=True, include_regions=True):
    """
    Create a comprehensive report visualization for a forgery detection result
    
    Args:
        image_path: Path to the image
        results: Dictionary with detection results
        include_frequency: Whether to include frequency domain analysis
        include_regions: Whether to include region-based analysis
        
    Returns:
        report_image: Report visualization image
    """
    # Load image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Could not load image from {image_path}")
    
    # Create figure
    fig = plt.figure(figsize=(12, 12))
    
    # Define grid layout based on whether we have frequency and region analysis
    n_rows = 2 + int(include_frequency) + int(include_regions)
    grid = plt.GridSpec(n_rows, 2, height_ratios=[3] + [2] * (n_rows - 1))
    
    # 1. Original image and heatmap
    ax_image = fig.add_subplot(grid[0, 0])
    ax_image.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    ax_image.set_title("Original Image")
    ax_image.axis("off")
    
    ax_heatmap = fig.add_subplot(grid[0, 1])
    suspicious_pairs = results.get('suspicious_pairs', [])
    similarity_scores = results.get('similarity_scores', [])
    overlay, _ = generate_heatmap(image, suspicious_pairs, similarity_scores)
    ax_heatmap.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    ax_heatmap.set_title("Forgery Heatmap")
    ax_heatmap.axis("off")
    
    # 2. Detailed suspicious pairs
    ax_pairs = fig.add_subplot(grid[1, 0])
    plot_suspicious_pairs_details(ax_pairs, results)
    
    # 3. Evidence summary
    ax_evidence = fig.add_subplot(grid[1, 1])
    plot_evidence_summary(ax_evidence, results)
    
    # 4. Frequency domain analysis if available
    current_row = 2
    if include_frequency and 'frequency_results' in results:
        ax_freq = fig.add_subplot(grid[current_row, :])
        plot_frequency_analysis(ax_freq, results['frequency_results'])
        current_row += 1
        
    # 5. Region-based analysis if available
    if include_regions and 'region_results' in results:
        ax_region = fig.add_subplot(grid[current_row, :])
        plot_region_analysis(ax_region, results['region_results'])
    
    # Finalize figure
    plt.tight_layout()
    
    # Convert matplotlib figure to image
    fig.canvas.draw()
    img = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
    img = img.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    
    plt.close(fig)
    
    return img


def plot_suspicious_pairs_details(ax, results):
    """Plot detailed visualization of suspicious pairs"""
    suspicious_pairs = results.get('suspicious_pairs', [])
    similarity_scores = results.get('similarity_scores', [])
    evidence_metrics = results.get('evidence_metrics', {})
    
    # If no pairs, show message
    if not suspicious_pairs:
        ax.text(0.5, 0.5, "No suspicious pairs detected", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
    
    # Create a scatter plot of offset vectors
    offsets = []
    for (x1, y1), (x2, y2) in suspicious_pairs:
        offsets.append((x2 - x1, y2 - y1))
        
    dx = [x for x, y in offsets]
    dy = [y for x, y in offsets]
    
    # Use similarity scores for color mapping if available
    if similarity_scores:
        c = similarity_scores
        norm = plt.Normalize(min(c), max(c))
        cmap = plt.cm.plasma
    else:
        c = 'blue'
        norm = None
        cmap = None
    
    # Scatter plot of offset vectors
    scatter = ax.scatter(dx, dy, c=c, cmap=cmap, alpha=0.7, edgecolors='k', linewidths=0.5)
    
    # Add colorbar if using similarity scores
    if similarity_scores:
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Similarity Score')
    
    # Mark center for reference
    ax.plot(0, 0, 'r+', markersize=10)
    
    # Plot circle representing maximum distance from center
    max_radius = max(np.sqrt(np.array(dx)**2 + np.array(dy)**2))
    circle = plt.Circle((0, 0), max_radius, fill=False, color='gray', linestyle='--', alpha=0.5)
    ax.add_patch(circle)
    
    # Add details from evidence metrics
    coherence = evidence_metrics.get('coherence', 0)
    pattern_strength = evidence_metrics.get('pattern_strength', 0)
    primary_offset = evidence_metrics.get('primary_offset')
    
    # Mark primary offset if available
    if primary_offset:
        pdx, pdy = primary_offset
        ax.plot(pdx, pdy, 'ro', markersize=8)
        ax.annotate("Primary offset", (pdx, pdy), xytext=(10, 10), 
                 textcoords="offset points", color='red')
    
    # Add stats text
    stats_text = (
        f"Suspicious pairs: {len(suspicious_pairs)}\n"
        f"Spatial coherence: {coherence:.3f}\n"
        f"Pattern strength: {pattern_strength:.3f}"
    )
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
         verticalalignment='top', horizontalalignment='left',
         bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title("Offset Vector Distribution")
    ax.set_xlabel("Horizontal Offset (dx)")
    ax.set_ylabel("Vertical Offset (dy)")
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Make axes equal to maintain circular appearance
    ax.set_aspect('equal')


def plot_evidence_summary(ax, results):
    """Plot evidence summary visualization"""
    is_forged = results.get('is_forged', False)
    confidence_score = results.get('confidence_score', 0)
    evidence_metrics = results.get('evidence_metrics', {})
    
    # Format decision text
    if is_forged:
        decision = "FORGED"
        color = 'red'
    else:
        decision = "AUTHENTIC"
        color = 'green'
        
    title_text = f"DECISION: {decision}"
    ax.text(0.5, 0.95, title_text, fontsize=16, ha='center', va='top', color=color, weight='bold')
    
    # Add confidence score
    ax.text(0.5, 0.85, f"Confidence: {confidence_score:.3f}", fontsize=14, ha='center', va='top')
    
    # Add confidence interval if available
    if 'confidence_interval' in results:
        lower, upper = results['confidence_interval']
        ax.text(0.5, 0.78, f"95% Confidence Interval: [{lower:.3f}, {upper:.3f}]", 
             fontsize=12, ha='center', va='top')
    
    # Create bar chart of evidence factors
    if 'evidence_factors' in results:
        factors = results['evidence_factors']
        if factors:
            labels = [f[0] for f in factors]
            values = [f[1] for f in factors]
            
            y_pos = np.arange(len(labels))
            
            # Create horizontal bar chart
            bars = ax.barh(y_pos, values, height=0.5)
            
            # Color bars based on values
            cmap = plt.cm.RdYlGn_r
            norm = plt.Normalize(0, 1)
            
            for bar, val in zip(bars, values):
                bar.set_color(cmap(norm(val)))
            
            # Add value labels to the right of bars
            for i, v in enumerate(values):
                ax.text(v + 0.01, i, f"{v:.2f}", va='center')
            
            # Set labels
            ax.set_yticks(y_pos)
            ax.set_yticklabels(labels)
            ax.set_xlabel('Factor Strength')
            ax.set_title('Key Evidence Factors')
            
            # Limit x-axis to 0-1
            ax.set_xlim(0, 1)
        else:
            # If no factors available
            ax.text(0.5, 0.5, "No evidence factors available", ha='center', va='center', fontsize=12)
            ax.axis('off')
    else:
        # If no detailed explanations available, show simple text
        if evidence_metrics:
            # Format suspicious pairs info
            n_pairs = evidence_metrics.get('suspicious_pairs', 0)
            coherence = evidence_metrics.get('coherence', 0)
            pattern_strength = evidence_metrics.get('pattern_strength', 0)
            
            evidence_text = (
                f"Suspicious Matching Regions: {n_pairs}\n"
                f"Spatial Coherence: {coherence:.3f}\n"
                f"Pattern Strength: {pattern_strength:.3f}"
            )
            
            ax.text(0.5, 0.5, evidence_text, ha='center', va='center', fontsize=12)
        else:
            ax.text(0.5, 0.5, "No detailed evidence metrics available", ha='center', va='center', fontsize=12)
            
        ax.axis('off')


def plot_frequency_analysis(ax, freq_results):
    """Plot frequency domain analysis results"""
    if not freq_results:
        ax.text(0.5, 0.5, "No frequency analysis results available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
        
    # Create a 3-panel subplot within the given axes
    gs = ax.get_gridspec()
    sub_gs = gs[gs.get_geometry()[0]-1, :].subgridspec(1, 3)
    
    # Remove the original axis
    ax.remove()
    
    # Create three panels
    ax1 = plt.subplot(sub_gs[0])
    ax2 = plt.subplot(sub_gs[1])
    ax3 = plt.subplot(sub_gs[2])
    
    # Panel 1: DCT similarity
    dct_similarity = freq_results.get('dct_similarity', 0)
    ax1.add_patch(patches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='k'))
    ax1.add_patch(patches.Rectangle((0, 0), 1, dct_similarity, facecolor='b', alpha=0.7))
    ax1.text(0.5, 0.5, f"{dct_similarity:.3f}", ha='center', va='center', fontsize=14, color='w' if dct_similarity > 0.5 else 'k')
    ax1.set_title("DCT Similarity")
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Panel 2: Compression artifacts heatmap
    if 'compression_heatmap' in freq_results and freq_results['compression_heatmap'] is not None:
        ax2.imshow(freq_results['compression_heatmap'], cmap='jet')
        
        # Add indicator for detected artifacts
        has_artifacts = freq_results.get('has_compression_artifacts', False)
        title = "JPEG Compression"
        if has_artifacts:
            title += "\n(Artifacts Detected)"
            ax2.patch.set_edgecolor('red')
            ax2.patch.set_linewidth(2)
    else:
        ax2.text(0.5, 0.5, "No compression\nanalysis", ha='center', va='center')
        title = "JPEG Compression"
        
    ax2.set_title(title)
    ax2.axis('off')
    
    # Panel 3: Noise variance map
    if 'noise_variance_map' in freq_results and freq_results['noise_variance_map'] is not None:
        ax3.imshow(freq_results['noise_variance_map'], cmap='viridis')
        
        # Mark inconsistent regions
        inconsistent_regions = freq_results.get('inconsistent_regions', [])
        for x, y in inconsistent_regions:
            ax3.add_patch(patches.Rectangle((x, y), 16, 16, fill=False, edgecolor='r', linewidth=1))
            
        title = "Noise Analysis"
        if inconsistent_regions:
            title += f"\n({len(inconsistent_regions)} inconsistencies)"
    else:
        ax3.text(0.5, 0.5, "No noise\nanalysis", ha='center', va='center')
        title = "Noise Analysis"
        
    ax3.set_title(title)
    ax3.axis('off')
    
    # Add overall frequency analysis result as a suptitle
    manipulated_probability = freq_results.get('manipulated_probability', 0)
    plt.suptitle(f"Frequency Domain Analysis - Manipulation Probability: {manipulated_probability:.3f}", 
              fontsize=14, y=1.05)


def plot_region_analysis(ax, region_results):
    """Plot region-based analysis results"""
    if not region_results:
        ax.text(0.5, 0.5, "No region analysis results available", ha='center', va='center', fontsize=12)
        ax.axis('off')
        return
        
    # Create visualization
    if 'visualization' in region_results and region_results['visualization'] is not None:
        ax.imshow(cv2.cvtColor(region_results['visualization'], cv2.COLOR_BGR2RGB))
    else:
        ax.text(0.5, 0.5, "Region visualization not available", ha='center', va='center')
        
    # Add title with region information
    forgery_regions = region_results.get('forgery_regions', [])
    if forgery_regions:
        title = f"Region-Based Analysis: {len(forgery_regions)} suspicious region pairs detected"
    else:
        title = "Region-Based Analysis: No suspicious regions detected"
        
    ax.set_title(title)
    ax.axis('off')


def save_visualization(visualization, output_path):
    """
    Save visualization to file
    
    Args:
        visualization: Visualization image
        output_path: Path to save the visualization
        
    Returns:
        None
    """
    # Convert to PIL Image
    if isinstance(visualization, np.ndarray):
        image = Image.fromarray(visualization)
        image.save(output_path)
    else:
        # If it's already a PIL Image
        visualization.save(output_path)
        
    return output_path