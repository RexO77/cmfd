"""
Decision calibration system for copy-move forgery detection.
This module provides tools to calibrate raw scores to more reliable probability
estimates and dynamic decision thresholds based on operating conditions.
"""

import numpy as np
from sklearn.isotonic import IsotonicRegression
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve
import cv2


class ProbabilityCalibrator:
    """Calibrates raw scores to reliable probability estimates"""
    
    def __init__(self, method='platt'):
        """
        Initialize probability calibrator
        
        Args:
            method: Calibration method ('platt', 'isotonic', or 'beta')
        """
        self.method = method
        self.calibrated = False
        
        # For Platt scaling (logistic regression)
        self.platt_a = 0
        self.platt_b = 1
        
        # For isotonic regression
        self.isotonic = IsotonicRegression(out_of_bounds='clip')
        
        # For beta calibration
        self.beta_params = None
        
    def fit(self, scores, labels):
        """
        Fit calibration model on validation data
        
        Args:
            scores: Raw prediction scores (uncalibrated)
            labels: Ground truth labels (1 for forged, 0 for original)
            
        Returns:
            self: Fitted calibrator
        """
        if len(scores) < 5 or len(np.unique(labels)) < 2:
            # Not enough data or only one class, use identity calibration
            self.calibrated = False
            return self
            
        if self.method == 'platt':
            # Platt scaling (simple logistic regression)
            self._fit_platt(scores, labels)
        elif self.method == 'isotonic':
            # Isotonic regression (non-parametric approach)
            self.isotonic.fit(scores, labels)
        elif self.method == 'beta':
            # Beta calibration (more robust for skewed distributions)
            self._fit_beta(scores, labels)
        else:
            raise ValueError(f"Unknown calibration method: {self.method}")
            
        self.calibrated = True
        return self
        
    def _fit_platt(self, scores, labels):
        """
        Fit Platt scaling (logistic regression)
        
        Args:
            scores: Raw prediction scores
            labels: Ground truth labels (1 for forged, 0 for original)
        """
        # Apply simple logistic regression to map scores to probabilities
        # p(y=1|score) = 1 / (1 + exp(a*score + b))
        
        # Avoid numerical instability with extreme values
        eps = 1e-12
        scores = np.clip(scores, eps, 1-eps)
        
        # Calculate optimal parameters using maximum likelihood estimation
        pos = scores[labels == 1]
        neg = scores[labels == 0]
        
        # Count number of positive and negative examples
        n_pos = len(pos)
        n_neg = len(neg)
        
        # Initialize parameters
        def objective(ab, scores, labels, n_pos, n_neg):
            a, b = ab
            logits = a * scores + b
            log_likelihood = 0.0
            
            # Calculate negative log likelihood
            for score, label in zip(scores, labels):
                logit = a * score + b
                if label == 1:
                    log_likelihood += -np.log(1 + np.exp(-logit))
                else:
                    log_likelihood += -np.log(1 + np.exp(logit))
                    
            return -log_likelihood
            
        # Use scipy optimize to find best parameters
        from scipy import optimize
        initial_params = [1, 0]  # Start with identity mapping
        result = optimize.minimize(
            objective, 
            initial_params, 
            args=(scores, labels, n_pos, n_neg),
            method='BFGS'
        )
        
        # Get optimized parameters
        self.platt_a, self.platt_b = result.x
        
    def _fit_beta(self, scores, labels):
        """
        Fit beta calibration model
        
        Args:
            scores: Raw prediction scores
            labels: Ground truth labels
        """
        # Beta calibration is more complex and handles skewed distributions better
        # For simplicity, we'll use a simplified version with curve fitting
        from scipy import optimize
        
        # Define beta calibration function
        def beta_calibration_func(score, a, b, c):
            # Transform score to avoid log(0)
            eps = 1e-12
            score = np.clip(score, eps, 1-eps)
            
            # Apply beta calibration transform
            logit = lambda p: np.log(p / (1 - p))
            return 1 / (1 + np.exp(-(a * logit(score) + b) + c))
            
        # Fit function to data
        def objective(params, scores, labels):
            a, b, c = params
            predicted = beta_calibration_func(scores, a, b, c)
            # Calculate negative log likelihood
            loss = 0
            for pred, label in zip(predicted, labels):
                if label == 1:
                    loss -= np.log(pred + 1e-12)
                else:
                    loss -= np.log(1 - pred + 1e-12)
            return loss
            
        # Initial parameters
        initial_params = [1.0, 0.0, 0.0]
        
        # Find optimal parameters
        result = optimize.minimize(
            objective, 
            initial_params, 
            args=(scores, labels),
            method='Nelder-Mead'
        )
        
        self.beta_params = result.x
        
    def calibrate(self, scores):
        """
        Calibrate raw scores to probabilities
        
        Args:
            scores: Raw prediction scores
            
        Returns:
            calibrated_scores: Calibrated probability estimates
        """
        if not self.calibrated:
            return scores  # Return original scores if not calibrated
            
        if self.method == 'platt':
            # Apply Platt scaling
            return self._apply_platt(scores)
        elif self.method == 'isotonic':
            # Apply isotonic regression
            return self.isotonic.transform(scores)
        elif self.method == 'beta':
            # Apply beta calibration
            return self._apply_beta(scores)
        else:
            return scores
            
    def _apply_platt(self, scores):
        """Apply fitted Platt scaling"""
        logits = self.platt_a * scores + self.platt_b
        return 1 / (1 + np.exp(-logits))
        
    def _apply_beta(self, scores):
        """Apply fitted beta calibration"""
        if self.beta_params is None:
            return scores
            
        # Extract parameters
        a, b, c = self.beta_params
        
        # Apply beta calibration transform
        eps = 1e-12
        scores = np.clip(scores, eps, 1-eps)
        logit = lambda p: np.log(p / (1 - p))
        return 1 / (1 + np.exp(-(a * logit(scores) + b) + c))
        
    def visualize_calibration(self, scores, labels, bins=10):
        """
        Visualize calibration curves before and after calibration
        
        Args:
            scores: Raw prediction scores
            labels: Ground truth labels
            bins: Number of bins for visualization
            
        Returns:
            fig: Matplotlib figure with calibration curves
        """
        # Create a figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Calculate calibration curves
        bin_edges = np.linspace(0, 1, bins + 1)
        bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        
        # Original scores
        bin_indices = np.digitize(scores, bin_edges) - 1
        bin_indices = np.clip(bin_indices, 0, bins - 1)
        
        # Calculate fraction of positives in each bin
        bin_positives = np.zeros(bins)
        bin_counts = np.zeros(bins)
        
        for i, label in enumerate(labels):
            bin_idx = bin_indices[i]
            bin_counts[bin_idx] += 1
            if label == 1:
                bin_positives[bin_idx] += 1
                
        bin_fractions = np.zeros(bins)
        for i in range(bins):
            if bin_counts[i] > 0:
                bin_fractions[i] = bin_positives[i] / bin_counts[i]
                
        # Plot original calibration
        ax1.plot(bin_centers, bin_fractions, 'o-', label='Original')
        ax1.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        ax1.set_xlabel('Predicted probability')
        ax1.set_ylabel('Fraction of positives')
        ax1.set_title('Calibration Curve - Original')
        ax1.legend(loc='best')
        ax1.grid(True)
        
        # Calibrated scores
        if self.calibrated:
            calibrated_scores = self.calibrate(scores)
            
            bin_indices = np.digitize(calibrated_scores, bin_edges) - 1
            bin_indices = np.clip(bin_indices, 0, bins - 1)
            
            # Calculate fraction of positives in each bin
            bin_positives = np.zeros(bins)
            bin_counts = np.zeros(bins)
            
            for i, label in enumerate(labels):
                bin_idx = bin_indices[i]
                bin_counts[bin_idx] += 1
                if label == 1:
                    bin_positives[bin_idx] += 1
                    
            bin_fractions = np.zeros(bins)
            for i in range(bins):
                if bin_counts[i] > 0:
                    bin_fractions[i] = bin_positives[i] / bin_counts[i]
                    
            # Plot calibrated scores
            ax2.plot(bin_centers, bin_fractions, 'o-', label=f'Calibrated ({self.method})')
            ax2.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
            ax2.set_xlabel('Predicted probability')
            ax2.set_ylabel('Fraction of positives')
            ax2.set_title('Calibration Curve - Calibrated')
            ax2.legend(loc='best')
            ax2.grid(True)
            
        plt.tight_layout()
        return fig


class DynamicThresholdSelector:
    """Selects optimal decision threshold based on operating conditions"""
    
    def __init__(self):
        """Initialize threshold selector"""
        self.precision_targets = None
        self.recall_targets = None
        self.f_score_targets = None
        
    def compute_threshold_metrics(self, scores, labels):
        """
        Compute metrics for different threshold values
        
        Args:
            scores: Prediction scores (calibrated probabilities)
            labels: Ground truth labels
            
        Returns:
            thresholds: Array of threshold values
            precisions: Array of precision values
            recalls: Array of recall values
            f_scores: Array of F1 score values
        """
        # Calculate precision-recall curve
        precisions, recalls, thresholds = precision_recall_curve(labels, scores)
        
        # Calculate F1 scores
        f_scores = 2 * precisions * recalls / (precisions + recalls + 1e-10)
        
        # Store results
        self.precision_targets = {}
        self.recall_targets = {}
        self.f_score_targets = {}
        
        # Map common target values to thresholds
        for target in [0.7, 0.8, 0.9, 0.95]:
            # Find threshold for target precision
            idx = np.argmin(np.abs(precisions - target))
            if idx < len(thresholds):
                self.precision_targets[target] = thresholds[idx]
                
            # Find threshold for target recall
            idx = np.argmin(np.abs(recalls - target))
            if idx < len(thresholds):
                self.recall_targets[target] = thresholds[idx]
                
            # Find threshold for target F1 score
            idx = np.argmin(np.abs(f_scores - target))
            if idx < len(thresholds):
                self.f_score_targets[target] = thresholds[idx]
                
        # Store optimal F1 threshold
        idx = np.argmax(f_scores)
        if idx < len(thresholds):
            self.f_optimal = thresholds[idx]
        else:
            self.f_optimal = 0.5
            
        return thresholds, precisions, recalls, f_scores
        
    def get_threshold(self, target_type='f1', target_value=None):
        """
        Get threshold for specific operating target
        
        Args:
            target_type: Type of target ('precision', 'recall', or 'f1')
            target_value: Target value (if None, returns optimal F1 threshold)
            
        Returns:
            threshold: Selected threshold value
        """
        if target_value is None:
            return self.f_optimal
            
        if target_type == 'precision':
            if self.precision_targets is not None and target_value in self.precision_targets:
                return self.precision_targets[target_value]
        elif target_type == 'recall':
            if self.recall_targets is not None and target_value in self.recall_targets:
                return self.recall_targets[target_value]
        elif target_type == 'f1':
            if self.f_score_targets is not None and target_value in self.f_score_targets:
                return self.f_score_targets[target_value]
                
        # Default to optimal F1 if target not found
        return self.f_optimal
        
    def visualize_threshold_metrics(self, thresholds, precisions, recalls, f_scores):
        """
        Visualize metrics for different threshold values
        
        Args:
            thresholds: Array of threshold values
            precisions: Array of precision values
            recalls: Array of recall values
            f_scores: Array of F1 score values
            
        Returns:
            fig: Matplotlib figure with threshold metrics
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot metrics
        ax.plot(thresholds, precisions[:-1], 'b-', label='Precision')
        ax.plot(thresholds, recalls[:-1], 'r-', label='Recall')
        ax.plot(thresholds, f_scores[:-1], 'g-', label='F1 Score')
        
        # Mark optimal F1 threshold
        optimal_idx = np.argmax(f_scores[:-1])
        optimal_threshold = thresholds[optimal_idx]
        optimal_f1 = f_scores[:-1][optimal_idx]
        
        ax.axvline(optimal_threshold, color='k', linestyle='--', label=f'Optimal F1 threshold: {optimal_threshold:.3f}')
        ax.plot(optimal_threshold, optimal_f1, 'ko', markersize=8)
        
        # Add annotations for specific target thresholds
        for target_value, threshold in self.precision_targets.items():
            ax.axvline(threshold, color='b', linestyle=':', alpha=0.5)
            ax.text(threshold, 0.2, f'P={target_value}', color='b', rotation=90, alpha=0.7)
            
        for target_value, threshold in self.recall_targets.items():
            ax.axvline(threshold, color='r', linestyle=':', alpha=0.5)
            ax.text(threshold, 0.4, f'R={target_value}', color='r', rotation=90, alpha=0.7)
            
        # Set labels and title
        ax.set_xlabel('Decision Threshold')
        ax.set_ylabel('Metric Value')
        ax.set_title('Precision, Recall, and F1 Score vs. Decision Threshold')
        ax.legend(loc='best')
        ax.grid(True)
        
        return fig


class DecisionExplainer:
    """Generates explanations for forgery detection decisions"""
    
    def generate_explanation(self, confidence_score, evidence_metrics, frequency_metrics=None, region_results=None):
        """
        Generate a human-readable explanation for the forgery detection decision
        
        Args:
            confidence_score: Final confidence score
            evidence_metrics: Dictionary of evidence metrics
            frequency_metrics: Optional dictionary of frequency domain metrics
            region_results: Optional dictionary of region-based results
            
        Returns:
            explanation: Dictionary with textual explanation and key evidence factors
        """
        explanation_text = []
        evidence_factors = []
        
        # Determine decision and confidence level
        if confidence_score < 0.2:
            decision = "The image is likely authentic"
            confidence_level = "very high"
        elif confidence_score < 0.4:
            decision = "The image is probably authentic"
            confidence_level = "moderate"
        elif confidence_score < 0.6:
            decision = "The image might contain a copy-move forgery"
            confidence_level = "moderate"
        elif confidence_score < 0.8:
            decision = "The image probably contains a copy-move forgery"
            confidence_level = "high"
        else:
            decision = "The image almost certainly contains a copy-move forgery"
            confidence_level = "very high"
            
        explanation_text.append(f"{decision} ({confidence_level} confidence).")
        
        # Add evidence from metrics
        if evidence_metrics:
            suspicious_pairs = evidence_metrics.get('suspicious_pairs', 0)
            coherence = evidence_metrics.get('coherence', 0)
            pattern_strength = evidence_metrics.get('pattern_strength', 0)
            
            # Describe suspicious matching regions
            if suspicious_pairs > 0:
                explanation_text.append(f"Found {suspicious_pairs} suspicious matching regions in the image.")
                if coherence > 0.7:
                    explanation_text.append("These regions show highly consistent spatial relationships, a strong indicator of copy-move forgery.")
                    evidence_factors.append(("High spatial coherence", coherence))
                elif coherence > 0.5:
                    explanation_text.append("These regions show somewhat consistent spatial relationships.")
                    evidence_factors.append(("Moderate spatial coherence", coherence))
                
                # Describe pattern strength
                if pattern_strength > 0.7:
                    explanation_text.append("The spatial pattern of matching regions is very characteristic of copy-move manipulation.")
                    evidence_factors.append(("Strong pattern indicators", pattern_strength))
                elif pattern_strength > 0.5:
                    explanation_text.append("The spatial pattern of matching regions shows some characteristics of copy-move manipulation.")
                    evidence_factors.append(("Moderate pattern indicators", pattern_strength))
        
        # Add frequency domain evidence
        if frequency_metrics:
            dct_similarity = frequency_metrics.get('dct_similarity', 0)
            has_compression_artifacts = frequency_metrics.get('has_compression_artifacts', False)
            inconsistent_regions = frequency_metrics.get('inconsistent_regions', [])
            
            if dct_similarity > 0.8:
                explanation_text.append("Frequency analysis shows very high similarity between suspicious regions.")
                evidence_factors.append(("High frequency similarity", dct_similarity))
            elif dct_similarity > 0.6:
                explanation_text.append("Frequency analysis shows moderate similarity between suspicious regions.")
                evidence_factors.append(("Moderate frequency similarity", dct_similarity))
                
            if has_compression_artifacts:
                explanation_text.append("Detected JPEG compression inconsistencies, which often indicate image manipulation.")
                evidence_factors.append(("Compression artifacts", 1.0))
                
            if inconsistent_regions and len(inconsistent_regions) > 0:
                explanation_text.append(f"Found {len(inconsistent_regions)} regions with inconsistent noise patterns.")
                evidence_factors.append(("Noise inconsistencies", len(inconsistent_regions)))
                
        # Add region-based evidence
        if region_results:
            forgery_regions = region_results.get('forgery_regions', [])
            
            if forgery_regions:
                explanation_text.append(f"Identified {len(forgery_regions)} specific image regions with strong indications of copy-move forgery.")
                evidence_factors.append(("Region-specific evidence", len(forgery_regions)))
                
        # Add uncertainty information if available
        if 'uncertainty' in evidence_metrics:
            uncertainty = evidence_metrics.get('uncertainty', 0)
            if uncertainty > 0.3:
                explanation_text.append(f"There is high uncertainty in this assessment (±{uncertainty:.2f}).")
            elif uncertainty > 0.15:
                explanation_text.append(f"There is moderate uncertainty in this assessment (±{uncertainty:.2f}).")
            else:
                explanation_text.append(f"This assessment has low uncertainty (±{uncertainty:.2f}).")
            
        return {
            'summary': decision,
            'confidence_level': confidence_level,
            'confidence_score': confidence_score,
            'explanation': ' '.join(explanation_text),
            'evidence_factors': evidence_factors
        }


# Add the missing functions required by infer.py
def check_if_authentic(image_metrics, texture_threshold=600, contrast_threshold=50):
    """
    Check if an image is likely to be authentic based on statistical metrics.
    
    Args:
        image_metrics: Dictionary containing image metrics like texture, contrast, etc.
        texture_threshold: Threshold for texture complexity
        contrast_threshold: Threshold for global image contrast
        
    Returns:
        is_authentic: Boolean indicating if image is likely authentic
        natural_patterns: Boolean indicating if image has natural patterns
        decision_threshold: Recommended decision threshold
    """
    # Extract metrics
    texture = image_metrics.get('texture', 0)
    contrast = image_metrics.get('contrast', 0)
    high_freq_energy = image_metrics.get('high_freq_energy', 0)
    
    # Determine if image has natural patterns (complex textures)
    has_natural_patterns = texture > texture_threshold
    
    # For images with high texture complexity, we need a higher threshold
    # to avoid false positives in natural patterns
    if has_natural_patterns:
        decision_threshold = 0.65  # Higher threshold for complex textures
    else:
        decision_threshold = 0.40  # Lower threshold for simpler images
    
    # Adjust threshold based on other factors
    if contrast > contrast_threshold:
        decision_threshold -= 0.05  # Lower threshold for high-contrast images
    
    # Determine authenticity (this is just a placeholder - actual implementation
    # would use more sophisticated analysis)
    is_authentic = True  # By default assume authentic
    
    return is_authentic, has_natural_patterns, decision_threshold


def select_optimal_threshold(image_metrics, suspicious_pairs_ratio):
    """
    Select optimal decision threshold based on image characteristics
    
    Args:
        image_metrics: Dictionary with image analysis metrics
        suspicious_pairs_ratio: Ratio of suspicious pairs to total pairs
        
    Returns:
        threshold: Optimal decision threshold
        threshold_info: Information about threshold selection
    """
    # Extract relevant metrics
    texture = image_metrics.get('texture', 0)
    contrast = image_metrics.get('contrast', 0)
    high_freq = image_metrics.get('high_freq_energy', 0)
    edge_pct = image_metrics.get('edge_percentage', 0)
    
    # Base threshold
    threshold = 0.5
    
    # Adjust based on image content
    if texture > 800:  # Very textured images need higher threshold
        threshold += 0.15
    elif texture > 400:  # Moderately textured
        threshold += 0.05
        
    if edge_pct > 0.2:  # Images with many edges may have more false matches
        threshold += 0.1
    
    # Adjust based on suspicious pairs ratio
    if suspicious_pairs_ratio > 0.01:  # Many suspicious pairs
        threshold -= 0.05
    elif suspicious_pairs_ratio < 0.001:  # Few suspicious pairs
        threshold += 0.05
        
    # Cap threshold to reasonable range
    threshold = max(0.35, min(0.75, threshold))
    
    # Prepare explanation
    threshold_info = {
        'base': 0.5,
        'texture_adjustment': threshold - 0.5,
        'final': threshold,
        'explanation': f"Using decision threshold: {threshold:.4f}"
    }
    
    return threshold, threshold_info


def analyze_image_metrics(image):
    """
    Analyze image to extract key metrics for threshold calibration
    
    Args:
        image: Input image (BGR format)
        
    Returns:
        metrics: Dictionary of image metrics
    """
    # Convert to grayscale for analysis
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()
    
    h, w = gray.shape[:2]
    
    # Calculate high frequency energy using Laplacian
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    high_freq_energy = np.mean(np.abs(laplacian))
    
    # Calculate edge percentage
    edges = cv2.Canny(gray, 100, 200)
    edge_percentage = np.sum(edges > 0) / (h * w)
    
    # Calculate texture measure using gradient
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    texture_measure = np.mean(np.sqrt(sobelx**2 + sobely**2))
    
    # Calculate contrast
    contrast = np.std(gray)
    
    # Calculate noise level estimate
    noise_sigma = estimate_noise(gray)
    
    # Return metrics dictionary
    metrics = {
        'texture': texture_measure,
        'contrast': contrast,
        'high_freq_energy': high_freq_energy,
        'edge_percentage': edge_percentage,
        'noise_level': noise_sigma
    }
    
    return metrics


def estimate_noise(image):
    """
    Estimate noise level in an image using median filter method
    
    Args:
        image: Input grayscale image
        
    Returns:
        noise_sigma: Estimated noise standard deviation
    """
    # Apply median filter
    denoised = cv2.medianBlur(image, 5)
    
    # Calculate difference
    diff = image.astype(float) - denoised.astype(float)
    
    # Estimate noise as std of difference
    noise_sigma = np.std(diff)
    
    return noise_sigma