"""
Frequency Domain Analysis Module for Copy-Move Forgery Detection.
This module analyzes potential forgery regions in the frequency domain 
to find inconsistencies that may not be visible in the spatial domain.
"""
import os
import numpy as np
import cv2
from scipy import fftpack
import matplotlib.pyplot as plt
from io import BytesIO
from scipy.fftpack import dct, idct
from scipy.ndimage import zoom
from scipy.signal import find_peaks, convolve2d
from utils.region_analysis import compute_correlation

class FrequencyAnalyzer:
    """
    Analyzes image regions in the frequency domain for forgery detection.
    
    This class provides methods to analyze suspicious region pairs using
    frequency domain techniques like DCT (Discrete Cosine Transform) and
    noise pattern analysis to identify evidence of copy-move forgeries.
    """
    
    def __init__(
        self, 
        dct_threshold=0.02,
        noise_threshold=0.6,
        mid_freq_focus=(1, 32),
        histogram_bins=50,
        noise_sampling_size=1000
    ):
        """
        Initialize frequency analyzer with configurable parameters.
        
        Args:
            dct_threshold: Threshold for DCT coefficient filtering
            noise_threshold: Threshold for noise pattern similarity (0-1)
            mid_freq_focus: Range of mid-frequency DCT coefficients to analyze (start, end)
            histogram_bins: Number of bins for noise histogram
            noise_sampling_size: Number of samples to use for noise correlation
        """
        self.dct_threshold = dct_threshold
        self.noise_threshold = noise_threshold
        self.mid_freq_focus = mid_freq_focus
        self.histogram_bins = histogram_bins
        self.noise_sampling_size = noise_sampling_size
        self.histogram_range = (-50, 50)  # Range for noise histogram
    
    def analyze_regions(self, image, suspicious_pairs, patch_size=64):
        """
        Analyze suspicious region pairs in the frequency domain
        
        Args:
            image: Input image (BGR or grayscale)
            suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
            patch_size: Size of patches to analyze
            
        Returns:
            Dictionary with frequency analysis results:
            - frequency_similarity: Similarity scores based on DCT coefficients
            - noise_pattern_similarity: Similarity scores based on noise residuals
            - double_compression: Evidence scores for double compression artifacts
            - visualization: Visualization of frequency analysis
        """
        if len(suspicious_pairs) == 0:
            return {
                'frequency_similarity': [],
                'noise_pattern_similarity': [],
                'double_compression': 0.0,
                'visualization': None
            }
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        # Initialize result arrays
        freq_similarity = []
        noise_similarity = []
        
        # Process each suspicious pair
        for (x1, y1), (x2, y2) in suspicious_pairs:
            # Extract source and destination patches
            src_patch = gray_image[y1:y1+patch_size, x1:x1+patch_size]
            dst_patch = gray_image[y2:y2+patch_size, x2:x2+patch_size]
            
            # Skip if patches are outside image bounds
            if src_patch.shape != (patch_size, patch_size) or dst_patch.shape != (patch_size, patch_size):
                freq_similarity.append(0.0)
                noise_similarity.append(0.0)
                continue
                
            # Analyze frequency domain
            freq_sim = self._compare_dct(src_patch, dst_patch)
            freq_similarity.append(freq_sim)
            
            # Analyze noise patterns
            noise_sim = self._compare_noise_residuals(src_patch, dst_patch)
            noise_similarity.append(noise_sim)
        
        # Check for double JPEG compression artifacts
        double_compression = self._detect_double_compression(gray_image)
        
        # Create visualization
        visualization = self._create_visualization(image, suspicious_pairs, 
                                              freq_similarity, noise_similarity, patch_size)
        
        return {
            'frequency_similarity': freq_similarity,
            'noise_pattern_similarity': noise_similarity,
            'double_compression': double_compression,
            'visualization': visualization
        }
    
    def _compare_dct(self, patch1, patch2):
        """
        Compare patches using Discrete Cosine Transform (DCT)
        
        Args:
            patch1, patch2: Grayscale image patches to compare
            
        Returns:
            Similarity score (0-1) based on DCT coefficients
        """
        # Apply DCT transform
        dct1 = fftpack.dct(fftpack.dct(patch1, axis=0), axis=1)
        dct2 = fftpack.dct(fftpack.dct(patch2, axis=0), axis=1)
        
        # Focus on mid-frequency coefficients (exclude DC and highest frequencies)
        # which are most useful for forgery detection
        dct1_mid = dct1[1:32, 1:32]
        dct2_mid = dct2[1:32, 1:32]
        
        # Calculate correlation between DCT coefficients
        dct1_flat = dct1_mid.flatten()
        dct2_flat = dct2_mid.flatten()
        
        # Normalize DCT coefficients
        dct1_norm = (dct1_flat - np.mean(dct1_flat)) / (np.std(dct1_flat) + 1e-10)
        dct2_norm = (dct2_flat - np.mean(dct2_flat)) / (np.std(dct2_flat) + 1e-10)
        
        # Calculate correlation coefficient
        corr = compute_correlation(dct1_norm, dct2_norm)
        
        # Handle NaN values
        if np.isnan(corr):
            corr = 0.0
            
        # Convert correlation to similarity score (0-1)
        similarity = 0.5 * (corr + 1.0)
        
        return similarity
    
    def _compare_noise_residuals(self, patch1, patch2):
        """
        Compare noise residuals between patches
        
        Args:
            patch1, patch2: Grayscale image patches to compare
            
        Returns:
            Similarity score (0-1) based on noise patterns
        """
        # Extract noise residuals using median filtering
        smooth1 = cv2.medianBlur(patch1, 3)
        smooth2 = cv2.medianBlur(patch2, 3)
        
        noise1 = patch1 - smooth1
        noise2 = patch2 - smooth2
        
        # Calculate statistical measures of noise
        noise1_mean = np.mean(noise1)
        noise1_std = np.std(noise1)
        noise2_mean = np.mean(noise2)
        noise2_std = np.std(noise2)
        
        # Calculate histograms of noise
        hist1, _ = np.histogram(noise1, bins=50, range=(-50, 50), density=True)
        hist2, _ = np.histogram(noise2, bins=50, range=(-50, 50), density=True)
        
        # Calculate correlation between histograms
        corr_hist = compute_correlation(hist1, hist2)
        if np.isnan(corr_hist):
            corr_hist = 0.0
        
        # Calculate correlation between noise patterns
        flat_noise1 = noise1.flatten()
        flat_noise2 = noise2.flatten()
        
        # Ensure we have enough samples
        min_samples = min(len(flat_noise1), len(flat_noise2))
        if min_samples >= 100:
            # Randomly sample noise pixels for correlation analysis
            indices = np.random.choice(min_samples, 100, replace=False)
            samples1 = flat_noise1[indices]
            samples2 = flat_noise2[indices]
            
            corr_patterns = compute_correlation(samples1, samples2)
            if np.isnan(corr_patterns):
                corr_patterns = 0.0
        else:
            corr_patterns = 0.0
        
        # Calculate noise statistics similarity
        mean_diff = abs(noise1_mean - noise2_mean) / max(abs(noise1_mean), abs(noise2_mean), 1e-5)
        std_diff = abs(noise1_std - noise2_std) / max(noise1_std, noise2_std, 1e-5)
        
        stat_sim = 1.0 - 0.5 * (mean_diff + std_diff)
        stat_sim = max(0.0, min(1.0, stat_sim))
        
        # Combine metrics into overall similarity score
        # Weight histogram correlation more heavily as it's more reliable
        similarity = 0.4 * corr_hist + 0.3 * corr_patterns + 0.3 * stat_sim
        
        # Ensure score is in valid range
        similarity = max(0.0, min(1.0, similarity))
        
        return similarity
    
    def _detect_double_compression(self, image):
        """
        Detect signs of double JPEG compression
        
        Args:
            image: Grayscale image to analyze
            
        Returns:
            Score (0-1) indicating likelihood of double compression
        """
        h, w = image.shape
        
        # Skip if image is too small
        if h < 128 or w < 128:
            return 0.0
            
        # We'll analyze the image in blocks
        block_size = min(256, min(h, w) // 2)
        
        # Select central region for analysis
        center_y, center_x = h // 2, w // 2
        roi = image[center_y - block_size//2:center_y + block_size//2, 
                   center_x - block_size//2:center_x + block_size//2]
        
        # Apply DCT to 8x8 blocks (JPEG block size)
        dct_coeffs = []
        for y in range(0, block_size - 8, 8):
            for x in range(0, block_size - 8, 8):
                block = roi[y:y+8, x:x+8].astype(float)
                dct_block = fftpack.dct(fftpack.dct(block, axis=0), axis=1)
                dct_coeffs.extend(dct_block[1:8, 1:8].flatten())  # Skip DC component
        
        # Convert to numpy array
        dct_coeffs = np.array(dct_coeffs)
        
        # Check for characteristic artifacts in DCT histogram
        # These appear as periodic peaks in histogram of certain DCT coefficients
        hist, bins = np.histogram(dct_coeffs, bins=100, range=(-50, 50))
        hist = hist / np.sum(hist)  # Normalize histogram
        
        # Calculate periodicity using autocorrelation
        autocorr = np.correlate(hist, hist, mode='full')
        autocorr = autocorr[len(hist)-1:]
        autocorr = autocorr / autocorr[0]  # Normalize
        
        # Look for periodic peaks in autocorrelation (characteristic of double compression)
        # Skip first few lags
        peaks = 0
        threshold = 0.2
        for i in range(5, min(50, len(autocorr))):
            if autocorr[i] > threshold and autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                peaks += 1
        
        # Calculate score based on number of peaks and their strength
        # More peaks = higher likelihood of double compression
        max_peaks = 5  # Normalize by expected maximum number of peaks
        score = min(1.0, peaks / max_peaks)
        
        return score
        
    def analyze_frequency_coherence(self, image, suspicious_pairs, patch_size=64):
        """
        Analyze frequency coherence between suspicious regions
        
        Args:
            image: Input image (BGR or grayscale)
            suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
            patch_size: Size of patches to analyze
            
        Returns:
            coherence_score: Score (0-1) indicating frequency domain coherence
        """
        if len(suspicious_pairs) == 0:
            return 0.0
            
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        # Extract DCT coefficients for all patches
        src_dcts = []
        dst_dcts = []
        
        for (x1, y1), (x2, y2) in suspicious_pairs:
            # Extract source and destination patches
            src_patch = gray_image[y1:y1+patch_size, x1:x1+patch_size]
            dst_patch = gray_image[y2:y2+patch_size, x2:x2+patch_size]
            
            # Skip if patches are outside image bounds
            if src_patch.shape != (patch_size, patch_size) or dst_patch.shape != (patch_size, patch_size):
                continue
                
            # Apply DCT transform and keep significant coefficients
            src_dct = fftpack.dct(fftpack.dct(src_patch, axis=0), axis=1)
            dst_dct = fftpack.dct(fftpack.dct(dst_patch, axis=0), axis=1)
            
            # Store only mid-frequency coefficients (most relevant for forgery detection)
            src_dcts.append(src_dct[1:32, 1:32].flatten())
            dst_dcts.append(dst_dct[1:32, 1:32].flatten())
            
        # If no valid patches, return zero coherence
        if len(src_dcts) == 0:
            return 0.0
            
        # Convert to numpy arrays
        src_dcts = np.array(src_dcts)
        dst_dcts = np.array(dst_dcts)
        
        # Calculate average DCT coefficients
        avg_src_dct = np.mean(src_dcts, axis=0)
        avg_dst_dct = np.mean(dst_dcts, axis=0)
        
        # Calculate correlation between average DCT coefficients
        corr = compute_correlation(avg_src_dct, avg_dst_dct)
        if np.isnan(corr):
            corr = 0.0
            
        # Calculate variance of coefficients within source and destination groups
        src_var = np.mean(np.var(src_dcts, axis=0))
        dst_var = np.mean(np.var(dst_dcts, axis=0))
        
        # Calculate ratio of within-group variance to between-group variance
        between_var = np.mean((avg_src_dct - avg_dst_dct) ** 2)
        within_var = 0.5 * (src_var + dst_var)
        
        # Avoid division by zero
        if between_var < 1e-10:
            ratio = 1000.0  # Large ratio means very similar (low between-group variance)
        else:
            ratio = within_var / between_var
            
        # Cap the ratio at a reasonable maximum
        ratio = min(ratio, 1000.0)
        
        # Calculate coherence score
        # High correlation and low between-group variance (high ratio) indicate coherence
        coherence_score = 0.5 * (corr + 1.0) * (1.0 - 1.0 / (1.0 + np.log10(1.0 + ratio)))
        
        # Ensure score is in valid range
        coherence_score = max(0.0, min(1.0, coherence_score))
        
        return coherence_score
    
    def analyze_frequency_bands(self, image, pairs, patch_size):
        """Analyze frequency band similarities between patch pairs"""
        if not pairs:
            return {"band_similarities": []}
        
        # Initialize results
        band_similarities = []
        
        for (p1, p2) in pairs:
            try:
                x1, y1 = p1
                x2, y2 = p2
                
                # Check if coordinates are valid
                img_h, img_w = image.shape[:2]
                if (x1 < 0 or y1 < 0 or x2 < 0 or y2 < 0 or
                    x1 + patch_size > img_w or y1 + patch_size > img_h or
                    x2 + patch_size > img_w or y2 + patch_size > img_h):
                    band_similarities.append(0.0)
                    continue
                
                # Extract patches
                patch1 = image[y1:y1+patch_size, x1:x1+patch_size].copy()
                patch2 = image[y2:y2+patch_size, x2:x2+patch_size].copy()
                
                # Ensure patches have valid dimensions
                if patch1.shape[0] <= 0 or patch1.shape[1] <= 0 or patch2.shape[0] <= 0 or patch2.shape[1] <= 0:
                    band_similarities.append(0.0)
                    continue
                
                # Convert patches to grayscale
                if len(patch1.shape) > 2:
                    patch1_gray = cv2.cvtColor(patch1, cv2.COLOR_BGR2GRAY)
                    patch2_gray = cv2.cvtColor(patch2, cv2.COLOR_BGR2GRAY)
                else:
                    patch1_gray = patch1.copy()
                    patch2_gray = patch2.copy()
                
                # Apply DCT transform
                dct1 = dct(dct(patch1_gray.astype(float), axis=0), axis=1)
                dct2 = dct(dct(patch2_gray.astype(float), axis=0), axis=1)
                
                # Define frequency bands
                bands = [
                    (0, 8),    # Low frequencies
                    (8, 16),   # Mid-low
                    (16, 32),  # Mid
                    (32, 48),  # Mid-high
                    (48, 64)   # High frequencies
                ]
                
                # Limit bands to patch size
                bands = [(start, min(end, patch_size)) for start, end in bands if start < patch_size]
                
                # Analyze band similarities
                band_similarity = []
                for start, end in bands:
                    if end <= start:
                        continue
                    
                    # Extract band coefficients
                    band1 = dct1[start:end, start:end].flatten()
                    band2 = dct2[start:end, start:end].flatten()
                    
                    # Use safe correlation method
                    similarity = safe_cross_correlation(band1, band2)
                    
                    # Convert to similarity range (0-1)
                    similarity = (similarity + 1) / 2
                    band_similarity.append(similarity)
                
                # Compute weighted average (low frequencies are more important)
                if band_similarity:
                    weights = np.array([0.5, 0.3, 0.15, 0.05, 0.0])[:len(band_similarity)]
                    weights = weights / np.sum(weights)
                    weighted_avg = np.sum(np.array(band_similarity) * weights)
                    band_similarities.append(weighted_avg)
                else:
                    band_similarities.append(0.0)
                    
            except Exception as e:
                print(f"Error in frequency band analysis: {e}")
                band_similarities.append(0.0)
        
        return {"band_similarities": band_similarities}
    
    def _create_band_visualization(self, image, suspicious_pairs, band_similarities, bands, patch_size):
        """Create visualization showing frequency band analysis results"""
        # Create a copy of the image for visualization
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Get unique band names
        band_names = [band['name'] for band in bands]
        
        # Color mapping for different bands (BGR format)
        band_colors = {
            'low': (0, 0, 255),      # Red for low frequency
            'mid-low': (0, 128, 255), # Orange for mid-low
            'mid': (0, 255, 255),    # Yellow for mid
            'mid-high': (0, 255, 0), # Green for mid-high
            'high': (255, 0, 0)      # Blue for high frequency
        }
        
        # Create a multi-band visualization
        for i, ((x1, y1), (x2, y2)) in enumerate(suspicious_pairs):
            # Skip if index out of range
            if i >= len(band_similarities):
                continue
                
            # Get band similarities for this pair
            pair_similarities = band_similarities[i]
            
            # Find the band with highest similarity
            max_band = max(band_names, key=lambda b: pair_similarities.get(b, 0))
            max_sim = pair_similarities.get(max_band, 0)
            
            # Use the color of the most similar band
            color = band_colors.get(max_band, (255, 255, 255))  # Default to white if band not found
            
            # Draw rectangles with band-specific color
            thickness = max(1, min(3, int(max_sim * 5)))  # Thickness based on similarity
            
            cv2.rectangle(vis_image, (x1, y1), (x1 + patch_size, y1 + patch_size), color, thickness)
            cv2.rectangle(vis_image, (x2, y2), (x2 + patch_size, y2 + patch_size), color, thickness)
            
            # Draw connecting line
            cv2.line(vis_image, (x1 + patch_size//2, y1 + patch_size//2),
                   (x2 + patch_size//2, y2 + patch_size//2), color, 1)
            
            # Add text showing the most influential band and its similarity
            label = f"{max_band}: {max_sim:.2f}"
            cv2.putText(vis_image, label, (x1, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
                       
            # Add small squares with similarity values for each band
            square_size = 10
            for j, band_name in enumerate(band_names):
                sim = pair_similarities.get(band_name, 0)
                band_color = band_colors.get(band_name, (255, 255, 255))
                
                # Fill small square based on similarity
                intensity = int(255 * sim)
                cv2.rectangle(vis_image,
                           (x1 + j*square_size*2, y1 - 20),
                           (x1 + (j+1)*square_size*2, y1 - 20 + square_size),
                           band_color,
                           -1)  # Filled rectangle
                           
                # Add mini-label
                cv2.putText(vis_image, band_name[0],  # First letter of band name
                          (x1 + j*square_size*2 + 2, y1 - 12),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
        
        return vis_image
    
    def _create_visualization(self, image, suspicious_pairs, freq_similarity, noise_similarity, patch_size):
        """Create visualization of frequency domain analysis"""
        # Create a copy of the image for visualization
        vis_image = image.copy() if len(image.shape) == 3 else cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Draw patches and similarities
        for i, ((x1, y1), (x2, y2)) in enumerate(suspicious_pairs):
            # Get similarity scores for this pair
            freq_sim = freq_similarity[i] if i < len(freq_similarity) else 0.0
            noise_sim = noise_similarity[i] if i < len(noise_similarity) else 0.0
            
            # Calculate combined similarity
            combined_sim = 0.5 * freq_sim + 0.5 * noise_sim
            
            # Determine color based on similarity (green=similar, red=different)
            # For frequency domain, we want to highlight similar regions
            color = (0, int(255 * combined_sim), int(255 * (1 - combined_sim)))
            
            # Draw rectangles for patches
            cv2.rectangle(vis_image, (x1, y1), (x1 + patch_size, y1 + patch_size), color, 2)
            cv2.rectangle(vis_image, (x2, y2), (x2 + patch_size, y2 + patch_size), color, 2)
            
            # Draw line connecting patches with thickness based on similarity
            thickness = max(1, int(combined_sim * 5))
            cv2.line(vis_image, 
                   (x1 + patch_size // 2, y1 + patch_size // 2), 
                   (x2 + patch_size // 2, y2 + patch_size // 2),
                   color, thickness)
            
            # Draw similarity score
            cv2.putText(vis_image, f"{combined_sim:.2f}", 
                      (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            
        return vis_image

    def analyze_block_artifacts(self, image, patch_size=64):
        """
        Analyze block artifacts that may indicate JPEG compression or manipulation
        
        Args:
            image: Input image (BGR or grayscale)
            patch_size: Size of analysis patches
            
        Returns:
            Dictionary with block artifact analysis results
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        h, w = gray_image.shape
        
        # Skip if image is too small
        if h < 64 or w < 64:
            return {
                'block_strength': 0.0,
                'block_periodicity': 0.0,
                'compression_level': 0.0,
                'visualization': None
            }
        
        # Extract center region for analysis if image is large
        if h > patch_size and w > patch_size:
            center_y, center_x = h // 2, w // 2
            gray_region = gray_image[center_y-patch_size//2:center_y+patch_size//2, 
                                   center_x-patch_size//2:center_x+patch_size//2]
        else:
            # Use the entire image if smaller than patch_size
            gray_region = gray_image
        
        # Compute horizontal and vertical gradients
        grad_x = cv2.Sobel(gray_region, cv2.CV_32F, 1, 0, ksize=1)
        grad_y = cv2.Sobel(gray_region, cv2.CV_32F, 0, 1, ksize=1)
        
        # Take absolute values
        abs_grad_x = np.abs(grad_x)
        abs_grad_y = np.abs(grad_y)
        
        # Average gradients along rows and columns
        row_sum = np.sum(abs_grad_x, axis=1)
        col_sum = np.sum(abs_grad_y, axis=0)
        
        # Compute power spectrum using FFT
        row_fft = np.abs(np.fft.fft(row_sum))
        col_fft = np.abs(np.fft.fft(col_sum))
        
        # Look for periodicity at 8 pixels (JPEG block size)
        # Block size index in FFT corresponds to (FFT size / block size)
        
        # For rows
        row_length = len(row_fft)
        block_idx_row = row_length // 8
        row_baseline = np.mean(row_fft[1:block_idx_row//2])  # Baseline - low frequencies except DC
        row_peak = row_fft[block_idx_row]  # Peak at block size frequency
        row_block_strength = row_peak / (row_baseline + 1e-10) - 1.0  # Normalized peak strength
        
        # For columns
        col_length = len(col_fft)
        block_idx_col = col_length // 8
        col_baseline = np.mean(col_fft[1:block_idx_col//2])
        col_peak = col_fft[block_idx_col]
        col_block_strength = col_peak / (col_baseline + 1e-10) - 1.0
        
        # Combine horizontal and vertical block strengths
        block_strength = max(0.0, min(1.0, (row_block_strength + col_block_strength) / 10.0))
        
        # Analyze periodicity of block artifacts
        # Higher periodicity is more indicative of JPEG compression
        
        # For rows (skip DC component)
        row_peaks = []
        for i in range(2, len(row_fft) // 2):
            if (row_fft[i] > row_fft[i-1] and row_fft[i] > row_fft[i+1] and 
                row_fft[i] > 2.0 * row_baseline):
                row_peaks.append((i, row_fft[i]))
        
        # For columns (skip DC component)
        col_peaks = []
        for i in range(2, len(col_fft) // 2):
            if (col_fft[i] > col_fft[i-1] and col_fft[i] > col_fft[i+1] and 
                col_fft[i] > 2.0 * col_baseline):
                col_peaks.append((i, col_fft[i]))
        
        # Calculate periodicity score based on number and strength of peaks
        # and their alignment with expected JPEG block frequencies (8, 16, 24...)
        periodicity_score = 0.0
        
        # Sort peaks by magnitude
        row_peaks.sort(key=lambda x: x[1], reverse=True)
        col_peaks.sort(key=lambda x: x[1], reverse=True)
        
        # Take top 5 peaks
        row_peaks = row_peaks[:5]
        col_peaks = col_peaks[:5]
        
        # Check if peaks correspond to JPEG block sizes
        for idx, _ in row_peaks:
            # If peak is at a multiple of block_idx_row
            if abs(idx % block_idx_row) < 2 or abs(idx % block_idx_row - block_idx_row) < 2:
                periodicity_score += 0.1
                
        for idx, _ in col_peaks:
            # If peak is at a multiple of block_idx_col
            if abs(idx % block_idx_col) < 2 or abs(idx % block_idx_col - block_idx_col) < 2:
                periodicity_score += 0.1
        
        periodicity_score = min(1.0, periodicity_score)
        
        # Estimate compression level based on block strength and other features
        # Higher block strength usually indicates higher compression levels
        compression_level = min(1.0, block_strength * 1.5)
        
        # Create visualization of block artifact analysis
        visualization = self._create_block_visualization(gray_image, block_strength, 
                                                     periodicity_score, compression_level)
        
        return {
            'block_strength': block_strength,
            'block_periodicity': periodicity_score,
            'compression_level': compression_level,
            'visualization': visualization
        }
        
    def _create_block_visualization(self, image, block_strength, periodicity_score, compression_level):
        """Create visualization of block artifact analysis"""
        # Create colored visualization
        vis_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
        h, w = image.shape[:2]
        
        # Draw grid lines at 8x8 block boundaries to visualize JPEG blocks
        # Intensity based on detected block strength
        intensity = int(255 * block_strength)
        color = (0, intensity, 255-intensity)  # Yellow/red based on strength
        
        # Draw horizontal and vertical grid lines
        for i in range(0, h, 8):
            cv2.line(vis_image, (0, i), (w, i), color, 1)
            
        for i in range(0, w, 8):
            cv2.line(vis_image, (i, 0), (i, h), color, 1)
            
        # Add text with analysis results
        text_y = 30
        cv2.putText(vis_image, f"Block Strength: {block_strength:.3f}", 
                  (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
        
        cv2.putText(vis_image, f"Periodicity: {periodicity_score:.3f}", 
                  (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        text_y += 25
        
        # Color-code compression level text
        comp_color = (0, 255, 0) if compression_level < 0.4 else \
                   (0, 255, 255) if compression_level < 0.7 else (0, 0, 255)
                   
        cv2.putText(vis_image, f"Est. Compression: {compression_level:.3f}", 
                  (20, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, comp_color, 2)
                  
        return vis_image
        
    def analyze_error_level(self, image, quality=90, patch_size=64):
        """
        Perform Error Level Analysis (ELA) to detect edited regions
        
        ELA works by re-saving the image at a specific quality level and
        analyzing the difference between the original and re-saved version.
        Edited regions often show different error levels.
        
        Args:
            image: Input image (BGR or RGB)
            quality: JPEG quality level for resaving (0-100)
            patch_size: Size of analysis patches
            
        Returns:
            Dictionary with ELA results
        """
        if len(image.shape) != 3:
            # Convert grayscale to RGB
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        
        # Create temporary files for JPEG compression
        original_file = "temp_original.jpg"
        resaved_file = "temp_resaved.jpg"
        
        try:
            # Save the original image
            cv2.imwrite(original_file, image, [cv2.IMWRITE_JPEG_QUALITY, 100])
            
            # Read the saved image (this ensures we're working with JPEG data)
            saved_image = cv2.imread(original_file)
            
            # Resave at the specified quality
            cv2.imwrite(resaved_file, saved_image, [cv2.IMWRITE_JPEG_QUALITY, quality])
            
            # Read the resaved image
            resaved_image = cv2.imread(resaved_file)
            
            # Calculate difference
            diff = cv2.absdiff(saved_image, resaved_image)
            
            # Enhance the difference for visualization
            # Scale each pixel value to make differences more visible
            diff_enhanced = cv2.convertScaleAbs(diff, alpha=10.0)
            
            # Convert to grayscale for analysis
            diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            
            # Analyze regions with high error levels
            h, w = diff_gray.shape
            block_size = min(patch_size, min(h, w) // 4)
            
            # Create grid for block analysis
            error_blocks = []
            
            for y in range(0, h - block_size, block_size // 2):
                for x in range(0, w - block_size, block_size // 2):
                    block = diff_gray[y:y+block_size, x:x+block_size]
                    avg_error = np.mean(block)
                    max_error = np.max(block)
                    
                    # Only consider blocks with significant error
                    if avg_error > 5.0 or max_error > 20.0:
                        error_blocks.append({
                            'position': (x, y),
                            'size': block_size,
                            'avg_error': avg_error,
                            'max_error': max_error
                        })
            
            # Sort blocks by average error
            error_blocks.sort(key=lambda b: b['avg_error'], reverse=True)
            
            # Keep only top 10% of blocks
            num_keep = max(1, len(error_blocks) // 10)
            top_error_blocks = error_blocks[:num_keep]
            
            # Calculate overall error metrics
            if len(error_blocks) > 0:
                avg_error = np.mean([b['avg_error'] for b in error_blocks])
                max_error = np.max([b['max_error'] for b in error_blocks])
                error_std = np.std([b['avg_error'] for b in error_blocks])
                
                # Normalize for scoring (0-1)
                error_score = min(1.0, avg_error / 50.0)
            else:
                avg_error = 0.0
                max_error = 0.0
                error_std = 0.0
                error_score = 0.0
                
            # Create visualization
            # First enhance the difference image for better visibility
            ela_visualization = self._create_ela_visualization(saved_image, diff_enhanced, top_error_blocks)
                
            # Check if high error blocks form clusters (potential editing)
            clusters = self._cluster_error_blocks(top_error_blocks)
            
            return {
                'error_score': error_score,
                'avg_error': avg_error,
                'max_error': max_error,
                'std_error': error_std,
                'error_blocks': top_error_blocks,
                'error_clusters': clusters,
                'diff_image': diff_enhanced,
                'visualization': ela_visualization
            }
            
        finally:
            # Clean up temporary files
            for file in [original_file, resaved_file]:
                if os.path.exists(file):
                    try:
                        os.remove(file)
                    except:
                        pass
                        
    def _create_ela_visualization(self, image, diff_image, error_blocks):
        """Create visualization for ELA results"""
        # Create a copy of the original image
        vis_image = image.copy()
        
        # Overlay the difference image with transparency
        alpha = 0.5
        vis_image = cv2.addWeighted(vis_image, 1.0, diff_image, alpha, 0)
        
        # Draw rectangles around high-error blocks
        max_error = max([b['max_error'] for b in error_blocks]) if error_blocks else 1.0
        
        for block in error_blocks:
            x, y = block['position']
            size = block['size']
            error = block['max_error']
            
            # Color based on error level (green to red)
            intensity = min(1.0, error / max_error)
            blue = 0
            green = int(255 * (1.0 - intensity))
            red = int(255 * intensity)
            
            cv2.rectangle(vis_image, (x, y), (x+size, y+size), (blue, green, red), 2)
        
        return vis_image
        
    def _cluster_error_blocks(self, error_blocks, distance_threshold=50):
        """Group error blocks into clusters based on proximity"""
        if not error_blocks:
            return []
            
        # Extract positions
        positions = np.array([[b['position'][0], b['position'][1]] for b in error_blocks])
        
        clusters = []
        remaining = set(range(len(positions)))
        
        while remaining:
            # Start a new cluster
            current = remaining.pop()
            current_pos = positions[current]
            cluster = [current]
            
            # Find all points close to this one
            to_check = list(remaining)
            for idx in to_check:
                pos = positions[idx]
                dist = np.sqrt(np.sum((current_pos - pos)**2))
                
                if dist < distance_threshold:
                    cluster.append(idx)
                    remaining.remove(idx)
                    
            # Add cluster to results
            avg_position = np.mean([positions[i] for i in cluster], axis=0)
            avg_error = np.mean([error_blocks[i]['avg_error'] for i in cluster])
            max_error = np.max([error_blocks[i]['max_error'] for i in cluster])
            
            clusters.append({
                'indices': cluster,
                'position': tuple(map(int, avg_position)),
                'size': len(cluster),
                'avg_error': avg_error,
                'max_error': max_error
            })
            
        # Sort clusters by size (largest first)
        clusters.sort(key=lambda c: c['size'], reverse=True)
        
        return clusters

    def visualize_dct_coefficients(self, image, suspicious_pairs, patch_size=64):
        """
        Create detailed visualizations of DCT coefficients for suspicious pairs
        
        This method generates visualizations showing the DCT coefficients for
        each suspicious pair, highlighting differences in the frequency domain.
        
        Args:
            image: Input image (BGR or grayscale)
            suspicious_pairs: List of ((x1, y1), (x2, y2)) coordinate pairs
            patch_size: Size of patches to analyze
            
        Returns:
            Dictionary with DCT visualization results:
            - coefficient_images: List of DCT coefficient visualizations
            - difference_images: List of difference visualizations
            - combined_visualization: Combined visualization of all pairs
        """
        if len(suspicious_pairs) == 0:
            return {
                'coefficient_images': [],
                'difference_images': [],
                'combined_visualization': None
            }
        
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
            
        # Initialize result lists
        coefficient_images = []
        difference_images = []
        
        # Create a large combined visualization
        max_pairs = min(6, len(suspicious_pairs))  # Limit to 6 pairs for combined visualization
        fig_height = max(1, (max_pairs + 1) // 2) * 6  # 2 pairs per row, 6 inches height per pair
        plt.figure(figsize=(12, fig_height))
        
        # Process each suspicious pair
        for i, ((x1, y1), (x2, y2)) in enumerate(suspicious_pairs[:max_pairs]):
            # Extract source and destination patches
            src_patch = gray_image[y1:y1+patch_size, x1:x1+patch_size]
            dst_patch = gray_image[y2:y2+patch_size, x2:x2+patch_size]
            
            # Skip if patches are outside image bounds
            if src_patch.shape != (patch_size, patch_size) or dst_patch.shape != (patch_size, patch_size):
                continue
                
            # Apply DCT transform
            dct1 = fftpack.dct(fftpack.dct(src_patch, axis=0), axis=1)
            dct2 = fftpack.dct(fftpack.dct(dst_patch, axis=0), axis=1)
            
            # Take logarithm to enhance visualization (handle zeros)
            dct1_log = np.log(np.abs(dct1) + 1)
            dct2_log = np.log(np.abs(dct2) + 1)
            
            # Calculate DCT coefficient difference
            dct_diff = np.abs(dct1_log - dct2_log)
            
            # Create individual visualizations for this pair
            # For individual coefficient images
            fig_coef, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(12, 4))
            
            # Plot first patch DCT
            im1 = ax1.imshow(dct1_log, cmap='viridis')
            ax1.set_title(f"DCT Coefficients - Source Patch")
            ax1.set_xlabel("Frequency X")
            ax1.set_ylabel("Frequency Y")
            fig_coef.colorbar(im1, ax=ax1, orientation='vertical', label='Log Magnitude')
            
            # Plot second patch DCT
            im2 = ax2.imshow(dct2_log, cmap='viridis')
            ax2.set_title(f"DCT Coefficients - Target Patch")
            ax2.set_xlabel("Frequency X")
            fig_coef.colorbar(im2, ax=ax2, orientation='vertical', label='Log Magnitude')
            
            # Plot difference
            im3 = ax3.imshow(dct_diff, cmap='inferno')
            ax3.set_title(f"Coefficient Difference")
            ax3.set_xlabel("Frequency X")
            fig_coef.colorbar(im3, ax=ax3, orientation='vertical', label='Difference')
            
            plt.tight_layout()
            # Save figure to BytesIO and convert to image
            coef_buf = BytesIO()
            fig_coef.savefig(coef_buf, format='png')
            coef_buf.seek(0)
            coef_img = cv2.imdecode(np.frombuffer(coef_buf.read(), np.uint8), 1)
            plt.close(fig_coef)
            coefficient_images.append(coef_img)
            
            # Create detailed difference visualization
            fig_diff, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
            
            # Show difference by frequency bands
            # Split frequency space into regions
            dct_diff_low = dct_diff.copy()
            dct_diff_low[patch_size//8:, :] = 0
            dct_diff_low[:, patch_size//8:] = 0
            
            dct_diff_high = dct_diff.copy()
            dct_diff_high[:patch_size//2, :] = 0
            dct_diff_high[:, :patch_size//2] = 0
            
            # Show low and high frequency differences
            ax1.imshow(dct_diff_low, cmap='hot')
            ax1.set_title("Low Frequency Differences")
            ax1.set_xlabel("Frequency X")
            ax1.set_ylabel("Frequency Y")
            
            ax2.imshow(dct_diff_high, cmap='hot')
            ax2.set_title("High Frequency Differences")
            ax2.set_xlabel("Frequency X")
            
            plt.tight_layout()
            diff_buf = BytesIO()
            fig_diff.savefig(diff_buf, format='png')
            diff_buf.seek(0)
            diff_img = cv2.imdecode(np.frombuffer(diff_buf.read(), np.uint8), 1)
            plt.close(fig_diff)
            difference_images.append(diff_img)
            
            # Add to combined visualization
            plt.subplot(max_pairs, 3, i*3 + 1)
            plt.imshow(src_patch, cmap='gray')
            plt.title(f"Pair {i+1} - Source")
            plt.axis('off')
            
            plt.subplot(max_pairs, 3, i*3 + 2)
            plt.imshow(dst_patch, cmap='gray')
            plt.title(f"Pair {i+1} - Target")
            plt.axis('off')
            
            plt.subplot(max_pairs, 3, i*3 + 3)
            plt.imshow(dct_diff, cmap='inferno')
            plt.title(f"DCT Difference")
            plt.axis('off')
        
        plt.tight_layout()
        # Save combined figure
        combined_buf = BytesIO()
        plt.savefig(combined_buf, format='png')
        combined_buf.seek(0)
        combined_img = cv2.imdecode(np.frombuffer(combined_buf.read(), np.uint8), 1)
        plt.close()
        
        return {
            'coefficient_images': coefficient_images,
            'difference_images': difference_images,
            'combined_visualization': combined_img
        }

def safe_cross_correlation(a, b):
    """
    Compute cross-correlation safely without division by zero errors.
    This function implements a safe version of numpy's correlation calculation.
    
    Args:
        a, b: Input arrays
        
    Returns:
        Correlation coefficient between -1 and 1
    """
    a = np.asarray(a).ravel()
    b = np.asarray(b).ravel()
    
    # Handle empty arrays
    if len(a) == 0 or len(b) == 0:
        return 0.0
    
    # Ensure same length
    min_len = min(len(a), len(b))
    a = a[:min_len]
    b = b[:min_len]
    
    # Remove NaN values
    mask = ~(np.isnan(a) | np.isnan(b))
    a = a[mask]
    b = b[mask]
    
    if len(a) < 2:  # Need at least 2 points for correlation
        return 0.0
    
    # Mean centering
    a_mean = np.mean(a)
    b_mean = np.mean(b)
    
    # Get numerator (covariance)
    numerator = np.sum((a - a_mean) * (b - b_mean))
    
    # Get denominators (std devs)
    a_std = np.sqrt(np.sum((a - a_mean) ** 2))
    b_std = np.sqrt(np.sum((b - b_mean) ** 2))
    
    # Check for zero standard deviations
    if a_std < 1e-10 or b_std < 1e-10:
        return 0.0
    
    # Calculate correlation
    correlation = numerator / (a_std * b_std)
    
    # Ensure result is in valid range
    if np.isnan(correlation) or np.isinf(correlation):
        return 0.0
    
    return max(-1.0, min(1.0, correlation))