"""
Multi-threshold ensemble system for copy-move forgery detection.
This module provides tools to analyze images at multiple threshold levels
simultaneously and combine results for improved detection accuracy.
"""

import numpy as np
import cv2
import torch
from collections import defaultdict
import xgboost as xgb
from sklearn.preprocessing import StandardScaler
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score, accuracy_score


class MultiThresholdEnsemble:
    """
    Multi-threshold ensemble system that combines results from multiple
    similarity thresholds to improve detection accuracy.
    """
    
    def __init__(self, thresholds=None, combiner_type='xgboost'):
        """
        Initialize multi-threshold ensemble
        
        Args:
            thresholds: List of similarity thresholds to use (default: [0.45, 0.55, 0.65])
            combiner_type: Type of ensemble combiner ('xgboost', 'mlp', 'voting')
        """
        self.thresholds = thresholds if thresholds is not None else [0.45, 0.55, 0.65]
        self.combiner_type = combiner_type
        self.combiner = None
        self.scaler = StandardScaler()
        self.is_fitted = False
        
    def create_combiner(self, input_dim):
        """
        Create ensemble combiner model
        
        Args:
            input_dim: Input feature dimension
        
        Returns:
            combiner: Model for combining results
        """
        if self.combiner_type == 'xgboost':
            # XGBoost classifier
            return xgb.XGBClassifier(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                objective='binary:logistic',
                random_state=42
            )
        elif self.combiner_type == 'mlp':
            # Simple MLP classifier
            class MLPCombiner(nn.Module):
                def __init__(self, input_dim):
                    super().__init__()
                    self.fc1 = nn.Linear(input_dim, 32)
                    self.fc2 = nn.Linear(32, 16)
                    self.fc3 = nn.Linear(16, 1)
                    self.dropout = nn.Dropout(0.3)
                    
                def forward(self, x):
                    x = F.relu(self.fc1(x))
                    x = self.dropout(x)
                    x = F.relu(self.fc2(x))
                    x = self.fc3(x)
                    return torch.sigmoid(x)
            
            return MLPCombiner(input_dim)
        elif self.combiner_type == 'voting':
            # Simple voting mechanism (no training needed)
            return None
        else:
            raise ValueError(f"Unknown combiner type: {self.combiner_type}")
            
    def fit(self, features, labels):
        """
        Fit ensemble combiner on validation data
        
        Args:
            features: List of dictionaries with detection results at different thresholds
            labels: Ground truth labels (1 for forged, 0 for authentic)
            
        Returns:
            self: Fitted ensemble
        """
        if len(features) == 0 or len(labels) == 0:
            return self
            
        # Extract features from detection results
        X = self._extract_features_from_results(features)
        y = np.array(labels)
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Create and fit combiner
        input_dim = X.shape[1]
        self.combiner = self.create_combiner(input_dim)
        
        if self.combiner_type == 'xgboost':
            self.combiner.fit(X_scaled, y)
        elif self.combiner_type == 'mlp':
            # Convert data to PyTorch tensors
            X_tensor = torch.FloatTensor(X_scaled)
            y_tensor = torch.FloatTensor(y).reshape(-1, 1)
            
            # Train the MLP
            optimizer = torch.optim.Adam(self.combiner.parameters(), lr=0.001)
            criterion = nn.BCELoss()
            
            self.combiner.train()
            for epoch in range(100):  # 100 epochs
                optimizer.zero_grad()
                outputs = self.combiner(X_tensor)
                loss = criterion(outputs, y_tensor)
                loss.backward()
                optimizer.step()
        
        self.is_fitted = True
        return self
    
    def _extract_features_from_results(self, results_list):
        """
        Extract features from detection results at different thresholds
        
        Args:
            results_list: List of dictionaries with detection results at different thresholds
            
        Returns:
            features: Array of extracted features
        """
        features = []
        
        for results_dict in results_list:
            # Initialize features for this sample
            sample_features = []
            
            # Extract features for each threshold
            for threshold in self.thresholds:
                threshold_key = str(threshold)  # Convert to string for dictionary key
                
                if threshold_key in results_dict:
                    threshold_results = results_dict[threshold_key]
                    
                    # Basic features
                    confidence = threshold_results.get('confidence_score', 0)
                    n_pairs = len(threshold_results.get('suspicious_pairs', []))
                    
                    # Evidence metrics
                    evidence = threshold_results.get('evidence_metrics', {})
                    coherence = evidence.get('coherence', 0)
                    pattern_strength = evidence.get('pattern_strength', 0)
                    uncertainty = evidence.get('uncertainty', 0)
                    
                    # Combine features
                    threshold_features = [
                        confidence,
                        min(1.0, n_pairs / 50),  # Normalize number of pairs
                        coherence,
                        pattern_strength,
                        1.0 - uncertainty  # Convert to certainty
                    ]
                else:
                    # Default values if this threshold wasn't used
                    threshold_features = [0, 0, 0, 0, 0]
                
                # Add to sample features
                sample_features.extend(threshold_features)
            
            # Add features from all thresholds together
            features.append(sample_features)
        
        return np.array(features)
    
    def predict(self, results_dict):
        """
        Predict forgery probability using the ensemble
        
        Args:
            results_dict: Dictionary with detection results at different thresholds
            
        Returns:
            probability: Forgery probability
            threshold_results: Dictionary with results at each threshold
            evidence: Combined evidence metrics
        """
        if not self.is_fitted and self.combiner_type != 'voting':
            # If not fitted, use simple voting
            return self._predict_with_voting(results_dict)
        
        # Extract features from results
        X = self._extract_features_from_results([results_dict])
        
        # Standardize features
        X_scaled = self.scaler.transform(X) if self.is_fitted else X
        
        # Predict probability
        if self.combiner_type == 'xgboost':
            probability = self.combiner.predict_proba(X_scaled)[0, 1]
        elif self.combiner_type == 'mlp':
            self.combiner.eval()
            X_tensor = torch.FloatTensor(X_scaled)
            with torch.no_grad():
                probability = self.combiner(X_tensor).item()
        else:
            # Use voting mechanism
            return self._predict_with_voting(results_dict)
        
        # Get threshold results
        threshold_results = {}
        for threshold in self.thresholds:
            threshold_key = str(threshold)
            if threshold_key in results_dict:
                threshold_results[threshold] = results_dict[threshold_key]
        
        # Combine evidence metrics
        evidence = self._combine_evidence_metrics(results_dict)
        
        return probability, threshold_results, evidence
    
    def _predict_with_voting(self, results_dict):
        """
        Predict using simple voting mechanism
        
        Args:
            results_dict: Dictionary with detection results at different thresholds
            
        Returns:
            probability: Forgery probability
            threshold_results: Dictionary with results at each threshold
            evidence: Combined evidence metrics
        """
        # Initialize votes and weights
        votes = []
        weights = []
        
        # Collect threshold results
        threshold_results = {}
        
        # Process each threshold
        for threshold in self.thresholds:
            threshold_key = str(threshold)
            if threshold_key in results_dict:
                results = results_dict[threshold_key]
                confidence = results.get('confidence_score', 0)
                is_forged = results.get('is_forged', False)
                
                # Add vote (1 for forged, 0 for authentic)
                votes.append(1 if is_forged else 0)
                
                # Weight by confidence and threshold (higher thresholds get more weight)
                # This gives more weight to stricter thresholds when they detect forgery
                weight = confidence * (0.7 + 0.1 * self.thresholds.index(threshold))
                weights.append(weight)
                
                # Store results
                threshold_results[threshold] = results
        
        # Calculate weighted probability
        if votes:
            # Convert to numpy arrays
            votes = np.array(votes)
            weights = np.array(weights)
            
            # Normalize weights
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
            
            # Calculate weighted probability
            probability = np.sum(votes * weights)
        else:
            probability = 0.5  # Default if no thresholds were used
        
        # Combine evidence metrics
        evidence = self._combine_evidence_metrics(results_dict)
        
        return probability, threshold_results, evidence
    
    def _combine_evidence_metrics(self, results_dict):
        """
        Combine evidence metrics from multiple thresholds
        
        Args:
            results_dict: Dictionary with detection results at different thresholds
            
        Returns:
            combined_evidence: Combined evidence metrics
        """
        # Initialize combined metrics
        suspicious_pairs_sets = []
        all_coherence_values = []
        all_pattern_strength_values = []
        primary_offsets = []
        uncertainties = []
        
        # Collect metrics from each threshold
        for threshold in self.thresholds:
            threshold_key = str(threshold)
            if threshold_key in results_dict:
                results = results_dict[threshold_key]
                suspicious_pairs = results.get('suspicious_pairs', [])
                evidence = results.get('evidence_metrics', {})
                
                # Collect suspicious pairs (convert to tuples for set operations)
                pairs_set = set((tuple(p[0]), tuple(p[1])) for p in suspicious_pairs)
                suspicious_pairs_sets.append(pairs_set)
                
                # Collect other metrics
                if 'coherence' in evidence:
                    all_coherence_values.append(evidence['coherence'])
                if 'pattern_strength' in evidence:
                    all_pattern_strength_values.append(evidence['pattern_strength'])
                if 'primary_offset' in evidence:
                    primary_offsets.append(evidence['primary_offset'])
                if 'uncertainty' in evidence:
                    uncertainties.append(evidence['uncertainty'])
        
        # Combine suspicious pairs (take union across thresholds)
        combined_pairs_set = set().union(*suspicious_pairs_sets) if suspicious_pairs_sets else set()
        combined_suspicious_pairs = [((p[0][0], p[0][1]), (p[1][0], p[1][1])) for p in combined_pairs_set]
        
        # Combine other metrics (weighted average, weighting stricter thresholds higher)
        combined_coherence = self._weighted_average(all_coherence_values)
        combined_pattern_strength = self._weighted_average(all_pattern_strength_values)
        
        # For primary offset, use the most common one
        if primary_offsets:
            offset_counts = defaultdict(int)
            for offset in primary_offsets:
                offset_counts[tuple(offset)] += 1
            combined_primary_offset = max(offset_counts.items(), key=lambda x: x[1])[0]
        else:
            combined_primary_offset = None
            
        # For uncertainty, take the average (lower is better)
        combined_uncertainty = np.mean(uncertainties) if uncertainties else 0.2
        
        # Determine evidence strength category
        if combined_coherence > 0.8 and combined_pattern_strength > 0.8:
            evidence_strength = "very strong"
        elif combined_coherence > 0.7 and combined_pattern_strength > 0.7:
            evidence_strength = "strong"
        elif combined_coherence > 0.6 and combined_pattern_strength > 0.6:
            evidence_strength = "moderate"
        elif combined_coherence > 0.5 and combined_pattern_strength > 0.5:
            evidence_strength = "weak"
        else:
            evidence_strength = "very weak"
        
        # Create combined evidence dictionary
        combined_evidence = {
            'suspicious_pairs': len(combined_suspicious_pairs),
            'coherence': combined_coherence,
            'pattern_strength': combined_pattern_strength,
            'primary_offset': combined_primary_offset,
            'uncertainty': combined_uncertainty,
            'evidence_strength': evidence_strength
        }
        
        return combined_evidence
    
    def _weighted_average(self, values, weights=None):
        """Calculate weighted average of values"""
        if not values:
            return 0.0
        
        if weights is None:
            # Default weights give more importance to later values (from stricter thresholds)
            weights = [0.7 + 0.1 * i for i in range(len(values))]
        
        weights = np.array(weights)
        values = np.array(values)
        
        # Normalize weights
        weights = weights / np.sum(weights)
        
        return np.sum(values * weights)


class FeatureExtractor:
    """
    Extracts features from detection results for use in the ensemble
    """
    
    @staticmethod
    def extract_features(detection_result, use_advanced_features=True):
        """
        Extract features from a single detection result
        
        Args:
            detection_result: Dictionary with detection results
            use_advanced_features: Whether to include advanced features
            
        Returns:
            features: Extracted feature vector
        """
        # Basic features
        confidence = detection_result.get('confidence_score', 0)
        suspicious_pairs = detection_result.get('suspicious_pairs', [])
        n_pairs = len(suspicious_pairs)
        
        # Evidence metrics
        evidence = detection_result.get('evidence_metrics', {})
        coherence = evidence.get('coherence', 0)
        pattern_strength = evidence.get('pattern_strength', 0)
        uncertainty = evidence.get('uncertainty', 0)
        
        # Basic feature vector
        features = [
            confidence,
            min(1.0, n_pairs / 50),  # Normalize number of pairs
            coherence,
            pattern_strength,
            1.0 - uncertainty  # Convert to certainty
        ]
        
        # Add advanced features if requested
        if use_advanced_features:
            # Analyze offset distribution
            if suspicious_pairs:
                # Calculate offset vectors
                offsets = []
                for (x1, y1), (x2, y2) in suspicious_pairs:
                    offsets.append((x2 - x1, y2 - y1))
                
                # Calculate statistics of offset vectors
                dx = np.array([x for x, y in offsets])
                dy = np.array([y for x, y in offsets])
                
                # Calculate mean and std of offsets
                mean_dx, mean_dy = np.mean(dx), np.mean(dy)
                std_dx, std_dy = np.std(dx), np.std(dy)
                
                # Calculate offset magnitudes
                magnitudes = np.sqrt(dx**2 + dy**2)
                mean_magnitude = np.mean(magnitudes)
                std_magnitude = np.std(magnitudes)
                
                # Calculate offset direction consistency
                if len(offsets) > 1:
                    angles = np.arctan2(dy, dx)
                    angle_std = np.std(angles)
                    angle_consistency = 1.0 - min(1.0, angle_std / np.pi)
                else:
                    angle_consistency = 1.0
                
                # Add offset features
                features.extend([
                    mean_magnitude / 100,  # Normalize by typical image size
                    min(1.0, std_magnitude / 50),  # Normalize
                    angle_consistency
                ])
            else:
                features.extend([0, 0, 1.0])  # Default values for no pairs
            
            # Add frequency domain features if available
            freq_results = detection_result.get('frequency_results', {})
            if freq_results:
                dct_similarity = freq_results.get('dct_similarity', 0)
                has_artifacts = float(freq_results.get('has_compression_artifacts', False))
                n_inconsistent = len(freq_results.get('inconsistent_regions', []))
                
                features.extend([
                    dct_similarity,
                    has_artifacts,
                    min(1.0, n_inconsistent / 10)  # Normalize number of inconsistent regions
                ])
            else:
                features.extend([0, 0, 0])
                
            # Add region-based features if available
            region_results = detection_result.get('region_results', {})
            if region_results:
                n_forgery_regions = len(region_results.get('forgery_regions', []))
                region_confidence = 0.0
                
                # Calculate average region confidence
                confidence_scores = region_results.get('region_confidence_scores', {})
                if confidence_scores:
                    region_confidence = sum(confidence_scores.values()) / len(confidence_scores)
                    
                features.extend([
                    min(1.0, n_forgery_regions / 5),  # Normalize number of forgery regions
                    region_confidence
                ])
            else:
                features.extend([0, 0])
        
        return features


def analyze_with_multiple_thresholds(detect_forgery_fn, image, thresholds=None, patch_size=64):
    """
    Analyze image with multiple similarity thresholds
    
    Args:
        detect_forgery_fn: Function to detect forgery at a specific threshold
        image: Input image
        thresholds: List of thresholds to use
        patch_size: Patch size for detection
        
    Returns:
        results_dict: Dictionary with results for each threshold
    """
    if thresholds is None:
        thresholds = [0.45, 0.55, 0.65]  # Default thresholds
    
    results_dict = {}
    
    for threshold in thresholds:
        # Detect forgery at this threshold
        result = detect_forgery_fn(image, threshold, patch_size)
        
        # Store result
        results_dict[str(threshold)] = result
    
    return results_dict