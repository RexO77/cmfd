import numpy as np
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score, accuracy_score
import torch

def compute_metrics(pred_mask, gt_mask, threshold=0.5):
    """
    Compute evaluation metrics for forgery detection.
    
    Args:
        pred_mask: Predicted forgery mask
        gt_mask: Ground truth mask
        threshold: Threshold to binarize prediction mask
        
    Returns:
        dict: Dictionary of metrics (F1, IoU, Precision, Recall, Accuracy)
    """
    # If tensors, convert to numpy
    if isinstance(pred_mask, torch.Tensor):
        pred_mask = pred_mask.detach().cpu().numpy()
    if isinstance(gt_mask, torch.Tensor):
        gt_mask = gt_mask.detach().cpu().numpy()
        
    # Flatten and binarize
    pred = pred_mask.flatten() > threshold
    gt = gt_mask.flatten() > 0.5
    
    # Handle edge case where all predictions or all ground truth are one class
    if np.sum(gt) == 0 and np.sum(pred) == 0:
        return {"F1": 1.0, "IoU": 1.0, "Precision": 1.0, "Recall": 1.0, "Accuracy": 1.0}
    if np.sum(gt) == 0:
        return {"F1": 0.0, "IoU": 0.0, "Precision": 0.0, "Recall": 1.0, "Accuracy": 1.0 - np.mean(pred)}
    if np.sum(pred) == 0:
        return {"F1": 0.0, "IoU": 0.0, "Precision": 1.0, "Recall": 0.0, "Accuracy": 1.0 - np.mean(gt)}
    
    # Calculate metrics
    f1 = f1_score(gt, pred)
    iou = jaccard_score(gt, pred)
    precision = precision_score(gt, pred)
    recall = recall_score(gt, pred)
    accuracy = accuracy_score(gt, pred)
    
    return {
        "F1": f1, 
        "IoU": iou, 
        "Precision": precision, 
        "Recall": recall,
        "Accuracy": accuracy
    }

def compute_classification_metrics(y_true, y_pred, threshold=0.5):
    """
    Compute classification metrics for forgery detection.
    
    Args:
        y_true: Ground truth labels (0 or 1)
        y_pred: Predicted probabilities
        threshold: Classification threshold
        
    Returns:
        dict: Dictionary of metrics
    """
    # Convert to numpy if needed
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
        
    # Ensure proper shapes
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    
    # Binarize predictions
    y_pred_binary = (y_pred > threshold).astype(int)
    
    # Calculate metrics
    accuracy = accuracy_score(y_true, y_pred_binary)
    
    # Handle edge cases to avoid warnings
    if np.sum(y_true) == 0 and np.sum(y_pred_binary) == 0:
        precision, recall, f1 = 1.0, 1.0, 1.0
    elif np.sum(y_true) == 0:
        precision, recall, f1 = 0.0, 1.0, 0.0
    elif np.sum(y_pred_binary) == 0:
        precision, recall, f1 = 1.0, 0.0, 0.0
    else:
        precision = precision_score(y_true, y_pred_binary)
        recall = recall_score(y_true, y_pred_binary)
        f1 = f1_score(y_true, y_pred_binary)
    
    return {
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1": f1
    }
