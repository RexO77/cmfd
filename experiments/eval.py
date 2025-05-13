import sys # Move sys import up
from pathlib import Path # Move Path import up

# Add project root to Python path (MUST be before other project imports)
sys.path.append(str(Path(__file__).parent.parent))

import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
import cv2
from models.vit_encoder import ViTEncoder
from models.siamese import SiameseNetwork
from utils.dataset import CMFDataset
from utils.metrics import compute_metrics, compute_classification_metrics
from utils.mac_utils import get_device, optimize_memory, recommend_batch_size
from accelerate import Accelerator

def evaluate(model_path, dataset_path, output_path="outputs/evaluations"):
    """
    Evaluates the model on the test dataset.

    Args:
        model_path (str): Path to the model checkpoint.
        dataset_path (str): Path to the *test* dataset directory.
        output_path (str): Directory to save evaluation results.
    """
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Get device
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize models
    vit = ViTEncoder(pretrained=False)  # No need for pretrained weights as we'll load them
    siamese = SiameseNetwork()
    
    # Move models to device
    vit.to(device)
    siamese.to(device)
    
    # Load saved model
    if os.path.exists(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)
        vit.load_state_dict(checkpoint['vit_state_dict'])
        siamese.load_state_dict(checkpoint['siamese_state_dict'])
        print(f"Model loaded (epoch {checkpoint.get('epoch', 'unknown')})")
    else:
        print(f"Model not found at {model_path}")
        return
    
    # Load test dataset - dataset_path should point directly to the test split
    print(f"Loading test dataset from: {dataset_path}")
    test_dataset = CMFDataset(dataset_path, training=False) # Assumes dataset_path is the test split dir
    if not test_dataset.samples:
        print(f"Error: No samples found in the test dataset path: {dataset_path}")
        print("Please ensure the path points to the directory containing image set folders (e.g., data/CoMoFoD_small_v2/test).")
        return
    
    batch_size = recommend_batch_size()
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=min(4, os.cpu_count() or 1)
    )
    
    # Set models to evaluation mode
    vit.eval()
    siamese.eval()
    
    # Evaluation metrics
    all_preds = []
    all_labels = []
    results = []
    
    # Evaluation loop
    with torch.no_grad():
        for img1, img2, label in tqdm(test_loader, desc="Evaluating"):
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Forward pass
            feat1, feat2 = vit(img1), vit(img2)
            preds = siamese(feat1, feat2)
            
            # Store predictions and labels
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(label.cpu().numpy())
            
            # Store results for each sample
            for i in range(len(preds)):
                results.append({
                    'pred': float(preds[i].item()),
                    'label': int(label[i].item()),
                    'correct': bool((preds[i] > 0.5).item() == label[i].item())
                })
    
    # Convert to numpy arrays
    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    
    # Calculate metrics
    metrics = compute_classification_metrics(all_labels, all_preds)
    
    # --- Calculate Average Confidence for Correct Predictions ---
    true_positives = (all_preds > 0.5) & (all_labels == 1)
    true_negatives = (all_preds <= 0.5) & (all_labels == 0)
    
    avg_conf_tp = np.mean(all_preds[true_positives]) if np.any(true_positives) else np.nan
    # For TN, confidence is (1 - pred_score), but let's report avg raw score for simplicity
    avg_score_tn = np.mean(all_preds[true_negatives]) if np.any(true_negatives) else np.nan 
    # --- End Calculate Average Confidence ---

    # Print results
    print("\nEvaluation Results (using threshold=0.5):") # Clarify threshold
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # --- Print Average Confidence --- 
    print(f"\n  Avg. Prediction Score (Confidence) for Correctly Identified Forged (TP): {avg_conf_tp:.4f}")
    print(f"  Avg. Prediction Score for Correctly Identified Original (TN): {avg_score_tn:.4f}")
    print("\nNote: Detailed prediction scores (percentages) for each sample are in outputs/evaluations/results.csv")
    # --- End Print Average Confidence ---

    # Plot precision-recall curve
    plt.figure(figsize=(10, 4))
    
    # Plot confusion matrix (FP, FN) distribution
    thresholds = np.linspace(0, 1, 100)
    precisions = []
    recalls = []
    f1_scores = []
    
    for threshold in thresholds:
        preds_binary = (all_preds > threshold).astype(int)
        tp = np.sum((preds_binary == 1) & (all_labels == 1))
        fp = np.sum((preds_binary == 1) & (all_labels == 0))
        fn = np.sum((preds_binary == 0) & (all_labels == 1))
        
        # Calculate precision and recall
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        
        # Calculate F1 score
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        precisions.append(precision)
        recalls.append(recall)
        f1_scores.append(f1)
    
    # Plot precision-recall curve
    plt.subplot(1, 2, 1)
    plt.plot(thresholds, precisions, 'b', label='Precision')
    plt.plot(thresholds, recalls, 'g', label='Recall')
    plt.plot(thresholds, f1_scores, 'r', label='F1 Score')
    plt.axvline(x=0.5, color='k', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Precision-Recall vs Threshold')
    plt.legend()
    plt.grid(True)
    
    # Plot prediction distribution
    plt.subplot(1, 2, 2)
    plt.hist([all_preds[all_labels == 0], all_preds[all_labels == 1]], 
             bins=20, range=(0, 1), 
             label=['Original', 'Forged'],
             color=['green', 'red'],
             alpha=0.7)
    plt.axvline(x=0.5, color='k', linestyle='--', label='Threshold = 0.5')
    plt.xlabel('Prediction Score')
    plt.ylabel('Count')
    plt.title('Prediction Distribution')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_path, 'evaluation_curves.png'))
    
    # Save metrics
    with open(os.path.join(output_path, 'metrics.txt'), 'w') as f:
        f.write("Evaluation Metrics:\n")
        for metric, value in metrics.items():
            f.write(f"{metric}: {value:.4f}\n")
    
    # Save results for each sample to CSV
    import csv
    with open(os.path.join(output_path, 'results.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['pred', 'label', 'correct'])
        writer.writeheader()
        writer.writerows(results)
    
    print(f"Evaluation results saved to {output_path}")
    
    return metrics

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate the CMFD model")
    parser.add_argument('--model', type=str, default="outputs/checkpoints/most_accuracy_model.pt", 
                        help="Path to the trained model checkpoint")
    parser.add_argument('--dataset', type=str, default="data/CoMoFoD_small_v2/test", 
                        help="Path to the *test* dataset directory")
    parser.add_argument('--output', type=str, default="outputs/evaluations", 
                        help="Path to save evaluation results")
    
    args = parser.parse_args()
    evaluate(args.model, args.dataset, args.output)
