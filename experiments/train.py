# Import statements
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import json
import matplotlib.pyplot as plt
from models.vit_encoder import ViTEncoder
from models.siamese import SiameseNetwork
from utils.dataset import CMFDataset
from utils.mac_utils import get_device, optimize_memory, recommend_batch_size, set_pytorch_threads, get_mac_info
import torchvision
import numpy as np
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

def train(config=None):
    # Enhanced default configuration
    if config is None:
        config = {
            "dataset": "data/CoMoFoD_small_v2",
            "batch_size": None,  # Will be automatically set
            "epochs": 25,        # Increased for better convergence
            "learning_rate": 3e-5,  # Optimized learning rate for transformer training
            "encoder_lr": 8e-6,     # Lower learning rate for encoder for stability
            "save_path": "outputs/checkpoints/",
            "log_interval": 10,
            "validation_split": 0.1,
            "early_stopping_patience": 8,  # Increased patience for better convergence
            "weight_decay": 2e-5,    # Adjusted for better regularization
            "feature_dim": 512,      # Feature dim from ViT to Siamese
            "vit_model": "vit_base_patch16_224",
            "vit_freeze_ratio": 0.6,  # Fine-tune more layers for better feature extraction
            "optimizer": "AdamW",     # Better optimizer for transformers
            "scheduler": "cosine_warmup",
            "use_mixed_precision": True,  # Enable mixed precision for faster training on M4 Pro
            "use_focal_loss": True,   # Better for imbalanced datasets
            "focal_alpha": 0.75,      # Weight for positive class
            "focal_gamma": 2.0,       # Focus on hard examples
            "gradient_clip": 1.0,     # Prevent gradient explosion
            "heavy_augmentations": True,  # Enable heavy augmentations for robustness
            "apply_label_smoothing": True,  # Apply label smoothing for better generalization
            "label_smoothing": 0.05,      # Smoothing factor
        }
    
    # Create output directories
    os.makedirs(config["save_path"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Print system information
    mac_info = get_mac_info()
    print(f"System Information:")
    for key, value in mac_info.items():
        print(f"  {key}: {value}")
    
    # Set PyTorch threads for CPU operations - optimized for M4 Pro
    num_threads = set_pytorch_threads()
    print(f"PyTorch using {num_threads} CPU threads")
    
    # Determine optimal batch size for M4 Pro if not specified
    if config["batch_size"] is None:
        config["batch_size"] = recommend_batch_size()
        # For mixed precision, we can use slightly larger batch sizes
        if config["use_mixed_precision"]:
            config["batch_size"] = min(32, config["batch_size"] + 4)  
        print(f"Automatically determined batch size: {config['batch_size']}")
    
    # Get device - MPS (Apple GPU) or CPU
    device = get_device()
    print(f"Using device: {device}")
    
    # Enable mixed precision if available
    if config["use_mixed_precision"] and device == "mps":
        print("Enabling mixed precision training for faster performance")
    
    # Initialize models with matching feature dimensions
    print("Initializing models...")
    vit = ViTEncoder(
        model_name=config["vit_model"],
        pretrained=True,
        feature_dim=config["feature_dim"],
        freeze_ratio=config["vit_freeze_ratio"],
        use_multiscale=True,  # Use multi-scale features for better performance
        use_mixed_precision=config["use_mixed_precision"]  # Set mixed precision based on config
    )
    
    # Create Siamese with matching feature dimensions
    siamese = SiameseNetwork(
        feature_dim=config["feature_dim"], 
        hidden_dims=[512, 256, 128], 
        dropout_rates=[0.4, 0.3, 0.2]  # Increased dropout for regularization
    )
    
    # Print model summaries
    print(f"ViT Encoder: Output feature dimension = {config['feature_dim']}")
    print(f"Parameter counts: ViT = {sum(p.numel() for p in vit.parameters() if p.requires_grad):,}, " +
          f"Siamese = {sum(p.numel() for p in siamese.parameters() if p.requires_grad):,}")
    
    # Move models to device
    vit.to(device)
    siamese.to(device)
    
    # Load dataset
    print(f"Loading dataset from {config['dataset']}...")
    try:
        # Use train, val folders as expected by the updated dataset
        train_dataset = CMFDataset(config['dataset'], split='train')
        val_dataset = CMFDataset(config['dataset'], split='val')
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True,
            num_workers=min(6, os.cpu_count() or 2),  # Increased workers for M4 Pro
            pin_memory=True,  # This speeds up data transfer to GPU
            drop_last=True,   # Avoid issues with single sample batches
            persistent_workers=True  # Keep workers alive between epochs for faster loading
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["batch_size"], 
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 2),
            pin_memory=True,
            persistent_workers=True
        )
        
        print(f"Dataset loaded with {len(train_dataset)} training and {len(val_dataset)} validation samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Setup loss function
    if config["use_focal_loss"]:
        # Focal Loss for handling potential class imbalance
        from torch.nn import functional as F
        class FocalLoss(nn.Module):
            def __init__(self, alpha=0.75, gamma=2.0, reduction='mean', label_smoothing=0.0):
                super(FocalLoss, self).__init__()
                self.alpha = alpha
                self.gamma = gamma
                self.reduction = reduction
                self.label_smoothing = label_smoothing
                
            def forward(self, inputs, targets):
                # Apply label smoothing if enabled
                if self.label_smoothing > 0:
                    # One-sided smoothing for binary case
                    targets = torch.where(targets > 0.5, 
                                         1.0 - self.label_smoothing, 
                                         self.label_smoothing)
                
                # Apply logits directly - no sigmoid in model
                inputs_sigmoid = torch.sigmoid(inputs)
                BCE_loss = F.binary_cross_entropy(inputs_sigmoid, targets, reduction='none')
                pt = torch.exp(-BCE_loss)
                focal_loss = self.alpha * (1-pt)**self.gamma * BCE_loss
                
                if self.reduction == 'mean':
                    return torch.mean(focal_loss)
                else:
                    return focal_loss
        
        label_smoothing = config["label_smoothing"] if config["apply_label_smoothing"] else 0.0
        criterion = FocalLoss(
            alpha=config["focal_alpha"], 
            gamma=config["focal_gamma"],
            label_smoothing=label_smoothing
        )
        print(f"Using Focal Loss with alpha={config['focal_alpha']}, gamma={config['focal_gamma']}")
        if label_smoothing > 0:
            print(f"Applied label smoothing: {label_smoothing}")
    else:
        # BCEWithLogitsLoss for numerical stability - sigmoid in loss, not model
        if config["apply_label_smoothing"]:
            criterion = nn.BCEWithLogitsLoss(label_smoothing=config["label_smoothing"])
            print(f"Using BCE with label smoothing: {config['label_smoothing']}")
        else:
            criterion = nn.BCEWithLogitsLoss()
            print("Using standard BCE Loss with logits")
    
    # Setup optimizer with parameter groups and separate learning rates
    if config["optimizer"] == "AdamW":
        # Use AdamW with separate parameter groups and M4-optimized settings
        optimizer = optim.AdamW([
            {'params': vit.parameters(), 'lr': config["encoder_lr"]},
            {'params': siamese.parameters(), 'lr': config["learning_rate"]}
        ], weight_decay=config["weight_decay"], eps=1e-8)
        print(f"Using AdamW optimizer with encoder_lr={config['encoder_lr']}, " + 
              f"classifier_lr={config['learning_rate']}, weight_decay={config['weight_decay']}")
    else:
        # Standard Adam
        optimizer = optim.Adam(
            list(vit.parameters()) + list(siamese.parameters()), 
            lr=config["learning_rate"],
            weight_decay=config["weight_decay"]
        )
        print(f"Using Adam optimizer with lr={config['learning_rate']}, weight_decay={config['weight_decay']}")
    
    # Learning rate scheduler
    if config["scheduler"] == "cosine_warmup":
        # Cosine annealing with warm restarts - better for transformer models
        from torch.optim.lr_scheduler import OneCycleLR
        
        # One cycle LR with warmup - highly effective for vision transformers
        total_steps = len(train_loader) * config["epochs"]
        scheduler = OneCycleLR(
            optimizer,
            max_lr=[config["encoder_lr"] * 5, config["learning_rate"] * 5],  # Peak LR during cycle
            total_steps=total_steps,
            pct_start=0.1,  # Use 10% of the training for warmup
            anneal_strategy='cos',
            div_factor=25.0,  # Initial LR = max_lr/div_factor
            final_div_factor=1000.0,  # Final LR = max_lr/final_div_factor
        )
        print("Using OneCycleLR scheduler with warmup")
    else:
        # Standard reduce on plateau
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3,
            verbose=True
        )
        print("Using ReduceLROnPlateau scheduler")
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": [],
        "val_precision": [],
        "val_recall": [],
        "val_f1": [],
        "learning_rates": [],
        "epoch_times": []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    best_val_f1 = 0.0
    patience_counter = 0
    
    # Save model configurations
    model_config = {
        "training_config": config,
        "vit_encoder": {
            "model_name": config["vit_model"],
            "feature_dim": config["feature_dim"],
            "freeze_ratio": config["vit_freeze_ratio"]
        },
        "siamese": {
            "feature_dim": config["feature_dim"],
            "hidden_dims": [512, 256, 128],
            "dropout_rates": [0.4, 0.3, 0.2]
        }
    }
    
    with open(os.path.join(config["save_path"], "model_config.json"), "w") as f:
        json.dump(model_config, f, indent=2)
    
    # Create gradient scaler for mixed precision training
    scaler = torch.cuda.amp.GradScaler() if torch.cuda.is_available() and config["use_mixed_precision"] else None
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        epoch_start_time = time.time()
        
        # Training phase
        vit.train()
        siamese.train()
        total_loss = 0
        train_samples = 0
        
        # Update progress bar description
        train_loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Train]")
        
        for i, (img1, img2, label) in enumerate(train_loop):
            # Move data to device
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Forward pass with automatic mixed precision
            optimizer.zero_grad()
            
            if config["use_mixed_precision"] and device == "mps":
                # Use PyTorch autocast for mixed precision training
                with torch.autocast(device_type="mps"):
                    # Process each image through ViT encoder
                    feat1 = vit(img1)
                    feat2 = vit(img2)
                    
                    # Pass features through Siamese network
                    preds = siamese(feat1, feat2)
                    
                    # Squeeze predictions and labels for loss calculation
                    preds = preds.squeeze()
                    label = label.squeeze()
                    
                    # Calculate loss
                    loss = criterion(preds, label)
                
                # Backward pass with gradient scaling for mixed precision
                if scaler is not None:
                    scaler.scale(loss).backward()
                    
                    # Gradient clipping to prevent exploding gradients
                    if config["gradient_clip"] > 0:
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(
                            list(vit.parameters()) + list(siamese.parameters()), 
                            config["gradient_clip"]
                        )
                    
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # For MPS which doesn't support CUDA scaler, just do regular backward
                    loss.backward()
                    
                    if config["gradient_clip"] > 0:
                        torch.nn.utils.clip_grad_norm_(
                            list(vit.parameters()) + list(siamese.parameters()), 
                            config["gradient_clip"]
                        )
                    
                    optimizer.step()
            else:
                # Standard precision training
                feat1 = vit(img1)
                feat2 = vit(img2)
                preds = siamese(feat1, feat2)
                preds = preds.squeeze()
                label = label.squeeze()
                loss = criterion(preds, label)
                
                loss.backward()
                
                if config["gradient_clip"] > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(vit.parameters()) + list(siamese.parameters()), 
                        config["gradient_clip"]
                    )
                
                optimizer.step()
            
            # Update learning rate for OneCycleLR every batch
            if config["scheduler"] == "cosine_warmup":
                scheduler.step()
            
            # Update stats
            batch_size = img1.size(0)
            total_loss += loss.item() * batch_size
            train_samples += batch_size
            
            # Update progress bar with current loss
            train_loop.set_postfix({"loss": f"{loss.item():.4f}"})
            
            # Free up memory
            if i % 5 == 0:
                optimize_memory()
        
        # Calculate epoch stats
        epoch_loss = total_loss / train_samples
        history["train_loss"].append(epoch_loss)
        
        # Store current learning rates
        current_lrs = [param_group['lr'] for param_group in optimizer.param_groups]
        history["learning_rates"].append(current_lrs)
        
        # Validation phase
        vit.eval()
        siamese.eval()
        val_loss = 0
        val_samples = 0
        
        # Track metrics with confusion matrix
        all_preds = []
        all_labels = []
        all_scores = []  # Store raw prediction scores for ROC analysis
        
        # Update progress bar description for validation
        val_loop = tqdm(val_loader, desc=f"Epoch {epoch+1}/{config['epochs']} [Val]")
        
        with torch.no_grad():
            for img1, img2, label in val_loop:
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                
                # Forward pass with mixed precision if enabled
                if config["use_mixed_precision"] and device == "mps":
                    with torch.autocast(device_type="mps"):
                        feat1 = vit(img1)
                        feat2 = vit(img2)
                        logits = siamese(feat1, feat2)
                        logits = logits.squeeze()
                        label = label.squeeze()
                        batch_loss = criterion(logits, label)
                else:
                    feat1 = vit(img1)
                    feat2 = vit(img2)
                    logits = siamese(feat1, feat2)
                    logits = logits.squeeze()
                    label = label.squeeze()
                    batch_loss = criterion(logits, label)
                
                val_loss += batch_loss.item() * img1.size(0)
                val_samples += img1.size(0)
                
                # Apply sigmoid to get probabilities for evaluation metrics
                probs = torch.sigmoid(logits)
                
                # Save predictions and labels for metrics
                all_preds.extend((probs > 0.5).float().cpu().numpy())
                all_labels.extend(label.cpu().numpy())
                all_scores.extend(probs.cpu().numpy())
                
                # Update validation progress bar
                val_loop.set_postfix({"loss": f"{batch_loss.item():.4f}"})
        
        # Calculate validation stats
        val_loss = val_loss / val_samples
        
        # Calculate metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        # Calculate accuracy
        val_accuracy = np.mean(all_preds == all_labels)
        
        # Calculate precision, recall, and F1 score
        true_positives = np.sum((all_preds == 1) & (all_labels == 1))
        false_positives = np.sum((all_preds == 1) & (all_labels == 0))
        false_negatives = np.sum((all_preds == 0) & (all_labels == 1))
        
        precision = true_positives / (true_positives + false_positives + 1e-10)
        recall = true_positives / (true_positives + false_negatives + 1e-10)
        f1_score = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # Store metrics in history
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        history["val_precision"].append(precision)
        history["val_recall"].append(recall)
        history["val_f1"].append(f1_score)
        
        # Calculate epoch time and store
        epoch_time = time.time() - epoch_start_time
        history["epoch_times"].append(epoch_time)
        
        # Update plateau scheduler if using
        if config["scheduler"] == "plateau":
            scheduler.step(val_loss)
        
        # Print epoch results
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Val F1: {f1_score:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'vit_state_dict': vit.state_dict(),
            'siamese_state_dict': siamese.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if hasattr(scheduler, 'state_dict') else None,
            'scaler_state_dict': scaler.state_dict() if scaler else None,
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
            'val_f1': f1_score,
            'config': config
        }
        
        # Save the latest model
        torch.save(checkpoint, os.path.join(config["save_path"], "latest_model.pt"))
        
        # Early stopping and best model saving logic (now based on multiple metrics)
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(checkpoint, os.path.join(config["save_path"], "best_loss_model.pt"))
            print(f"New best validation loss: {best_val_loss:.4f}. Model saved.")
            patience_counter = 0
        
        # Also save model with best accuracy separately
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            torch.save(checkpoint, os.path.join(config["save_path"], "best_accuracy_model.pt"))
            print(f"New best validation accuracy: {best_val_acc:.4f}. Model saved.")
            patience_counter = 0
            
        # Save model with best F1 score (best balance of precision and recall)
        if f1_score > best_val_f1:
            best_val_f1 = f1_score
            torch.save(checkpoint, os.path.join(config["save_path"], "best_f1_model.pt"))
            print(f"New best F1 score: {best_val_f1:.4f}. Model saved.")
            patience_counter = 0
        else:
            patience_counter += 1
            print(f"No improvement. Patience: {patience_counter}/{config['early_stopping_patience']}")
            
            if patience_counter >= config["early_stopping_patience"]:
                print("Early stopping triggered!")
                break
        
        # Save training history with unique filename (including hyperparams)
        history_filename = f"training_history_lr{config['learning_rate']}_wd{config['weight_decay']}_bs{config['batch_size']}.json"
        with open(f"logs/{history_filename}", "w") as f:
            json.dump(history, f, indent=2)
        
        # Also save with generic name for convenience
        with open("logs/training_history.json", "w") as f:
            json.dump(history, f, indent=2)
        
        # Plot enhanced learning curves
        plt.figure(figsize=(15, 12))
        
        plt.subplot(2, 3, 1)
        plt.plot(history["train_loss"], label="Train Loss")
        plt.plot(history["val_loss"], label="Validation Loss")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.title("Loss Curves")
        
        plt.subplot(2, 3, 2)
        plt.plot(history["val_accuracy"], label="Accuracy")
        plt.plot(history["val_f1"], label="F1 Score")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.title("Performance Metrics")
        
        plt.subplot(2, 3, 3)
        plt.plot(history["val_precision"], label="Precision")
        plt.plot(history["val_recall"], label="Recall")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.title("Precision & Recall")
        
        plt.subplot(2, 3, 4)
        for i, lr in enumerate(zip(*history["learning_rates"])):
            plt.plot(lr, label=f"Group {i+1}")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.title("Learning Rate Schedule")
        
        plt.subplot(2, 3, 5)
        plt.bar(range(len(history["epoch_times"])), history["epoch_times"])
        plt.xlabel("Epoch")
        plt.ylabel("Time (seconds)")
        plt.title("Epoch Training Times")
        
        # Save plot with unique filename
        curve_filename = f"learning_curves_lr{config['learning_rate']}_wd{config['weight_decay']}_bs{config['batch_size']}.png"
        plt.tight_layout()
        plt.savefig(f"logs/{curve_filename}")
        plt.savefig("logs/learning_curves.png")  # Also save with generic name
        plt.close()
        
        # Clean up memory before next epoch
        optimize_memory()
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}, Best accuracy: {best_val_acc:.4f}, Best F1: {best_val_f1:.4f}")
    print(f"Models saved to {config['save_path']}")
    
    return history

if __name__ == "__main__":
    train()
