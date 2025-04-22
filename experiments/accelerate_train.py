import os
import sys
import time
import json
from pathlib import Path
import argparse
import math

# Add project root to Python path
sys.path.append(str(Path(__file__).parent.parent))

# Third-party imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import project modules
from models.vit_encoder import ViTEncoder
from models.siamese import SiameseNetwork
from utils.dataset import CMFDataset
from utils.mac_utils import (
    get_device, 
    optimize_memory, 
    recommend_batch_size, 
    set_pytorch_threads, 
    get_mac_info,
    optimize_memory_for_m_series,
    get_accelerate_config
)

# Add Model EMA (Exponential Moving Average) for more stable models
class ModelEMA:
    """
    Model Exponential Moving Average
    Keep a moving average of everything in the model state_dict (parameters and buffers).
    """
    def __init__(self, model, decay=0.999, device=None):
        self.ema = {}
        self.decay = decay
        self.device = device
        
        # Register model parameters
        for name, param in model.state_dict().items():
            if param.requires_grad:
                self.ema[name] = param.detach().clone()
    
    def update(self, model):
        with torch.no_grad():
            for name, param in model.state_dict().items():
                if name in self.ema:
                    self.ema[name] = self.ema[name] * self.decay + param.detach() * (1. - self.decay)
    
    def apply_to(self, model):
        """Apply EMA weights to model for evaluation"""
        for name, param in model.state_dict().items():
            if name in self.ema:
                param.copy_(self.ema[name])

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, using defaults")
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def accelerate_train(config_path="config.yaml", args=None):
    """
    Train the model with Accelerate library for optimized performance on Apple Silicon.
    
    Args:
        config_path: Path to the configuration file
        args: Parsed command-line arguments (optional)
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration sections with defaults
    dataset_config = config.get("dataset", {"path": "data/CoMoFoD_small_v2", "validation_split": 0.1})
    training_config = config.get("training", {
        "batch_size": None,
        "epochs": 20,  # Increased default epochs
        "learning_rate": 5e-5,  # Lower default learning rate
        "weight_decay": 2e-5,  # Adjusted weight decay
        "early_stopping_patience": 8,  # Increased patience
        "gradient_clip_val": 1.0,  # Add gradient clipping
        "use_ema": True,  # Use EMA by default
        "ema_decay": 0.999,  # EMA decay rate
        "warmup_epochs": 2  # Learning rate warmup epochs
    })
    paths_config = config.get("paths", {
        "checkpoints": "outputs/checkpoints/",
        "logs": "logs/"
    })

    # Override config with command-line arguments if provided
    if args:
        if args.lr is not None:
            training_config["learning_rate"] = args.lr
            print(f"Overriding learning_rate with command-line value: {args.lr}")
        if args.wd is not None:
            training_config["weight_decay"] = args.wd
            print(f"Overriding weight_decay with command-line value: {args.wd}")
        if args.batch_size is not None:
            training_config["batch_size"] = args.batch_size
            print(f"Overriding batch_size with command-line value: {args.batch_size}")
        if args.epochs is not None:
            training_config["epochs"] = args.epochs
            print(f"Overriding epochs with command-line value: {args.epochs}")
        if args.patience is not None:
            training_config["early_stopping_patience"] = args.patience
            print(f"Overriding early_stopping_patience with command-line value: {args.patience}")

    # Get accelerate config optimized for the current device
    accelerate_config = get_accelerate_config()
    
    # Override with user config if provided
    if "acceleration" in config:
        for key, value in config["acceleration"].items():
            accelerate_config[key] = value
    
    # Print accelerate configuration
    print(f"Accelerate configuration: {accelerate_config}")
    
    # Initialize accelerator
    # Force mixed_precision to "no" for MPS compatibility
    accelerator = Accelerator(
        mixed_precision="no",
        gradient_accumulation_steps=accelerate_config["gradient_accumulation_steps"]
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directories
    os.makedirs(paths_config.get("checkpoints", "outputs/checkpoints/"), exist_ok=True)
    os.makedirs(paths_config.get("logs", "logs/"), exist_ok=True)
    
    # Print system information
    mac_info = get_mac_info()
    accelerator.print(f"System Information:")
    for key, value in mac_info.items():
        accelerator.print(f"  {key}: {value}")
    
    # Set PyTorch threads for CPU operations
    num_threads = set_pytorch_threads()
    accelerator.print(f"PyTorch using {num_threads} CPU threads")
    
    # Optimize memory for M-series Macs
    if mac_info.get("apple_silicon", False):
        optimize_memory_for_m_series()
        accelerator.print("Applied M-series specific memory optimizations")
    
    # Determine batch size if not specified OR overridden
    batch_size = training_config.get("batch_size")
    if batch_size is None:
        # For MPS backend, use a smaller default batch size to avoid OOM errors
        if torch.backends.mps.is_available():
            batch_size = 16  # Reduced from the auto-determined value
            accelerator.print(f"Using smaller batch size for MPS: {batch_size}")
        else:
            batch_size = recommend_batch_size()
            accelerator.print(f"Automatically determined batch size: {batch_size}")
    else:
        accelerator.print(f"Using batch size: {batch_size}")
        
    # Set MPS memory optimization environment variables if running on MPS
    if torch.backends.mps.is_available():
        os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.7"  # Lower memory usage threshold
        accelerator.print("Set MPS memory optimization parameters")
    
    # Initialize models with improved configurations
    accelerator.print("Initializing models...")
    vit = ViTEncoder(pretrained=True, freeze_ratio=0.7)  # Use selective freezing
    siamese = SiameseNetwork(
        feature_dim=vit.feature_dim, 
        hidden_dims=[512, 256, 128], 
        dropout_rates=[0.3, 0.3, 0.2]
    )
    
    # Setup dataset
    base_dataset_path = dataset_config.get("path", "data/CoMoFoD_small_v2")
    train_data_path = os.path.join(base_dataset_path, "train")
    val_data_path = os.path.join(base_dataset_path, "val")
    
    accelerator.print(f"Loading training dataset from {train_data_path}...")
    accelerator.print(f"Loading validation dataset from {val_data_path}...")
    
    try:
        # Load pre-split datasets directly
        train_dataset = CMFDataset(train_data_path, training=True, num_pairs_per_set=15)  # More pairs per set
        val_dataset = CMFDataset(val_data_path, training=False)
        
        # Ensure datasets are not empty
        if len(train_dataset) == 0:
            accelerator.print(f"Error: No training samples found in {train_data_path}")
            return
        if len(val_dataset) == 0:
            accelerator.print(f"Warning: No validation samples found in {val_data_path}")
        
        # Create data loaders with optimized settings for M-series
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True if (os.cpu_count() or 0) > 1 else False,
            drop_last=True  # Drop last batch to avoid issues with batch norm
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True if (os.cpu_count() or 0) > 1 else False
        )
        
        accelerator.print(f"Dataset loaded with {len(train_dataset)} training and {len(val_dataset)} validation samples")
        
    except Exception as e:
        accelerator.print(f"Error loading dataset: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Loss function with label smoothing for better generalization
    criterion = nn.BCEWithLogitsLoss(reduction='mean')
    # Remove static pos_weight tensor initialization as it could cause device mismatch issues
    # The accelerator will handle moving everything to the right device
    
    # Optimizer with decoupled weight decay
    optimizer = torch.optim.AdamW(
        [
            {'params': vit.parameters(), 'lr': training_config.get("learning_rate") * 0.1},  # Lower LR for pretrained model
            {'params': siamese.parameters(), 'lr': training_config.get("learning_rate")}
        ],
        weight_decay=training_config.get("weight_decay")
    )
    
    # Calculate total steps for scheduling
    num_epochs = training_config.get("epochs", 20)
    warmup_epochs = training_config.get("warmup_epochs", 2)
    total_steps = len(train_loader) * num_epochs
    warmup_steps = len(train_loader) * warmup_epochs
    
    # Learning rate scheduler with warmup
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            # Linear warmup
            return float(current_step) / float(max(1, warmup_steps))
        # Cosine annealing
        progress = float(current_step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # Model EMA setup
    use_ema = training_config.get("use_ema", True)
    ema_vit = None
    ema_siamese = None
    if use_ema:
        accelerator.print("Using EMA for model averaging")
        ema_vit = ModelEMA(vit, decay=training_config.get("ema_decay", 0.999))
        ema_siamese = ModelEMA(siamese, decay=training_config.get("ema_decay", 0.999))
    
    # Prepare training components with Accelerator
    vit, siamese, optimizer, train_loader, val_loader = accelerator.prepare(
        vit, siamese, optimizer, train_loader, val_loader
    )
    
    # Training history
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "learning_rates": []}
    
    # Early stopping variables
    best_val_loss = float('inf')
    best_val_acc = 0.0
    patience_counter = 0
    early_stopping_patience = training_config.get("early_stopping_patience", 8)
    
    # Training loop
    accelerator.print(f"Starting training for {num_epochs} epochs...")
    accelerator.print(f"Hyperparameters: LR={training_config['learning_rate']}, "
                     f"WD={training_config['weight_decay']}, BS={batch_size}, "
                     f"Patience={early_stopping_patience}, Warmup={warmup_epochs}")
    
    global_step = 0
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # Training phase
        vit.train()
        siamese.train()
        total_loss = 0
        train_samples = 0
        
        progress_bar = tqdm(
            train_loader, 
            desc=f"Epoch {epoch+1}/{num_epochs}", 
            disable=not accelerator.is_local_main_process
        )
        
        for i, (img1, img2, label) in enumerate(progress_bar):
            global_step += 1
            
            # Forward pass
            with accelerator.accumulate(vit, siamese):
                feat1, feat2 = vit(img1), vit(img2)
                preds = siamese(feat1, feat2)
                
                # Calculate loss (preds should not have sigmoid applied for BCEWithLogitsLoss)
                loss = criterion(preds, label)
                
                # Update loss statistics
                total_loss += loss.item() * img1.size(0)
                train_samples += img1.size(0)
                
                # Backward pass and optimization
                accelerator.backward(loss)
                
                # Gradient clipping
                if training_config.get("gradient_clip_val", 0) > 0:
                    accelerator.clip_grad_norm_(
                        list(vit.parameters()) + list(siamese.parameters()),
                        training_config["gradient_clip_val"]
                    )
                
                # Optimizer step
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                
                # Update learning rate
                scheduler.step()
                
                # Update EMA model after optimizer step
                if use_ema and accelerator.sync_gradients:
                    ema_vit.update(accelerator.unwrap_model(vit))
                    ema_siamese.update(accelerator.unwrap_model(siamese))
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item(), 
                "lr": optimizer.param_groups[1]['lr']  # Show siamese network LR
            })
            
            # Periodically free up memory
            if i % 10 == 0:
                optimize_memory()
        
        # Calculate epoch stats
        epoch_loss = total_loss / train_samples
        history["train_loss"].append(epoch_loss)
        # Track learning rates
        history["learning_rates"].append(optimizer.param_groups[1]['lr'])
        
        # Validation phase - use EMA model if enabled
        if use_ema and accelerator.is_local_main_process:
            # Apply EMA weights for validation
            ema_vit.apply_to(accelerator.unwrap_model(vit))
            ema_siamese.apply_to(accelerator.unwrap_model(siamese))
        
        # Evaluation mode
        vit.eval()
        siamese.eval()
        val_loss = 0
        val_correct = 0
        val_samples = 0
        
        # Use torch.inference_mode for faster evaluation
        with torch.inference_mode():
            for img1, img2, label in tqdm(val_loader, desc="Validation"):
                feat1, feat2 = vit(img1), vit(img2)
                preds = siamese(feat1, feat2)
                
                # Calculate loss (BCEWithLogitsLoss requires unnormalized logits)
                batch_loss = criterion(preds, label)
                val_loss += batch_loss.item() * img1.size(0)
                
                # Apply sigmoid for binary classification
                preds_sigmoid = torch.sigmoid(preds)
                predicted = (preds_sigmoid > 0.5).float()
                val_correct += (predicted == label).sum().item()
                val_samples += img1.size(0)
        
        # If using EMA, restore original weights after validation
        if use_ema and accelerator.is_local_main_process:
            # Store original weights before validation
            original_vit_state = {k: v.clone() for k, v in accelerator.unwrap_model(vit).state_dict().items()}
            original_siamese_state = {k: v.clone() for k, v in accelerator.unwrap_model(siamese).state_dict().items()}
            
            # After validation, restore the original weights
            accelerator.unwrap_model(vit).load_state_dict(original_vit_state)
            accelerator.unwrap_model(siamese).load_state_dict(original_siamese_state)
            
        # Calculate validation stats
        val_loss = val_loss / val_samples
        val_accuracy = val_correct / val_samples
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch results
        accelerator.print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"Train Loss: {epoch_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Accuracy: {val_accuracy:.4f}, "
            f"LR: {optimizer.param_groups[1]['lr']:.6f}, "
            f"Time: {epoch_time:.2f}s"
        )
        
        # Save checkpoint
        if accelerator.is_local_main_process:
            # Unwrap models before saving
            unwrapped_vit = accelerator.unwrap_model(vit)
            unwrapped_siamese = accelerator.unwrap_model(siamese)
            
            # Create checkpoint
            checkpoint = {
                'epoch': epoch + 1,
                'vit_state_dict': unwrapped_vit.state_dict(),
                'siamese_state_dict': unwrapped_siamese.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
                'hyperparams': {
                    'learning_rate': training_config['learning_rate'],
                    'weight_decay': training_config['weight_decay'],
                    'batch_size': batch_size,
                    'epochs': num_epochs
                }
            }
            
            # Add EMA weights if using them
            if use_ema:
                # Copy EMA weights to separate state dict for saving
                ema_vit.apply_to(unwrapped_vit)
                ema_siamese.apply_to(unwrapped_siamese)
                checkpoint['vit_ema_state_dict'] = unwrapped_vit.state_dict().copy()
                checkpoint['siamese_ema_state_dict'] = unwrapped_siamese.state_dict().copy()
                # Restore original weights
                unwrapped_vit.load_state_dict(checkpoint['vit_state_dict'])
                unwrapped_siamese.load_state_dict(checkpoint['siamese_state_dict'])
            
            # Save latest model
            torch.save(checkpoint, os.path.join(paths_config.get("checkpoints"), "latest_model.pt"))
            
            # Save the best model (two criteria: loss and accuracy)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(checkpoint, os.path.join(paths_config.get("checkpoints"), "best_loss_model.pt"))
                accelerator.print(f"New best validation loss: {best_val_loss:.4f}. Model saved.")
                # Reset patience if loss improves
                patience_counter = 0
                
            if val_accuracy > best_val_acc:
                best_val_acc = val_accuracy
                torch.save(checkpoint, os.path.join(paths_config.get("checkpoints"), "best_accuracy_model.pt"))
                accelerator.print(f"New best validation accuracy: {best_val_acc:.4f}. Model saved.")
                # Reset patience if accuracy improves
                patience_counter = 0
            elif val_loss >= best_val_loss and val_accuracy <= best_val_acc:
                # Only increment patience if both metrics don't improve
                patience_counter += 1
                accelerator.print(f"No improvement. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    accelerator.print("Early stopping triggered!")
                    break
            
            # Save training history
            history_log = {
                "hyperparameters": {
                    "learning_rate": training_config['learning_rate'],
                    "weight_decay": training_config['weight_decay'],
                    "batch_size": batch_size,
                    "epochs_run": epoch + 1,
                    "early_stopping_patience": early_stopping_patience,
                    "use_ema": use_ema
                },
                "best_metrics": {
                    "best_val_loss": best_val_loss,
                    "best_val_accuracy": best_val_acc
                },
                "history": history
            }
            
            log_filename = f"training_history_lr{training_config['learning_rate']}_wd{training_config['weight_decay']}_bs{batch_size}.json"
            with open(os.path.join(paths_config.get("logs"), log_filename), "w") as f:
                json.dump(history_log, f, indent=4)
            
            # Plot learning curves
            plot_filename = f"learning_curves_lr{training_config['learning_rate']}_wd{training_config['weight_decay']}_bs{batch_size}.png"
            plt.figure(figsize=(15, 5))
            
            plt.subplot(1, 3, 1)
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title(f"Loss (LR={training_config['learning_rate']}, WD={training_config['weight_decay']}, BS={batch_size})")
            
            plt.subplot(1, 3, 2)
            plt.plot(history["val_accuracy"], label="Validation Accuracy")
            plt.axhline(y=best_val_acc, color='r', linestyle='--', label=f"Best: {best_val_acc:.4f}")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title(f"Validation Accuracy")
            
            plt.subplot(1, 3, 3)
            plt.plot(history["learning_rates"], label="Learning Rate")
            plt.xlabel("Epoch")
            plt.ylabel("Learning Rate")
            plt.yscale('log')
            plt.legend()
            plt.title("Learning Rate Schedule")
            
            plt.tight_layout()
            plt.savefig(os.path.join(paths_config.get("logs"), plot_filename))
            plt.close()
        
        # Clean up memory before next epoch
        optimize_memory()
    
    accelerator.print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    accelerator.print(f"Best validation accuracy: {best_val_acc:.4f}")
    accelerator.print(f"Models saved to {paths_config.get('checkpoints')}")
    
    return history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Accelerated training for Copy-Move Forgery Detection with Hyperparameter Tuning"
    )
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate (overrides config)")
    parser.add_argument("--wd", type=float, default=None, help="Weight decay (overrides config)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config)")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs (overrides config)")
    parser.add_argument("--patience", type=int, default=None, help="Early stopping patience (overrides config)")
    
    args = parser.parse_args()
    accelerate_train(config_path=args.config, args=args)
