import os
import sys
import time
import json
from pathlib import Path

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

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, using defaults")
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def accelerate_train(config_path="config.yaml"):
    """
    Train the model with Accelerate library for optimized performance on Apple Silicon.
    
    Args:
        config_path: Path to the configuration file
    """
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration sections with defaults
    dataset_config = config.get("dataset", {"path": "data/CoMoFoD_small_v2", "validation_split": 0.1})
    training_config = config.get("training", {
        "batch_size": None,
        "epochs": 10,
        "learning_rate": 0.0001,
        "weight_decay": 1e-5,
        "early_stopping_patience": 5
    })
    paths_config = config.get("paths", {
        "checkpoints": "outputs/checkpoints/",
        "logs": "logs/"
    })
    
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
    
    # Determine batch size if not specified
    batch_size = training_config.get("batch_size")
    if batch_size is None:
        batch_size = recommend_batch_size()
        accelerator.print(f"Automatically determined batch size: {batch_size}")
    
    # Initialize models
    accelerator.print("Initializing models...")
    vit = ViTEncoder(pretrained=True)
    siamese = SiameseNetwork()
    
    # Setup dataset
    accelerator.print(f"Loading dataset from {dataset_config.get('path')}...")
    try:
        full_dataset = CMFDataset(dataset_config.get("path"), training=True)
        
        # Split dataset
        val_size = int(dataset_config.get("validation_split", 0.1) * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        # Create data loaders with optimized settings for M-series
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True if os.cpu_count() > 1 else False
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=batch_size,
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True,
            persistent_workers=True if os.cpu_count() > 1 else False
        )
        
        accelerator.print(f"Dataset loaded with {len(train_dataset)} training and {len(val_dataset)} validation samples")
        
    except Exception as e:
        accelerator.print(f"Error loading dataset: {e}")
        return
    
    # Loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        list(vit.parameters()) + list(siamese.parameters()),
        lr=training_config.get("learning_rate", 0.0001),
        weight_decay=training_config.get("weight_decay", 1e-5)
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )
    
    # Prepare training components with Accelerator
    vit, siamese, optimizer, train_loader, val_loader = accelerator.prepare(
        vit, siamese, optimizer, train_loader, val_loader
    )
    
    # Training history
    history = {"train_loss": [], "val_loss": [], "val_accuracy": []}
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    early_stopping_patience = training_config.get("early_stopping_patience", 5)
    
    # Training loop
    num_epochs = training_config.get("epochs", 10)
    accelerator.print(f"Starting training for {num_epochs} epochs...")
    
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
            # Forward pass
            with accelerator.accumulate(vit, siamese):
                feat1, feat2 = vit(img1), vit(img2)
                preds = siamese(feat1, feat2)
                loss = criterion(preds, label)
                
                # Update loss statistics
                total_loss += loss.item() * img1.size(0)
                train_samples += img1.size(0)
                
                # Backward pass and optimization
                accelerator.backward(loss)
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)  # More efficient than zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
            
            # Periodically free up memory
            if i % 10 == 0:
                optimize_memory()
        
        # Calculate epoch stats
        epoch_loss = total_loss / train_samples
        history["train_loss"].append(epoch_loss)
        
        # Validation phase
        vit.eval()
        siamese.eval()
        val_loss = 0
        correct = 0
        val_samples = 0
        
        with torch.no_grad():
            for img1, img2, label in tqdm(val_loader, desc="Validation"):
                feat1, feat2 = vit(img1), vit(img2)
                preds = siamese(feat1, feat2)
                
                # Calculate loss
                batch_loss = criterion(preds, label)
                val_loss += batch_loss.item() * img1.size(0)
                
                # Calculate accuracy
                predicted = (preds > 0.5).float()
                correct += (predicted == label).sum().item()
                val_samples += img1.size(0)
        
        # Calculate validation stats
        val_loss = val_loss / val_samples
        val_accuracy = correct / val_samples
        history["val_loss"].append(val_loss)
        history["val_accuracy"].append(val_accuracy)
        
        # Update learning rate
        scheduler.step(val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - start_time
        
        # Print epoch results
        accelerator.print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint
        if accelerator.is_local_main_process:
            # Unwrap models before saving
            unwrapped_vit = accelerator.unwrap_model(vit)
            unwrapped_siamese = accelerator.unwrap_model(siamese)
            
            checkpoint = {
                'epoch': epoch + 1,
                'vit_state_dict': unwrapped_vit.state_dict(),
                'siamese_state_dict': unwrapped_siamese.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': epoch_loss,
                'val_loss': val_loss,
                'val_accuracy': val_accuracy,
            }
            
            # Save latest model
            torch.save(checkpoint, os.path.join(paths_config.get("checkpoints"), "latest_model.pt"))
            
            # Early stopping check
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                # Save best model
                torch.save(checkpoint, os.path.join(paths_config.get("checkpoints"), "best_model.pt"))
                accelerator.print(f"New best validation loss: {best_val_loss:.4f}. Model saved.")
            else:
                patience_counter += 1
                accelerator.print(f"Validation loss did not improve. Patience: {patience_counter}/{early_stopping_patience}")
                
                if patience_counter >= early_stopping_patience:
                    accelerator.print("Early stopping triggered!")
                    break
            
            # Save training history
            with open(os.path.join(paths_config.get("logs"), "training_history.json"), "w") as f:
                json.dump(history, f)
            
            # Plot learning curves
            plt.figure(figsize=(12, 4))
            
            plt.subplot(1, 2, 1)
            plt.plot(history["train_loss"], label="Train Loss")
            plt.plot(history["val_loss"], label="Validation Loss")
            plt.xlabel("Epoch")
            plt.ylabel("Loss")
            plt.legend()
            plt.title("Loss Curves")
            
            plt.subplot(1, 2, 2)
            plt.plot(history["val_accuracy"], label="Validation Accuracy")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.legend()
            plt.title("Accuracy Curves")
            
            plt.tight_layout()
            plt.savefig(os.path.join(paths_config.get("logs"), "learning_curves.png"))
            plt.close()
        
        # Clean up memory before next epoch
        optimize_memory()
    
    accelerator.print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    accelerator.print(f"Models saved to {paths_config.get('checkpoints')}")
    
    return history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Accelerated training for Copy-Move Forgery Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    accelerate_train(args.config)
