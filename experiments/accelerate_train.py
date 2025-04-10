# 📁 experiments/accelerate_train.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
import time
import yaml

# Import Accelerate
from accelerate import Accelerator
from accelerate.utils import set_seed

# Import project modules
from models.vit_encoder import ViTEncoder
from models.siamese import SiameseNetwork
from utils.dataset import CMFDataset
from utils.mac_utils import get_device, optimize_memory, recommend_batch_size

def load_config(config_path="config.yaml"):
    """Load configuration from YAML file"""
    if not os.path.exists(config_path):
        print(f"Config file not found at {config_path}, using defaults")
        return {}
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config

def accelerate_train(config_path="config.yaml"):
    """Train with Accelerate for optimized performance on MacBook Pro"""
    # Load configuration
    config = load_config(config_path)
    
    # Extract configuration sections with defaults
    dataset_config = config.get("dataset", {"path": "data/CoMoFoD_small_v2", "validation_split": 0.1})
    training_config = config.get("training", {
        "batch_size": None,
        "epochs": 10,
        "learning_rate": 0.0001,
        "mixed_precision": "fp16",
        "gradient_accumulation_steps": 4
    })
    paths_config = config.get("paths", {"checkpoints": "outputs/checkpoints/"})
    
    # Initialize accelerator
    accelerator = Accelerator(
        mixed_precision=training_config.get("mixed_precision", "fp16"),
        gradient_accumulation_steps=training_config.get("gradient_accumulation_steps", 4)
    )
    
    # Set seed for reproducibility
    set_seed(42)
    
    # Create output directories
    os.makedirs(paths_config.get("checkpoints", "outputs/checkpoints/"), exist_ok=True)
    
    # Determine batch size if not specified
    batch_size = training_config.get("batch_size")
    if batch_size is None:
        batch_size = recommend_batch_size()
        accelerator.print(f"Automatically determined batch size: {batch_size}")
    
    # Initialize models
    vit = ViTEncoder(pretrained=True)
    siamese = SiameseNetwork()
    
    # Setup dataset
    full_dataset = CMFDataset(dataset_config.get("path"), training=True)
    
    # Split dataset
    val_size = int(dataset_config.get("validation_split", 0.1) * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True
    )
    
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
                optimizer.zero_grad()
            
            # Update progress bar
            progress_bar.set_postfix({"loss": loss.item()})
        
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
        
        # Print epoch results
        accelerator.print(f"Epoch {epoch+1}/{num_epochs} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")
        
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
        
        # Clean up memory
        optimize_memory()
    
    accelerator.print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    return history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Accelerated training for Copy-Move Forgery Detection")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    
    args = parser.parse_args()
    accelerate_train(args.config)
