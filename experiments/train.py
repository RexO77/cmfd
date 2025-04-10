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

def train(config=None):
    # Default configuration
    if config is None:
        config = {
            "dataset": "data/CoMoFoD_small_v2",
            "batch_size": None,  # Will be automatically set
            "epochs": 10,
            "learning_rate": 0.0001,
            "save_path": "outputs/checkpoints/",
            "log_interval": 10,
            "validation_split": 0.1,
            "early_stopping_patience": 5,
            "weight_decay": 1e-5
        }
    
    # Create output directories
    os.makedirs(config["save_path"], exist_ok=True)
    os.makedirs("logs", exist_ok=True)
    
    # Print system information
    mac_info = get_mac_info()
    print(f"System Information:")
    for key, value in mac_info.items():
        print(f"  {key}: {value}")
    
    # Set PyTorch threads for CPU operations
    num_threads = set_pytorch_threads()
    print(f"PyTorch using {num_threads} CPU threads")
    
    # Determine batch size if not specified
    if config["batch_size"] is None:
        config["batch_size"] = recommend_batch_size()
        print(f"Automatically determined batch size: {config['batch_size']}")
    
    # Get device - MPS (Apple GPU) or CPU
    device = get_device()
    print(f"Using device: {device}")
    
    # Initialize models
    print("Initializing models...")
    vit = ViTEncoder(pretrained=True)
    siamese = SiameseNetwork()
    
    # Move models to device
    vit.to(device)
    siamese.to(device)
    
    # Load dataset
    print(f"Loading dataset from {config['dataset']}...")
    try:
        full_dataset = CMFDataset(config["dataset"], training=True)
        
        # Split into training and validation
        val_size = int(config["validation_split"] * len(full_dataset))
        train_size = len(full_dataset) - val_size
        train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=config["batch_size"], 
            shuffle=True,
            num_workers=min(4, os.cpu_count() or 1),  # Use multiple workers but limit to avoid system slowdown
            pin_memory=True  # This speeds up data transfer to GPU
        )
        
        val_loader = DataLoader(
            val_dataset, 
            batch_size=config["batch_size"], 
            shuffle=False,
            num_workers=min(4, os.cpu_count() or 1),
            pin_memory=True
        )
        
        print(f"Dataset loaded with {len(train_dataset)} training and {len(val_dataset)} validation samples")
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Setup loss function and optimizer
    criterion = nn.BCELoss()
    optimizer = optim.Adam(
        list(vit.parameters()) + list(siamese.parameters()), 
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"]  # Add L2 regularization
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, 
        mode='min', 
        factor=0.5, 
        patience=2,
        verbose=True
    )
    
    # Training history
    history = {
        "train_loss": [],
        "val_loss": [],
        "val_accuracy": []
    }
    
    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    
    # Training loop
    print(f"Starting training for {config['epochs']} epochs...")
    for epoch in range(config["epochs"]):
        start_time = time.time()
        
        # Training phase
        vit.train()
        siamese.train()
        total_loss = 0
        train_samples = 0
        
        for i, (img1, img2, label) in enumerate(tqdm(train_loader, desc=f"Epoch {epoch+1}/{config['epochs']}")):
            # Move data to device
            img1, img2, label = img1.to(device), img2.to(device), label.to(device)
            
            # Forward pass
            with torch.cuda.amp.autocast(enabled=False):  # MPS doesn't support AMP yet
                feat1, feat2 = vit(img1), vit(img2)
                preds = siamese(feat1, feat2)
                loss = criterion(preds, label)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Update stats
            batch_size = img1.size(0)
            total_loss += loss.item() * batch_size
            train_samples += batch_size
            
            # Print progress
            if (i + 1) % config["log_interval"] == 0:
                print(f"  Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")
            
            # Free up memory
            if i % 5 == 0:
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
                img1, img2, label = img1.to(device), img2.to(device), label.to(device)
                
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
        elapsed_time = time.time() - start_time
        print(f"Epoch {epoch+1}/{config['epochs']} - "
              f"Train Loss: {epoch_loss:.4f}, "
              f"Val Loss: {val_loss:.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}, "
              f"Time: {elapsed_time:.2f}s")
        
        # Save checkpoint
        checkpoint = {
            'epoch': epoch + 1,
            'vit_state_dict': vit.state_dict(),
            'siamese_state_dict': siamese.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'train_loss': epoch_loss,
            'val_loss': val_loss,
            'val_accuracy': val_accuracy,
        }
        
        # Save the latest model
        torch.save(checkpoint, os.path.join(config["save_path"], "latest_model.pt"))
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            # Save the best model
            torch.save(checkpoint, os.path.join(config["save_path"], "best_model.pt"))
            print(f"New best validation loss: {best_val_loss:.4f}. Model saved.")
        else:
            patience_counter += 1
            print(f"Validation loss did not improve. Patience: {patience_counter}/{config['early_stopping_patience']}")
            
            if patience_counter >= config["early_stopping_patience"]:
                print("Early stopping triggered!")
                break
        
        # Save training history
        with open("logs/training_history.json", "w") as f:
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
        plt.savefig("logs/learning_curves.png")
        plt.close()
        
        # Clean up memory before next epoch
        optimize_memory()
    
    print(f"Training completed! Best validation loss: {best_val_loss:.4f}")
    print(f"Models saved to {config['save_path']}")
    
    return history

if __name__ == "__main__":
    train()
