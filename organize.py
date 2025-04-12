import os
import shutil
from pathlib import Path
import random
import re

def organize_dataset(source_dir, dest_dir, train_ratio=0.7, val_ratio=0.15):
    # Ensure source directory exists
    if not os.path.exists(source_dir):
        print(f"Error: Source directory '{source_dir}' not found")
        return False
        
    # Create main directories
    dest_path = Path(dest_dir)
    train_path = dest_path / "train"
    val_path = dest_path / "val"
    test_path = dest_path / "test"
    
    for path in [train_path, val_path, test_path]:
        if path.exists():
            shutil.rmtree(path)  # Remove if exists to avoid mixed data
        path.mkdir(parents=True, exist_ok=True)

    # Get all image files
    all_files = [f for f in os.listdir(source_dir) if os.path.isfile(os.path.join(source_dir, f)) 
                and f.endswith(('.png', '.jpg', '.jpeg', '.tif', '.bmp'))]
    
    # Extract unique image set identifiers
    image_sets = set()
    for file in all_files:
        # Extract the base identifier (e.g., "001" from "001_O.png" or "001_F_1.png")
        match = re.match(r'(\d+)_[OMF]', file)
        if match:
            image_sets.add(match.group(1))
    
    image_sets = sorted(list(image_sets))
    
    # Check if we found any image sets
    if not image_sets:
        print(f"Error: No image sets found in '{source_dir}'")
        print(f"Files in directory: {all_files[:10]}...")
        return False
        
    print(f"Found {len(image_sets)} image sets in source directory")
    random.shuffle(image_sets)  # Shuffle for random split

    # Calculate split indices
    total_sets = len(image_sets)
    train_split = int(total_sets * train_ratio)
    val_split = int(total_sets * (train_ratio + val_ratio))

    # Split datasets
    train_sets = image_sets[:train_split]
    val_sets = image_sets[train_split:val_split]
    test_sets = image_sets[val_split:]

    print(f"Split: {len(train_sets)} train, {len(val_sets)} validation, {len(test_sets)} test sets")

    # Function to copy files for a specific set
    def copy_files(set_ids, destination):
        copied = 0
        for set_id in set_ids:
            # Create a directory for this set
            dest_folder = os.path.join(destination, set_id)
            os.makedirs(dest_folder, exist_ok=True)
            
            # Find all files for this set
            set_files = [f for f in all_files if f.startswith(f"{set_id}_")]
            
            if not set_files:
                print(f"Warning: No files found for set {set_id}")
                continue
                
            # Check for required files (original + mask at minimum)
            original_file = f"{set_id}_O.png"
            mask_file = f"{set_id}_M.png"
            
            if original_file not in set_files or mask_file not in set_files:
                print(f"Warning: Set {set_id} missing required files, skipping")
                continue
                
            # Copy all related files
            for file in set_files:
                src_file = os.path.join(source_dir, file)
                dest_file = os.path.join(dest_folder, file)
                shutil.copy2(src_file, dest_file)
            copied += 1
        return copied

    # Copy files to respective directories
    train_copied = copy_files(train_sets, train_path)
    val_copied = copy_files(val_sets, val_path)
    test_copied = copy_files(test_sets, test_path)

    print(f"Dataset organized into {dest_dir}")
    print(f"Train sets: {train_copied}")
    print(f"Validation sets: {val_copied}")
    print(f"Test sets: {test_copied}")
    
    return train_copied > 0 and val_copied > 0 and test_copied > 0

# Function to verify dataset structure
def verify_dataset(dataset_dir):
    for split in ['train', 'val', 'test']:
        split_dir = os.path.join(dataset_dir, split)
        if not os.path.exists(split_dir):
            print(f"Error: {split} directory not found")
            return False
        
        image_sets = os.listdir(split_dir)
        if not image_sets:
            print(f"Error: No image sets found in {split} directory")
            return False
        
        # Check a random image set
        random_set = random.choice(image_sets)
        set_path = os.path.join(split_dir, random_set)
        files = os.listdir(set_path)
        
        if f"{random_set}_O.png" not in files or f"{random_set}_M.png" not in files:
            print(f"Error: Required files missing in {set_path}")
            return False
            
        forged_files = [f for f in files if f.startswith(f"{random_set}_F_")]
        if not forged_files:
            print(f"Error: No forged images found in {set_path}")
            return False
    
    print("Dataset structure verified successfully")
    return True

# Usage
if __name__ == "__main__":
    source_directory = "data/CoMoFoD_small_v2"  # Adjust this to your source directory
    destination_directory = "data/organized_dataset"
    
    # Check if source directory exists
    if not os.path.exists(source_directory):
        print(f"Error: Source directory '{source_directory}' does not exist")
    else:
        print(f"Organizing dataset from {source_directory} to {destination_directory}...")
        success = organize_dataset(source_directory, destination_directory)
        if success:
            verify_dataset(destination_directory)
