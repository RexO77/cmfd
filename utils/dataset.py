import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.transforms import get_transforms
from PIL import Image

def is_forged(mask_path):
    """Check if the mask contains forgery (non-zero pixels)"""
    if not os.path.exists(mask_path):
        return False
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return False
    return np.sum(mask) > 0  # Determine if forgery exists

class CMFDataset(Dataset):
    def __init__(self, root_dir, split='train', num_pairs_per_set=10): # Changed num_pairs logic, added split
        self.samples = []
        # Determine training mode based on split for transforms
        training = (split == 'train') 
        self.transforms = get_transforms(training=training)
        
        # Construct the path for the specific split
        split_dir = os.path.join(root_dir, split)

        # Validate directory existence
        if not os.path.exists(split_dir):
            raise FileNotFoundError(f"Dataset directory {split_dir} not found!")
            
        folders = [f for f in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, f))]
        
        # Collect sample pairs
        print(f"Loading dataset from {split_dir}...")
        if not folders:
             print(f"Warning: No subdirectories found in {split_dir}") # Added warning

        for f in folders:
            folder = os.path.join(split_dir, f)
            
            original_img_path = os.path.join(folder, f"{f}_O.png")
            mask_path = os.path.join(folder, f"{f}_M.png")
            forged_imgs = [os.path.join(folder, i) for i in os.listdir(folder) 
                           if '_F' in i and i.lower().endswith(('.png', '.jpg', '.jpeg'))]

            # Ensure original image exists
            if not os.path.exists(original_img_path):
                print(f"Warning: Original image {original_img_path} not found for set {f}. Skipping.")
                continue
                
            # Ensure mask exists to determine label (though we might override)
            if not os.path.exists(mask_path):
                 print(f"Warning: Mask {mask_path} not found for set {f}. Assuming label 1 for forged images.")

            label = 1 # Label is 1 if we compare Original vs Forged

            # Create pairs: Original vs Forged
            set_pairs = []
            if forged_imgs:
                for forged_img_path in forged_imgs:
                    set_pairs.append((original_img_path, forged_img_path, label))
            
            # Limit pairs per set if needed
            if len(set_pairs) > num_pairs_per_set:
                set_pairs = random.sample(set_pairs, num_pairs_per_set)
                
            self.samples.extend(set_pairs)
        
        if not self.samples:
             print(f"Warning: No valid image pairs found in {split_dir}. Check file naming and structure (_O.png, _F*.png).")

        random.shuffle(self.samples)  # Shuffle the sample pairs
        print(f"Loaded {len(self.samples)} image pairs from {split_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.samples[idx]
        
        # Handle corrupted images
        try:
            # Read images as RGB using PIL instead of OpenCV
            img1 = Image.open(img1_path).convert('RGB')
            img2 = Image.open(img2_path).convert('RGB')
            
            # Apply transforms (which expect PIL images)
            im1 = self.transforms(img1)
            im2 = self.transforms(img2)
            
            return im1, im2, torch.tensor([label], dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading image pair: {img1_path}, {img2_path}, {e}")
            # Return a placeholder if processing fails
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), torch.tensor([0], dtype=torch.float32)
