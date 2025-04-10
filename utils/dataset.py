import os
import cv2
import random
import numpy as np
import torch
from torch.utils.data import Dataset
from utils.transforms import get_transforms

def is_forged(mask_path):
    """Check if the mask contains forgery (non-zero pixels)"""
    if not os.path.exists(mask_path):
        return False
    mask = cv2.imread(mask_path, 0)
    if mask is None:
        return False
    return np.sum(mask) > 0  # Determine if forgery exists

class CMFDataset(Dataset):
    def __init__(self, root_dir, num_pairs=1000, training=True):
        self.samples = []
        self.transforms = get_transforms(training=training)
        
        # Validate directory existence
        if not os.path.exists(root_dir):
            raise FileNotFoundError(f"Dataset directory {root_dir} not found!")
            
        folders = [f for f in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, f))]
        
        # Collect sample pairs
        print(f"Loading dataset from {root_dir}...")
        for f in folders:
            folder = os.path.join(root_dir, f)
            imgs = [i for i in os.listdir(folder) if '_F' in i and (i.endswith('.png') or i.endswith('.jpg'))]
            
            # Skip if no images found
            if not imgs:
                continue
                
            mask_path = os.path.join(folder, f"{f}_M.png")
            label = int(is_forged(mask_path))
            
            # Create image pairs
            pairs = []
            for i in range(len(imgs)):
                for j in range(i+1, len(imgs)):
                    pairs.append((os.path.join(folder, imgs[i]),
                                 os.path.join(folder, imgs[j]), label))
            
            # Limit the number of pairs if too many
            if len(pairs) > num_pairs // len(folders):
                pairs = random.sample(pairs, num_pairs // len(folders))
                
            self.samples.extend(pairs)
        
        random.shuffle(self.samples)  # Shuffle the sample pairs
        print(f"Loaded {len(self.samples)} image pairs")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img1_path, img2_path, label = self.samples[idx]
        
        # Handle corrupted images
        try:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            
            if img1 is None or img2 is None:
                # Return a placeholder if image reading fails
                placeholder = np.zeros((224, 224, 3), dtype=np.uint8)
                return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), torch.tensor([0], dtype=torch.float32)
                
            img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
            img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
            
            im1 = self.transforms(img1)
            im2 = self.transforms(img2)
            
            return im1, im2, torch.tensor([label], dtype=torch.float32)
            
        except Exception as e:
            print(f"Error loading image pair: {img1_path}, {img2_path}, {e}")
            # Return a placeholder if processing fails
            return torch.zeros(3, 224, 224), torch.zeros(3, 224, 224), torch.tensor([0], dtype=torch.float32)
