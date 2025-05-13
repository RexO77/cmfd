import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import random
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Import for JPEG compression
from io import BytesIO

class RandomGamma:
    """Apply random gamma correction to simulate lighting variations"""
    def __init__(self, gamma_range=(0.8, 1.2)):
        self.gamma_range = gamma_range
        
    def __call__(self, img):
        gamma = random.uniform(self.gamma_range[0], self.gamma_range[1])
        return TF.adjust_gamma(img, gamma)

class RandomJPEGCompression:
    """Apply random JPEG compression to simulate image compression artifacts"""
    def __init__(self, quality_range=(60, 100)):
        self.quality_range = quality_range
        
    def __call__(self, img):
        quality = random.randint(self.quality_range[0], self.quality_range[1])
        buffer = BytesIO()
        img.save(buffer, format="JPEG", quality=quality)
        buffer.seek(0)
        return Image.open(buffer)

class RandomNoise:
    """Add random noise to simulate sensor noise"""
    def __init__(self, noise_range=(0.01, 0.05)):
        self.noise_range = noise_range
        
    def __call__(self, tensor_img):
        noise_level = random.uniform(self.noise_range[0], self.noise_range[1])
        noise = torch.randn_like(tensor_img) * noise_level
        return torch.clamp(tensor_img + noise, 0, 1)

class CopyMoveAugmentationPipeline:
    """Advanced augmentation pipeline specifically for copy-move forgery detection"""
    def __init__(self, img_size=224, training=True, apply_heavy_augmentations=False):
        self.img_size = img_size
        self.training = training
        self.apply_heavy_augmentations = apply_heavy_augmentations
        
        # Base transforms for both training and validation/testing
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ImageNet stats
        ])
        
        # Training-specific transforms with various augmentations
        self.train_transform = T.Compose([
            T.Resize((int(img_size * 1.1), int(img_size * 1.1))),  # Slightly larger for random crop
            T.RandomCrop((img_size, img_size)),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply([T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05)], p=0.5),
            T.RandomApply([RandomGamma()], p=0.3),
            T.RandomApply([T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0))], p=0.2),
            T.ToTensor(),
            T.RandomApply([RandomNoise()], p=0.3),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        
        # Heavy augmentations for domain robustness
        if apply_heavy_augmentations:
            self.heavy_transform = A.Compose([
                A.Resize(height=img_size, width=img_size),
                A.OneOf([
                    A.RandomBrightnessContrast(brightness_limit=0.3, contrast_limit=0.3),
                    A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20),
                    A.CLAHE(clip_limit=4.0, p=0.7),
                ], p=0.7),
                A.OneOf([
                    A.GaussNoise(var_limit=(10.0, 50.0)),
                    A.ISONoise(color_shift=(0.01, 0.05), intensity=(0.1, 0.5)),
                    A.ImageCompression(quality_lower=60, quality_upper=100, p=0.7),
                ], p=0.7),
                A.OneOf([
                    A.GaussianBlur(blur_limit=(3, 7)),
                    A.MedianBlur(blur_limit=5),
                    A.MotionBlur(blur_limit=7),
                ], p=0.5),
                A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.05, rotate_limit=10, border_mode=cv2.BORDER_CONSTANT, p=0.5),
                ToTensorV2(),
                # Normalization will be applied separately
            ])
    
    def __call__(self, img):
        """
        Apply transforms to the image
        Args:
            img: PIL Image or numpy array
        """
        if not self.training:
            return self.base_transform(img)
            
        if self.apply_heavy_augmentations and random.random() < 0.5:
            # Apply albumentations-based heavy transforms
            if isinstance(img, Image.Image):
                img_np = np.array(img)
                augmented = self.heavy_transform(image=img_np)
                tensor = augmented['image'] / 255.0  # Normalize to 0-1
                # Apply ImageNet normalization
                return T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(tensor)
            else:
                raise TypeError("For heavy augmentations, input should be a PIL Image")
        else:
            # Apply standard PyTorch transforms
            return self.train_transform(img)
            
def get_transforms(img_size=224, training=True, apply_heavy_augmentations=True):
    """Factory function to get appropriate transforms - enhanced with optimized augmentations"""
    # Enable heavy augmentations by default for training - crucial for generalization
    # Switching to 224x224 optimal size for ViT models
    return CopyMoveAugmentationPipeline(
        img_size=img_size, 
        training=training,
        apply_heavy_augmentations=apply_heavy_augmentations if training else False
    )
            
# Specialized transforms for forgery detection specific needs
class LocalizedJPEGCompression:
    """Apply different JPEG compression levels to different parts of the image"""
    def __init__(self, quality_range_foreground=(65, 85), quality_range_background=(75, 95)):
        self.quality_range_foreground = quality_range_foreground
        self.quality_range_background = quality_range_background
        
    def __call__(self, img, mask=None):
        # If no mask provided, create a random mask
        if mask is None:
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            x1, y1 = random.randint(0, img.width // 2), random.randint(0, img.height // 2)
            x2, y2 = random.randint(x1 + img.width // 4, img.width), random.randint(y1 + img.height // 4, img.height)
            mask[y1:y2, x1:x2] = 255
            
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Apply different compression to foreground (masked area)
        fg_quality = random.randint(self.quality_range_foreground[0], self.quality_range_foreground[1])
        bg_quality = random.randint(self.quality_range_background[0], self.quality_range_background[1])
        
        # Create a mask where forgery might be
        mask_bool = mask > 0
        
        # Encode and decode with different qualities
        encode_param_fg = [int(cv2.IMWRITE_JPEG_QUALITY), fg_quality]
        encode_param_bg = [int(cv2.IMWRITE_JPEG_QUALITY), bg_quality]
        
        # Split the image
        fg = img_np.copy()
        bg = img_np.copy()
        
        # Apply compression separately
        _, encoded_fg = cv2.imencode('.jpg', cv2.cvtColor(fg, cv2.COLOR_RGB2BGR), encode_param_fg)
        _, encoded_bg = cv2.imencode('.jpg', cv2.cvtColor(bg, cv2.COLOR_RGB2BGR), encode_param_bg)
        
        decoded_fg = cv2.imdecode(encoded_fg, cv2.IMREAD_COLOR)
        decoded_bg = cv2.imdecode(encoded_bg, cv2.IMREAD_COLOR)
        
        # Convert back to RGB
        decoded_fg = cv2.cvtColor(decoded_fg, cv2.COLOR_BGR2RGB)
        decoded_bg = cv2.cvtColor(decoded_bg, cv2.COLOR_BGR2RGB)
        
        # Combine using the mask
        result = np.where(mask_bool[:, :, np.newaxis], decoded_fg, decoded_bg)
        
        return Image.fromarray(result)

class RandomLocalizedBlur:
    """Apply blur only to specific regions to simulate inconsistent processing"""
    def __init__(self, kernel_range=(3, 9), sigma_range=(0.1, 2.0)):
        self.kernel_range = kernel_range
        self.sigma_range = sigma_range
        
    def __call__(self, img, mask=None):
        # If no mask provided, create a random mask
        if mask is None:
            mask = np.zeros((img.height, img.width), dtype=np.uint8)
            x1, y1 = random.randint(0, img.width // 2), random.randint(0, img.height // 2)
            x2, y2 = random.randint(x1 + img.width // 4, img.width), random.randint(y1 + img.height // 4, img.height)
            mask[y1:y2, x1:x2] = 255
        
        # Convert PIL to numpy
        img_np = np.array(img)
        
        # Choose random kernel size (must be odd)
        kernel_size = random.randrange(self.kernel_range[0], self.kernel_range[1], 2)
        sigma = random.uniform(self.sigma_range[0], self.sigma_range[1])
        
        # Apply blur to the entire image
        blurred = cv2.GaussianBlur(img_np, (kernel_size, kernel_size), sigma)
        
        # Create a mask where forgery might be
        mask_bool = mask > 0
        
        # Combine original and blurred using the mask
        result = np.where(mask_bool[:, :, np.newaxis], blurred, img_np)
        
        return Image.fromarray(result)

def convert_to_tensor(img):
    """
    Convert an image (PIL, numpy or filepath) to a normalized tensor.
    Args:
        img: PIL Image, numpy array, or string (filepath)
    Returns:
        Normalized tensor suitable for model input
    """
    if isinstance(img, str):
        # Load image from filepath
        img = Image.open(img).convert('RGB')
    elif isinstance(img, np.ndarray):
        # Convert numpy array to PIL Image
        img = Image.fromarray(np.uint8(img))
        
    # Standard normalization for pre-trained models
    transform = T.Compose([
        T.Resize((224, 224)),  # Default size
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet stats
    ])
    
    return transform(img)

# Make the imports and classes available
__all__ = [
    'get_transforms',
    'convert_to_tensor',
    'CopyMoveAugmentationPipeline',
    'RandomGamma',
    'RandomJPEGCompression',
    'RandomNoise',
    'LocalizedJPEGCompression',
    'RandomLocalizedBlur'
]
