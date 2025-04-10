import torch
import torch.nn as nn
from torchvision.models.vision_transformer import vit_b_16
import warnings

class ViTEncoder(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        # Suppress warnings about weights being downloaded
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.model = vit_b_16(pretrained=pretrained)
        self.model.heads = nn.Identity()  # Remove classification head to use as feature extractor
        
        # Freeze earlier layers to reduce memory usage and improve performance
        for param in list(self.model.parameters())[:-4]:
            param.requires_grad = False

    def forward(self, x):
        return self.model(x)  # Return the feature representation
