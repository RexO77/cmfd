"""
ViT Siamese Model for Copy-Move Forgery Detection
Based on Vision Transformer with Siamese branches for similarity detection
"""

import os
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import timm

###################################
# MODEL ARCHITECTURE COMPONENTS
###################################

# 1. Vision Transformer Backbone
class ViTBackbone(nn.Module):
    def __init__(self, pretrained=True, model_name='vit_base_patch16_224', img_size=224, patch_size=16):
        super().__init__()
        # Load pretrained ViT model
        self.vit = timm.create_model(
            model_name,
            pretrained=pretrained,
            img_size=img_size,
            patch_size=patch_size,
            num_classes=0,  # Remove classification head
            global_pool=''  # No global pooling
        )

        # Get model configuration
        self.embed_dim = self.vit.embed_dim
        self.num_patches = (img_size // patch_size) ** 2

        # Enable gradient checkpointing for memory efficiency
        self.vit.gradient_checkpointing = True

    def forward(self, x):
        # Extract features from ViT
        features = self.vit(x)  # [B, num_patches+1, embed_dim]

        # Separate CLS token and patch tokens
        cls_token = features[:, 0]  # [B, embed_dim]
        patch_tokens = features[:, 1:]  # [B, num_patches, embed_dim]

        # Calculate grid size
        h = w = int(math.sqrt(patch_tokens.size(1)))

        # Reshape patch tokens to 2D feature map
        patch_tokens_2d = patch_tokens.transpose(1, 2).reshape(-1, self.embed_dim, h, w)

        return cls_token, patch_tokens, patch_tokens_2d

# 2. Siamese Branch for Similarity Detection
class SiameseBranch(nn.Module):
    def __init__(self, embed_dim=768, proj_dim=256):
        super().__init__()

        # Projection head for feature embedding
        self.projection = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.ReLU(),
            nn.Linear(512, proj_dim)
        )

        # Attention mechanism for patch relations
        self.attention = nn.MultiheadAttention(embed_dim=proj_dim, num_heads=8, batch_first=True)

    def forward(self, patch_tokens):
        # Project patch tokens
        proj_tokens = self.projection(patch_tokens)  # [B, num_patches, proj_dim]

        # L2 normalize embeddings
        norm_tokens = F.normalize(proj_tokens, p=2, dim=2)

        # Compute self-attention on projected tokens
        attn_tokens, _ = self.attention(norm_tokens, norm_tokens, norm_tokens)

        # Compute similarity matrix (dot product)
        similarity_matrix = torch.bmm(norm_tokens, norm_tokens.transpose(1, 2))  # [B, num_patches, num_patches]

        return similarity_matrix, attn_tokens

# 3. Segmentation Head
class SegmentationHead(nn.Module):
    def __init__(self, in_channels, img_size=224):
        super().__init__()

        # Upsampling network
        self.decoder = nn.Sequential(
            nn.Conv2d(in_channels, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 1, kernel_size=1),
            nn.Sigmoid()
        )

        self.img_size = img_size

    def forward(self, x):
        x = self.decoder(x)

        # Ensure output size matches the target image size
        if x.shape[-1] != self.img_size:
            x = F.interpolate(x, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)

        return x

# 4. Classification Head
class ClassificationHead(nn.Module):
    def __init__(self, in_features, num_classes=2):
        super().__init__()

        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)

###################################
# COMPLETE MODEL IMPLEMENTATION
###################################

# 5. Complete ViT+Siamese Model
class ViTSiameseModel(nn.Module):
    def __init__(self, pretrained=True, img_size=224, patch_size=16):
        super().__init__()

        # ViT backbone
        self.backbone = ViTBackbone(pretrained=pretrained, img_size=img_size, patch_size=patch_size)
        embed_dim = self.backbone.embed_dim

        # Siamese branch
        self.siamese_branch = SiameseBranch(embed_dim=embed_dim)

        # Classification head
        self.cls_head = ClassificationHead(in_features=embed_dim)

        # Segmentation head
        self.seg_head = SegmentationHead(in_channels=embed_dim, img_size=img_size)

        # Fusion module for similarity and feature maps
        self.fusion = nn.Sequential(
            nn.Conv2d(2, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 1, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        B, C, H, W = x.shape

        # Extract features from ViT backbone
        cls_token, patch_tokens, patch_tokens_2d = self.backbone(x)

        # Classification task
        cls_output = self.cls_head(cls_token)

        # Segmentation task
        seg_output = self.seg_head(patch_tokens_2d)

        # Siamese branch for similarity learning
        similarity_matrix, _ = self.siamese_branch(patch_tokens)

        # Convert similarity matrix to heatmap
        h = w = int(math.sqrt(patch_tokens.size(1)))
        sim_heatmap = similarity_matrix.mean(dim=1).reshape(B, 1, h, w)
        sim_heatmap = F.interpolate(sim_heatmap, size=(H, W), mode='bilinear', align_corners=False)

        # Fusion of segmentation and similarity maps
        fused_map = torch.cat([seg_output, sim_heatmap], dim=1)
        final_map = self.fusion(fused_map)

        return {
            'cls_output': cls_output,               # Classification output [B, 2]
            'seg_output': seg_output,               # Segmentation output [B, 1, H, W]
            'sim_matrix': similarity_matrix,        # Similarity matrix [B, num_patches, num_patches]
            'sim_heatmap': sim_heatmap,             # Similarity heatmap [B, 1, H, W]
            'final_map': final_map                  # Fused output map [B, 1, H, W]
        }
