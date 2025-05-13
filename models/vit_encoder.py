import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import ViTModel, ViTConfig
import timm
import math
from typing import List, Optional, Tuple, Union


class MultiScaleAttention(nn.Module):
    """Multi-scale attention module for detecting forgeries at different resolutions"""
    def __init__(self, embed_dim, num_heads=8, attn_drop=0.0):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        
        # Calculate attention scores with scaled dot product
        attn = (q @ k.transpose(-2, -1)) * self.scale
        
        # Apply softmax and dropout
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        # Weighted aggregation of values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        
        return x


class FeaturePyramidNetwork(nn.Module):
    """Feature pyramid network for multi-scale feature extraction"""
    def __init__(self, in_channels: List[int], out_channel: int):
        super().__init__()
        self.lateral_convs = nn.ModuleList()
        self.output_convs = nn.ModuleList()
        
        # Create lateral connections (1x1 convs)
        for in_channel in in_channels:
            self.lateral_convs.append(nn.Conv2d(in_channel, out_channel, 1))
            self.output_convs.append(nn.Conv2d(out_channel, out_channel, 3, padding=1))
            
    def forward(self, features: List[torch.Tensor]):
        """
        Forward pass through FPN
        Args:
            features: List of feature maps at different scales (fine to coarse)
        """
        results = []
        prev_features = None
        
        # Process from coarsest to finest resolution (reversed order)
        for i, (lateral_conv, output_conv) in enumerate(zip(self.lateral_convs[::-1], self.output_convs[::-1])):
            feat_level = features[-1-i]  # Start from the coarsest level
            lateral_feat = lateral_conv(feat_level)
            
            # Upsample and add features from coarser level
            if prev_features is not None:
                h, w = lateral_feat.shape[-2:]
                prev_features = F.interpolate(prev_features, size=(h, w), mode="nearest")
                lateral_feat = lateral_feat + prev_features
                
            # Apply output convolution
            output = output_conv(lateral_feat)
            
            # Save for next iteration
            prev_features = output
            results.insert(0, output)
            
        return results


class ViTEncoder(nn.Module):
    """Enhanced ViT encoder for copy-move forgery detection"""
    def __init__(
        self, 
        model_name="vit_base_patch16_224", 
        pretrained=True, 
        feature_dim=768,
        freeze_ratio=0.7,
        use_multiscale=True,
        use_intermediate_features=True,
        fpn_dims=256,
        use_mixed_precision=True  # Enable mixed precision by default
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.use_multiscale = use_multiscale
        self.use_intermediate_features = use_intermediate_features
        self.use_mixed_precision = use_mixed_precision
        
        # Load pretrained ViT model
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,  # Remove classification head
            global_pool=''   # Don't pool features
        )
        
        # Feature dimensions based on model variant
        if 'base' in model_name:
            vit_features = 768
            num_blocks = 12
        elif 'small' in model_name:
            vit_features = 512
            num_blocks = 12
        elif 'large' in model_name:
            vit_features = 1024
            num_blocks = 24
        else:
            vit_features = 768  # Default to base
            num_blocks = 12
            
        # Freeze some layers if specified
        if freeze_ratio > 0:
            num_frozen = int(freeze_ratio * len(self.backbone.blocks))
            for i, block in enumerate(self.backbone.blocks):
                if i < num_frozen:
                    for param in block.parameters():
                        param.requires_grad = False
            
            # Always freeze patch embedding and position embedding
            for param in self.backbone.patch_embed.parameters():
                param.requires_grad = False
            self.backbone.pos_embed.requires_grad = False
        
        # For extracting intermediate features
        if use_intermediate_features:
            self.selected_layers = [3, 7, 11] if num_blocks >= 12 else [2, 4, 6]
            
            # Feature pyramid for multi-scale features
            if use_multiscale:
                self.fpn = FeaturePyramidNetwork(
                    in_channels=[vit_features] * 3,  # Same dim for all transformer layers
                    out_channel=fpn_dims
                )
                
                # Final feature dimension is fpn_dims * number of selected layers
                self.output_dim = fpn_dims * len(self.selected_layers)
                
                # Projection from output_dim to feature_dim
                self.proj = nn.Sequential(
                    nn.Linear(self.output_dim, feature_dim),
                    nn.LayerNorm(feature_dim),
                    nn.GELU()
                )
            else:
                # Just use the features from the last selected layer
                self.output_dim = vit_features
                self.proj = nn.Sequential(
                    nn.Linear(self.output_dim, feature_dim),
                    nn.LayerNorm(feature_dim)
                )
        else:
            # Just use the final output
            self.output_dim = vit_features
            self.proj = nn.Sequential(
                nn.Linear(self.output_dim, feature_dim),
                nn.LayerNorm(feature_dim)
            )
            
        # Multi-scale attention for enhanced representation
        self.msa = MultiScaleAttention(feature_dim)
        
        # Positional embedding for final feature map
        self.pos_embed = nn.Parameter(torch.zeros(1, 196, feature_dim))
        
        # Initialize with positional encoding
        self._init_pos_embed()
        
    def _init_pos_embed(self):
        """Initialize positional embedding with sine-cosine positional encoding"""
        pos_embed = self._get_sincos_pos_embed(self.pos_embed.shape[-1], int(self.pos_embed.shape[-2]**0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))
        
    def _get_sincos_pos_embed(self, embed_dim, grid_size, cls_token=False):
        """
        Generate sinusoidal 2D positional embeddings
        Args:
            embed_dim: embedding dimension
            grid_size: grid size (assuming square grid)
            cls_token: whether to include cls token position
        """
        grid_h = grid_size
        grid_w = grid_size
        
        # We need to halve the embedding dimension because we'll create sin/cos pairs
        embed_dim = embed_dim // 2
        
        # Generate positional embeddings for each dimension
        pos_h = torch.arange(grid_h).float()
        pos_w = torch.arange(grid_w).float()
        
        # Create position encoding
        dim_t = torch.arange(embed_dim).float()
        dim_t = 10000 ** (2 * (dim_t // 2) / embed_dim)
        
        # Create position matrices
        pos_h = pos_h[:, None] / dim_t
        pos_w = pos_w[:, None] / dim_t
        
        # Apply sin/cos
        pos_h = torch.stack((torch.sin(pos_h), torch.cos(pos_h)), dim=2).flatten(1)
        pos_w = torch.stack((torch.sin(pos_w), torch.cos(pos_w)), dim=2).flatten(1)
        
        # Create 2D positions
        pos = torch.cat([
            pos_h[:, None, :].expand(-1, grid_w, -1).reshape(-1, embed_dim * 2),
            pos_w[None, :, :].expand(grid_h, -1, -1).reshape(-1, embed_dim * 2)
        ], dim=1)
        
        # Add cls token position if needed
        if cls_token:
            pos = torch.cat([torch.zeros(1, embed_dim * 2 * 2), pos], dim=0)
            
        # Ensure output shape matches expected embedding size
        if pos.shape[1] > self.feature_dim:
            pos = pos[:, :self.feature_dim]  # Truncate if needed
        
        return pos.numpy()
    
    def _extract_intermediate_features(self, x):
        """Extract intermediate features from transformer layers"""
        B = x.shape[0]
        features = []
        
        # Initial processing through patch embedding
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        # Process through transformer blocks, collecting features from selected layers
        for i, block in enumerate(self.backbone.blocks):
            x = block(x)
            if i in self.selected_layers:
                # Remove cls token and reshape to spatial feature map
                feat = x[:, 1:]  # remove cls token
                H = W = int(math.sqrt(feat.shape[1]))
                # Ensure tensor is contiguous before reshaping
                feat = feat.transpose(1, 2).contiguous().reshape(B, -1, H, W)
                features.append(feat)
                
        return features
    
    def _process_features(self, features):
        """Process extracted features with FPN if using multi-scale"""
        if self.use_multiscale:
            try:
                # Apply FPN to get multi-scale features
                fpn_features = self.fpn(features)
                
                # Pool global features from each level
                pooled_features = []
                for feat in fpn_features:
                    # Global average pooling
                    pooled = F.adaptive_avg_pool2d(feat, 1).flatten(1)
                    pooled_features.append(pooled)
                
                # Concatenate all scale features
                return torch.cat(pooled_features, dim=1)
            except RuntimeError as e:
                # Provide more detailed error message and use more robust fallback
                print(f"Warning: FPN processing error: {e}. Using last feature map only.")
                # Ensure the feature is contiguous and properly shaped
                last_feat = features[-1].contiguous()
                return F.adaptive_avg_pool2d(last_feat, 1).flatten(1)
        else:
            # Just use the last feature map with global pooling - ensure contiguous
            return F.adaptive_avg_pool2d(features[-1].contiguous(), 1).flatten(1)
    
    def _reshape_output(self, x):
        """Reshape output to have a spatial structure (for visualization)"""
        B = x.shape[0]
        # Convert flat features to a small feature map (useful for attention visualization)
        h = w = 14  # 14x14 feature map (can be adjusted)
        return x.reshape(B, h, w, -1).permute(0, 3, 1, 2)
        
    def extract_patch_features(self, x):
        """Extract per-patch features for detailed analysis"""
        B = x.shape[0]
        
        # Initial processing
        x = self.backbone.patch_embed(x)
        cls_token = self.backbone.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_token, x), dim=1)
        x = self.backbone.pos_drop(x + self.backbone.pos_embed)
        
        # Process through transformer blocks
        for block in self.backbone.blocks:
            x = block(x)
            
        # Remove cls token
        patch_features = x[:, 1:]  # [B, num_patches, dim]
        
        # Reshape to spatial feature map: [B, dim, H, W]
        H = W = int(math.sqrt(patch_features.shape[1]))
        return patch_features.transpose(1, 2).reshape(B, -1, H, W)
        
    def forward(self, x):
        """
        Forward pass through the ViT encoder
        Args:
            x: Input images of shape [B, C, H, W]
        Returns:
            Features of shape [B, feature_dim]
        """
        # Use autocast for mixed precision - optimized for Apple M4 Pro
        if self.use_mixed_precision and (torch.backends.mps.is_available() or torch.cuda.is_available()):
            with torch.autocast(device_type='cuda' if torch.cuda.is_available() else 'mps'):
                if self.use_intermediate_features:
                    # Extract and process intermediate features
                    features = self._extract_intermediate_features(x)
                    x = self._process_features(features)
                else:
                    # Use only the final features from ViT
                    x = self.backbone(x)
                
                # Project to the specified feature dimension
                x = self.proj(x)
                
                # Apply multi-scale attention for enhanced representation
                # First reshape to [B, HW, C] form expected by attention
                B = x.shape[0]
                x = x.unsqueeze(1).expand(-1, 196, -1)  # Expand to patch size
                
                # Add positional encoding
                x = x + self.pos_embed
                
                # Apply multi-scale attention
                x = self.msa(x)
                
                # Global pooling for final feature vector
                x = x.mean(dim=1)  # [B, feature_dim]
                
                return x
        else:
            # Regular precision mode
            if self.use_intermediate_features:
                features = self._extract_intermediate_features(x)
                x = self._process_features(features)
            else:
                x = self.backbone(x)
            
            x = self.proj(x)
            B = x.shape[0]
            x = x.unsqueeze(1).expand(-1, 196, -1)
            x = x + self.pos_embed
            x = self.msa(x)
            x = x.mean(dim=1)
            
            return x


if __name__ == "__main__":
    # Test code
    model = ViTEncoder(
        model_name="vit_base_patch16_224", 
        pretrained=True, 
        feature_dim=512,
        use_multiscale=True
    )
    
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    
    print(f"Output shape: {output.shape}")  # Should be [2, 512]
    
    # Count parameters
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable parameters: {num_params:,}")
