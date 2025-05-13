import torch
import torch.nn as nn
import torch.nn.functional as F

class SelfAttention(nn.Module):
    """Self-attention module to enhance feature representation"""
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        
    def forward(self, x):
        q = self.query(x)
        k = self.key(x)
        v = self.value(x)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        
        out = attn @ v
        return self.out(out)

class ContrastiveModule(nn.Module):
    """Module for learning contrastive features between pairs"""
    def __init__(self, feature_dim):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(feature_dim, feature_dim // 2),
            nn.LayerNorm(feature_dim // 2),
            nn.GELU(),
            nn.Linear(feature_dim // 2, feature_dim // 2)
        )
        
    def forward(self, x1, x2):
        # Project features to contrastive space
        p1 = self.projector(x1)
        p2 = self.projector(x2)
        
        # Normalize for cosine similarity
        z1 = F.normalize(p1, p=2, dim=1)
        z2 = F.normalize(p2, p=2, dim=1)
        
        return z1, z2

class RelationModule(nn.Module):
    """Module to learn relationship between feature pairs"""
    def __init__(self, feature_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, feature_dim)
        )
        
    def forward(self, x1, x2):
        concat = torch.cat([x1, x2], dim=1)
        return self.net(concat)

class SiameseNetwork(nn.Module):
    def __init__(self, feature_dim=768, hidden_dims=[512, 256, 128], dropout_rates=[0.3, 0.3, 0.2]):
        super().__init__()
        
        # Feature adaptation layer enhanced with self-attention
        self.feature_adapter = nn.Sequential(
            nn.Linear(feature_dim, feature_dim),
            nn.LayerNorm(feature_dim),
            nn.GELU()  # GELU activation works well with transformer features
        )
        
        # Self-attention module to enhance feature representation
        self.self_attention = SelfAttention(feature_dim)
        
        # Contrastive module to learn better representations
        self.contrastive = ContrastiveModule(feature_dim)
        
        # Relation module to model relationships between features
        self.relation = RelationModule(feature_dim)
        
        # Multiple similarity computation methods
        self.compute_l1_distance = True
        self.compute_l2_distance = True
        self.compute_cosine_sim = True
        self.compute_correlation = True
        
        # Calculate input dimension for the classifier network
        combined_dim = feature_dim * 2  # Concatenated features
        if self.compute_l1_distance:
            combined_dim += feature_dim  # Add L1 distance features
        if self.compute_l2_distance:
            combined_dim += feature_dim  # Add L2 distance features
        if self.compute_cosine_sim:
            combined_dim += 1  # Add cosine similarity score
        if self.compute_correlation:
            combined_dim += 1  # Add correlation score
        
        # Add relation features
        combined_dim += feature_dim
        
        # Build multi-layer classifier with configurable dimensions and dropout
        layers = []
        in_dim = combined_dim
        
        for i, (out_dim, drop_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            layers.append(nn.Linear(in_dim, out_dim))
            layers.append(nn.BatchNorm1d(out_dim))  # Add batch normalization
            layers.append(nn.GELU() if i == 0 else nn.ReLU())  # GELU for first layer
            layers.append(nn.Dropout(drop_rate))
            in_dim = out_dim
            
        # Final classification layer with NO sigmoid - will be handled by loss function
        # This prevents double sigmoid issues with BCEWithLogitsLoss
        layers.append(nn.Linear(in_dim, 1))
        
        self.fc = nn.Sequential(*layers)

    def compute_correlations(self, x1, x2):
        """Compute correlation between feature vectors"""
        # Center the features
        x1_centered = x1 - x1.mean(dim=1, keepdim=True)
        x2_centered = x2 - x2.mean(dim=1, keepdim=True)
        
        # Compute correlation
        corr_num = torch.sum(x1_centered * x2_centered, dim=1, keepdim=True)
        corr_den = torch.sqrt(torch.sum(x1_centered**2, dim=1, keepdim=True) * 
                              torch.sum(x2_centered**2, dim=1, keepdim=True))
        corr_den = torch.clamp(corr_den, min=1e-8)  # Avoid division by zero
        
        return corr_num / corr_den

    def forward(self, feat1, feat2):
        # Apply feature adaptation
        f1 = self.feature_adapter(feat1)
        f2 = self.feature_adapter(feat2)
        
        # Apply self-attention
        f1_att = f1 + self.self_attention(f1.unsqueeze(1)).squeeze(1)
        f2_att = f2 + self.self_attention(f2.unsqueeze(1)).squeeze(1)
        
        # Get contrastive features
        z1, z2 = self.contrastive(f1_att, f2_att)
        
        # Get relation features
        relation_features = self.relation(f1_att, f2_att)
        
        # Combine features for the decision network
        combined_features = [torch.cat([f1_att, f2_att], dim=1)]
        
        # Add L1 distance if enabled
        if self.compute_l1_distance:
            l1_distance = torch.abs(f1_att - f2_att)
            combined_features.append(l1_distance)
        
        # Add L2 distance if enabled
        if self.compute_l2_distance:
            l2_distance = torch.sqrt(torch.sum((f1_att - f2_att)**2, dim=1, keepdim=True).expand(-1, f1_att.size(1)) + 1e-8)
            combined_features.append(l2_distance)
        
        # Add cosine similarity if enabled
        if self.compute_cosine_sim:
            cosine_sim = torch.sum(z1 * z2, dim=1, keepdim=True)
            combined_features.append(cosine_sim)
            
        # Add correlation if enabled
        if self.compute_correlation:
            correlation = self.compute_correlations(f1_att, f2_att)
            combined_features.append(correlation)
            
        # Add relation features
        combined_features.append(relation_features)
        
        # Combine all features
        x = torch.cat(combined_features, dim=1)
        
        # Pass through classifier network WITHOUT sigmoid activation
        # BCEWithLogitsLoss will add sigmoid internally for better numerical stability
        return self.fc(x)
