import torch
import torch.nn as nn

class SiameseNetwork(nn.Module):
    def __init__(self, feature_dim=768):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(feature_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(0.3),  # Add dropout for regularization
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.2),  # Add dropout for regularization
            nn.Linear(128, 1),
            nn.Sigmoid()  # Output similarity score between 0 and 1
        )

    def forward(self, feat1, feat2):
        x = torch.cat([feat1, feat2], dim=1)  # Concatenate both feature vectors
        return self.fc(x)  # Pass through MLP to get similarity score
