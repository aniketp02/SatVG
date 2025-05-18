"""
Prediction Head for the TransVG model
Used for bounding box regression
"""

import torch
import torch.nn as nn


class PredictionHead(nn.Module):
    """
    Prediction Head for TransVG
    Regresses bounding box coordinates [x_center, y_center, width, height]
    """
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        
        # MLP for regression
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim // 2, 4)  # 4 for [x_center, y_center, width, height]
        )
        
        # Sigmoid activation to ensure output values between 0 and 1
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        """
        Args:
            x: Input feature vector (B, input_dim)
            
        Returns:
            bbox: Predicted bounding box coordinates (B, 4)
                 [x_center, y_center, width, height]
        """
        # Apply MLP for regression
        bbox = self.mlp(x)
        
        # Apply sigmoid to constrain values between 0 and 1
        bbox = self.sigmoid(bbox)
        
        return bbox 