"""
DINO ViT backbone for visual grounding
Implements DINO ViT as a backbone for TransVG
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vit_b_16, ViT_B_16_Weights

class DinoVitBackbone(nn.Module):
    """Vision backbone using DINO ViT for TransVG"""
    def __init__(self, config):
        super().__init__()
        
        # Initialize ViT backbone with DINO weights
        if config.pretrained:
            # SWAG E2E V1 weights require 384x384 input
            self.backbone = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1)
            # Store input size explicitly
            self.image_size = 384
        else:
            self.backbone = vit_b_16()
            self.image_size = 224  # Default size if not using pretrained
        
        # Check if config has correct image size
        if hasattr(config, 'image_size') and config.image_size != self.image_size:
            print(f"Warning: Config specifies image_size={config.image_size}, but model requires {self.image_size}.")
            print(f"Using {self.image_size}x{self.image_size} for ViT backbone.")
        
        # Handle backbone freezing strategy
        if config.freeze_backbone:
            if hasattr(config, 'partial_freeze_vision') and config.partial_freeze_vision:
                # Partially freeze the backbone - freeze encoder blocks except last N layers
                unfreeze_last_n = 4  # Number of transformer layers to keep trainable
                num_layers = len(self.backbone.encoder.layers)
                layers_to_freeze = num_layers - unfreeze_last_n
                
                # Freeze embeddings
                for param in self.backbone.conv_proj.parameters():
                    param.requires_grad = False
                
                # Freeze class token and position embeddings
                self.backbone.class_token.requires_grad = False
                self.backbone.encoder.pos_embedding.requires_grad = False
                
                # Freeze specific early layers
                for i in range(layers_to_freeze):
                    for param in self.backbone.encoder.layers[i].parameters():
                        param.requires_grad = False
                        
                print(f"Partially froze vision backbone: embeddings and first {layers_to_freeze} layers frozen, last {unfreeze_last_n} layers trainable")
            else:
                # Completely freeze the backbone
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"Completely froze vision backbone")
        
        # Get the hidden dimension from the ViT model
        # For ViT-B/16, this is 768
        self.output_dim = 768
        
        # Save original head weights
        self._original_head = self.backbone.heads
        
        # Create a forward hook to capture features before the head
        self.features = None
        self.hook = self.backbone.encoder.register_forward_hook(self._capture_features)
        
    def _capture_features(self, module, input, output):
        """Hook to capture encoder's output features"""
        self.features = output
        
    def forward(self, x):
        """
        Forward pass through DINO ViT backbone
        
        Args:
            x: Image tensor of shape (B, C, H, W)
            
        Returns:
            Features of shape (B, N, D)
            where N is the number of tokens (patch tokens + class token)
            and D is the hidden dimension (768 for ViT-B/16)
        """
        # Check input dimensions and resize if needed
        B, C, H, W = x.shape
        if H != self.image_size or W != self.image_size:
            x = F.interpolate(x, size=(self.image_size, self.image_size), mode='bilinear', align_corners=False)
            print(f"Warning: Input images resized from {H}x{W} to {self.image_size}x{self.image_size}")
        
        # Run through backbone but discard the classification output
        _ = self.backbone(x)
        
        # Return the features captured by the hook - shape is [B, num_patches+1, hidden_dim]
        # The +1 is for the class token
        return self.features 