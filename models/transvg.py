"""
TransVG model implementation for visual grounding
This implements a more complete version of the TransVG architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, ResNet50_Weights
from transformers import BertModel, BertConfig

class VisionEncoder(nn.Module):
    """Vision encoder based on ResNet backbone with transformer layers"""
    def __init__(self, config):
        super().__init__()
        # Initialize ResNet backbone
        if config.pretrained:
            self.backbone = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            self.backbone = resnet50()
            
        # Handle backbone freezing strategy
        if config.freeze_backbone:
            if hasattr(config, 'partial_freeze_vision') and config.partial_freeze_vision:
                # Partially freeze the backbone - freeze only early layers
                # Get all children as a list to access by index
                backbone_children = list(self.backbone.children())
                
                # Freeze specific early layers (conv1, bn1, maxpool, and first 2 residual blocks)
                for i in range(6):  # First 6 modules (0-5) of ResNet
                    if i < len(backbone_children):
                        for param in backbone_children[i].parameters():
                            param.requires_grad = False
                
                print(f"Partially froze vision backbone: first 6 modules frozen, later modules trainable")
            else:
                # Completely freeze the backbone
                for param in self.backbone.parameters():
                    param.requires_grad = False
                print(f"Completely froze vision backbone")
        
        # Remove the final classification layer
        self.backbone = nn.Sequential(*list(self.backbone.children())[:-2])
        
        # Projection layer to match hidden dimension
        self.proj = nn.Conv2d(2048, config.hidden_dim, kernel_size=1)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation='relu'
        )
        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=config.num_encoder_layers
        )
        
        # Position embeddings for transformer
        self.pos_embed = nn.Parameter(torch.zeros(1, config.hidden_dim, 7, 7))
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, x):
        """
        Forward pass through vision encoder
        
        Args:
            x: Image tensor of shape (B, C, H, W)
            
        Returns:
            Vision features of shape (B, N, D)
            where N is the number of visual tokens and D is hidden_dim
        """
        # Extract features through backbone (B, 2048, H/32, W/32)
        x = self.backbone(x)
        
        # Project to hidden dimension (B, hidden_dim, H/32, W/32)
        x = self.proj(x)
        
        # Add position embeddings
        x = x + self.pos_embed
        
        # Reshape for transformer: (B, hidden_dim, H, W) -> (B, hidden_dim, H*W) -> (H*W, B, hidden_dim)
        batch_size, dim, h, w = x.shape
        x = x.flatten(2).permute(2, 0, 1)
        
        # Pass through transformer
        x = self.transformer(x)
        
        # Reshape back: (H*W, B, hidden_dim) -> (B, H*W, hidden_dim)
        x = x.permute(1, 0, 2)
        
        return x


class LanguageEncoder(nn.Module):
    """Language encoder based on BERT"""
    def __init__(self, config):
        super().__init__()
        
        # Initialize BERT model
        self.bert = BertModel.from_pretrained(config.bert_model)
        
        # Handle BERT freezing strategy
        if hasattr(config, 'freeze_linguistic') and config.freeze_linguistic:
            if hasattr(config, 'partial_freeze_linguistic') and config.partial_freeze_linguistic:
                # Freeze embeddings
                for param in self.bert.embeddings.parameters():
                    param.requires_grad = False
                
                # Freeze first N layers of BERT (8 out of 12 layers)
                unfreeze_last_n = 4  # Number of transformer layers to unfreeze
                num_layers = len(self.bert.encoder.layer)
                layers_to_freeze = num_layers - unfreeze_last_n
                
                for i in range(layers_to_freeze):
                    for param in self.bert.encoder.layer[i].parameters():
                        param.requires_grad = False
                
                print(f"Partially froze linguistic backbone: embeddings and first {layers_to_freeze} layers frozen, last {unfreeze_last_n} layers trainable")
            else:
                # Completely freeze BERT
                for param in self.bert.parameters():
                    param.requires_grad = False
                print(f"Completely froze linguistic backbone")
        
        # Projection to match hidden dimension if needed
        if self.bert.config.hidden_size != config.hidden_dim:
            self.proj = nn.Linear(self.bert.config.hidden_size, config.hidden_dim)
        else:
            self.proj = nn.Identity()
    
    def forward(self, tokens, attention_mask):
        """
        Forward pass through language encoder
        
        Args:
            tokens: Token ids of shape (B, L)
            attention_mask: Attention mask of shape (B, L)
            
        Returns:
            Language features of shape (B, L, D)
            where L is sequence length and D is hidden_dim
        """
        # Get BERT outputs
        outputs = self.bert(
            input_ids=tokens,
            attention_mask=attention_mask,
            return_dict=True
        )
        
        # Extract sequence output
        x = outputs.last_hidden_state
        
        # Project to hidden dimension if needed
        x = self.proj(x)
        
        return x


class CrossAttention(nn.Module):
    """Cross-attention module for fusing vision and language features"""
    def __init__(self, config):
        super().__init__()
        
        # Multi-head cross-attention
        self.multihead_attn = nn.MultiheadAttention(
            embed_dim=config.hidden_dim,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        # Feed-forward network
        self.linear1 = nn.Linear(config.hidden_dim, config.dim_feedforward)
        self.dropout = nn.Dropout(config.dropout)
        self.linear2 = nn.Linear(config.dim_feedforward, config.hidden_dim)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(config.hidden_dim)
        self.norm2 = nn.LayerNorm(config.hidden_dim)
        self.dropout1 = nn.Dropout(config.dropout)
        self.dropout2 = nn.Dropout(config.dropout)
        
        self.activation = F.relu
        
        # Initialize parameters with better values
        self._reset_parameters()
    
    def _reset_parameters(self):
        """Initialize parameters with Xavier/Glorot initialization"""
        nn.init.xavier_uniform_(self.linear1.weight)
        nn.init.xavier_uniform_(self.linear2.weight)
        nn.init.constant_(self.linear1.bias, 0.0)
        nn.init.constant_(self.linear2.bias, 0.0)
    
    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        """
        Forward pass through cross-attention module
        
        Args:
            tgt: Target sequence (query sequence)
            memory: Memory sequence (key/value sequence)
            tgt_mask: Target sequence mask
            memory_mask: Memory sequence mask
            
        Returns:
            Updated target sequence
        """
        # Self-attention
        tgt2 = self.multihead_attn(
            query=tgt,
            key=memory,
            value=memory,
            attn_mask=memory_mask
        )[0]
        tgt = tgt + self.dropout1(tgt2)
        tgt = self.norm1(tgt)
        
        # Feed-forward network
        tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
        tgt = tgt + self.dropout2(tgt2)
        tgt = self.norm2(tgt)
        
        return tgt


class TransVG(nn.Module):
    """
    Transformer for Visual Grounding (TransVG) model
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Vision encoder
        self.vision_encoder = VisionEncoder(config)
        
        # Language encoder
        self.language_encoder = LanguageEncoder(config)
        
        # Cross-modal encoder (vision guided by language)
        self.cross_encoder = nn.ModuleList([
            CrossAttention(config) for _ in range(config.cross_layers)
        ])
        
        # Prediction head for bounding box regression
        self.bbox_head = nn.Sequential(
            nn.Linear(config.hidden_dim, config.mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.mlp_hidden_dim, 4)  # [xmin, ymin, xmax, ymax]
        )
        
        # Global feature token (similar to CLS token in BERT)
        self.global_token = nn.Parameter(torch.zeros(1, 1, config.hidden_dim))
        
        # Initialize parameters
        self._init_parameters()
    
    def _init_parameters(self):
        """Initialize model parameters for better convergence"""
        # Initialize global token with normal distribution
        nn.init.normal_(self.global_token, mean=0.0, std=0.02)
        
        # Initialize bbox prediction head
        nn.init.xavier_uniform_(self.bbox_head[0].weight)
        nn.init.xavier_uniform_(self.bbox_head[3].weight)
        nn.init.constant_(self.bbox_head[0].bias, 0.0)
        
        # Apply a better initialization for the final layer bias
        # Initialize to predict a centered box covering about 40% of the image
        # This is a better default than predicting the entire image
        with torch.no_grad():
            # Initialize bias toward a centered box with reasonable size
            # Values are in normalized coordinates [0,1]
            self.bbox_head[3].bias.data = torch.tensor([0.3, 0.3, 0.7, 0.7])
    
    def forward(self, img, text_tokens, text_mask):
        """
        Forward pass through the model
        
        Args:
            img: Input image tensor (B, C, H, W)
            text_tokens: Text token tensor (B, L)
            text_mask: Text attention mask (B, L)
            
        Returns:
            bbox: Predicted bounding boxes in [xmin, ymin, xmax, ymax] format (B, 4)
                 in normalized coordinates [0,1]
        """
        batch_size = img.size(0)
        
        # Extract vision features
        vision_features = self.vision_encoder(img)  # (B, N_v, D)
        
        # Extract language features
        language_features = self.language_encoder(text_tokens, text_mask)  # (B, N_l, D)
        
        # Add global token to vision features
        global_tokens = self.global_token.expand(batch_size, -1, -1)  # (B, 1, D)
        vision_features = torch.cat([global_tokens, vision_features], dim=1)  # (B, 1+N_v, D)
        
        # Prepare for cross-attention
        vision_features = vision_features.transpose(0, 1)  # (1+N_v, B, D)
        language_features = language_features.transpose(0, 1)  # (N_l, B, D)
        
        # Apply cross-attention layers
        for layer in self.cross_encoder:
            # Create proper mask shape for cross-attention
            # Convert mask from [B, L] to [B, 1+N_v, L]
            vision_seq_len = vision_features.size(0)
            lang_seq_len = language_features.size(0)
            
            # We need to expand the mask to match expected dimensions
            # Original mask is [B, L] where True means attend, False means mask
            # We need to invert it and expand to [B, 1+N_v, L]
            expanded_mask = (~text_mask.bool()).unsqueeze(1).expand(-1, vision_seq_len, -1)
            
            vision_features = layer(
                tgt=vision_features,
                memory=language_features,
                memory_mask=None  # Using None for now as a temporary fix
            )
        
        # Extract global feature (first token)
        global_feature = vision_features[0]  # (B, D)
        
        # Predict bounding box coordinates in [xmin, ymin, xmax, ymax] format
        bbox = self.bbox_head(global_feature)  # (B, 4)
        
        # Apply sigmoid to get normalized coordinates in [0, 1] range
        bbox = torch.sigmoid(bbox)
        
        # Ensure xmax > xmin and ymax > ymin while staying in [0, 1] range
        xmin, ymin, xmax, ymax = bbox.unbind(-1)
        
        # Add a small epsilon to ensure width/height are positive while clamping to [0,1]
        width = (xmax - xmin).clamp(min=0.05)
        height = (ymax - ymin).clamp(min=0.05)
        
        # Ensure coordinates stay in [0,1] range by clamping
        xmin = xmin.clamp(min=0.0, max=0.95)
        ymin = ymin.clamp(min=0.0, max=0.95)
        xmax = (xmin + width).clamp(max=1.0)
        ymax = (ymin + height).clamp(max=1.0)
        
        # Stack back to bbox format
        bbox = torch.stack([xmin, ymin, xmax, ymax], dim=-1)
        
        return bbox


def build_model(config):
    """
    Build TransVG model from configuration
    
    Args:
        config: Model configuration
        
    Returns:
        TransVG model instance
    """
    model = TransVG(config)
    return model 