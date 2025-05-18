"""
Vision Encoder for the TransVG model
Based on Vision Transformer (ViT) architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np


def get_sinusoid_encoding_table(n_position, d_hid):
    """
    Create sinusoidal position encoding table
    """
    def get_position_angle_vec(position):
        return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

    sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_position)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
    return torch.FloatTensor(sinusoid_table).unsqueeze(0)


class PatchEmbedding(nn.Module):
    """
    Convert input images to patch embeddings
    """
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        
        # Linear projection to convert patches to embed_dim
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # Position embeddings - adding +2 for CLS token and REG token
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 2, embed_dim))
        
        # Initialize position embeddings with sine-cosine encoding
        pos_embed = get_sinusoid_encoding_table(self.num_patches + 2, embed_dim)
        self.pos_embed.data.copy_(pos_embed)
        
        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # REG token (for region grounding)
        self.reg_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Initialize tokens
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.reg_token, std=0.02)
    
    def forward(self, x):
        B, C, H, W = x.shape
        
        # Extract patches and project
        x = self.proj(x)  # (B, embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        
        # Add class token and REG token
        cls_token = self.cls_token.expand(B, -1, -1)
        reg_token = self.reg_token.expand(B, -1, -1)
        
        # Concatenate tokens with patch embeddings
        x = torch.cat([cls_token, reg_token, x], dim=1)
        
        # Add position embeddings
        x = x + self.pos_embed[:, :x.size(1)]
        
        return x


class TransformerEncoderLayer(nn.Module):
    """
    Transformer Encoder Layer as used in ViT
    """
    def __init__(self, d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
    
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        # Self-attention block
        src2 = self.norm1(src)
        src2 = src2.transpose(0, 1)  # (seq_len, batch, dim) for nn.MultiheadAttention
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=src_mask, key_padding_mask=src_key_padding_mask)
        src2 = src2.transpose(0, 1)  # (batch, seq_len, dim)
        src = src + self.dropout1(src2)
        
        # Feed-forward block
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        return src


class LanguageConditionedEncoderLayer(nn.Module):
    """
    Language-conditioned Vision Encoder Layer
    Incorporates language features to guide the vision encoding
    """
    def __init__(self, d_model=768, nhead=12, dim_feedforward=3072, dropout=0.1):
        super().__init__()
        
        # Self-attention for vision features
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Cross-attention to incorporate language features
        self.cross_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
    
    def forward(self, vision_features, lang_features, vision_mask=None, lang_key_padding_mask=None):
        """
        Args:
            vision_features: Vision features (B, S, d_model)
            lang_features: Language features (B, L, hidden_dim) - Note: might have different dim than d_model
            vision_mask: Mask for vision features (optional)
            lang_key_padding_mask: Padding mask for language features (optional)
        """
        # Self-attention for vision features
        src2 = self.norm1(vision_features)
        src2 = src2.transpose(0, 1)  # (seq_len, batch, dim)
        src2, _ = self.self_attn(src2, src2, src2, attn_mask=vision_mask)
        src2 = src2.transpose(0, 1)  # (batch, seq_len, dim)
        vision_features = vision_features + self.dropout1(src2)
        
        # Cross-attention with language features
        src2 = self.norm2(vision_features)
        src2 = src2.transpose(0, 1)  # (seq_len, batch, dim)
        
        # Ensure language features have the same dimension as vision features
        # This assumes lang_features were already projected in the language encoder
        
        # Shape check and debug print - remove after debugging
        if vision_features.shape[-1] != lang_features.shape[-1]:
            print(f"Dimension mismatch: vision={vision_features.shape}, lang={lang_features.shape}")
            # We need to ensure the dimensions match
            raise ValueError(f"Dimension mismatch in cross-attention: vision={vision_features.shape[-1]}, lang={lang_features.shape[-1]}")
        
        lang_features = lang_features.transpose(0, 1)  # (seq_len, batch, dim)
        
        # Convert mask to boolean if it exists
        padded_mask = None
        if lang_key_padding_mask is not None:
            # Convert to boolean mask (1 = keep, 0 = mask)
            padded_mask = ~(lang_key_padding_mask.bool())
        
        src2, _ = self.cross_attn(src2, lang_features, lang_features, key_padding_mask=padded_mask)
        src2 = src2.transpose(0, 1)  # (batch, seq_len, dim)
        vision_features = vision_features + self.dropout2(src2)
        
        # Feed-forward
        src2 = self.norm3(vision_features)
        src2 = self.linear2(self.dropout(F.gelu(self.linear1(src2))))
        vision_features = vision_features + self.dropout3(src2)
        
        return vision_features


class VisionEncoder(nn.Module):
    """
    Vision Encoder for TransVG
    """
    def __init__(self, config):
        super().__init__()
        
        # Image to patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=config.image_size,
            patch_size=config.vision_patch_size,
            embed_dim=config.vision_width
        )
        
        # Transformer encoder layers
        self.encoder_layers = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.vision_width,
                nhead=config.n_heads,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout
            ) 
            for _ in range(config.vision_layers - 1)  # Last layer is language conditioned
        ])
        
        # Language conditioned encoder layer
        self.lang_cond_layer = LanguageConditionedEncoderLayer(
            d_model=config.vision_width,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout
        )
        
        # Project to hidden dimension (for final prediction) - we do this after language conditioning
        self.proj = nn.Linear(config.vision_width, config.hidden_dim)
    
    def forward(self, img, lang_features, lang_mask=None):
        # Get patch embeddings
        x = self.patch_embed(img)  # (B, num_patches + 2, embed_dim)
        
        # Apply transformer encoder layers
        for layer in self.encoder_layers:
            x = layer(x)
        
        # Apply language conditioned layer
        # Note: lang_features should already be projected to vision_width in the language encoder
        x = self.lang_cond_layer(x, lang_features, lang_key_padding_mask=lang_mask)
        
        # Project to hidden dimension
        x = self.proj(x)
        
        # Return the REG token (index 1, after CLS token)
        reg_token = x[:, 1]
        
        return reg_token, x 