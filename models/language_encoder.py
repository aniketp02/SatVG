"""
Language Encoder for the TransVG model
Uses BERT for text encoding
"""

import torch
import torch.nn as nn
from transformers import BertModel, BertConfig


class LanguageEncoder(nn.Module):
    """
    Language Encoder for TransVG
    Based on BERT with a transformer on top
    """
    def __init__(self, config):
        super().__init__()
        
        # Initialize BERT
        bert_config = BertConfig.from_pretrained(config.bert_model)
        self.bert = BertModel.from_pretrained(config.bert_model, config=bert_config)
        
        # Freeze BERT parameters (optional, can be configured)
        # for param in self.bert.parameters():
        #     param.requires_grad = False
        
        # Linear projection to match vision features dimension (vision_width)
        # This ensures dimensions match for cross-attention
        self.proj = nn.Linear(bert_config.hidden_size, config.vision_width)
        
        # Store vision width for debugging
        self.vision_width = config.vision_width
        
        # Transformer layers on top of BERT
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=config.vision_width,
            nhead=config.n_heads,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(transformer_layer, num_layers=config.cross_layers)
    
    def forward(self, text_tokens, attention_mask=None):
        """
        Args:
            text_tokens: Tokenized text input (B, L)
            attention_mask: Attention mask for text (B, L)
            
        Returns:
            projected_embeddings: Language features (B, L, vision_width)
            attention_mask: Attention mask (B, L)
        """
        # Get BERT embeddings
        bert_outputs = self.bert(input_ids=text_tokens, attention_mask=attention_mask)
        embeddings = bert_outputs.last_hidden_state  # (B, L, bert_hidden_size)
        
        # Project to match vision features dimension
        projected_embeddings = self.proj(embeddings)  # (B, L, vision_width)
        
        # Apply transformer layers
        # For transformer, True = KEEP the position, False = MASK the position
        # Original attention_mask: 1 = attend, 0 = don't attend
        key_padding_mask = None
        if attention_mask is not None:
            key_padding_mask = ~(attention_mask.bool())  # Invert for PyTorch transformer
        
        transformer_output = self.transformer(projected_embeddings, src_key_padding_mask=key_padding_mask)
        
        # Debug print - remove after debugging
        # print(f"Language features shape: {transformer_output.shape}, expected dim: {self.vision_width}")
        
        return transformer_output, attention_mask 