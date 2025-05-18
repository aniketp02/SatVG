"""
Model configuration for the TransVG model
"""

import os
from pathlib import Path

class ModelConfig:
    """Configuration for the TransVG model and related components"""
    
    # Vision Encoder
    vision_width = 768
    vision_layers = 12
    vision_patch_size = 16
    
    # Language Encoder
    lang_width = 768
    lang_layers = 12
    lang_heads = 12
    
    # Cross-Modal Encoder
    cross_layers = 4
    
    # Transformer parameters
    dim_feedforward = 2048
    
    # Prediction Head
    mlp_hidden_dim = 256
    
    # Dataset parameters
    batch_size = 32
    num_workers = 4
    
    # Training parameters
    lr = 1e-4
    lr_bert = 1e-5
    weight_decay = 1e-4
    epochs = 100
    lr_drop = 70
    
    # Paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_dir = os.path.join(base_dir, 'logs')
    checkpoint_dir = os.path.join(base_dir, 'checkpoints')
    visualization_dir = os.path.join(base_dir, 'visualizations')
    
    # Logging
    use_wandb = False
    project_name = 'sat-visual-grounding'
    
    # Loss
    giou_weight = 2.0
    l1_weight = 5.0
    
    def __init__(self):
        # Data settings
        self.data_root = '/home/pokle/Trans-VG/visual_grounding/dior-rsvg'
        self.image_size = 224
        self.max_text_len = 40
        
        # Model settings
        self.hidden_dim = 256
        self.bert_model = 'bert-base-uncased'
        self.dropout = 0.1
        self.num_heads = 8
        self.num_encoder_layers = 6
        self.cross_layers = 4
        self.dim_feedforward = 2048
        self.mlp_hidden_dim = 256
        
        # Vision backbone settings
        self.backbone = 'resnet50'
        self.pretrained = True
        self.freeze_backbone = True
        self.partial_freeze_vision = True
        
        # Linguistic backbone settings
        self.freeze_linguistic = False
        self.partial_freeze_linguistic = True
        
        # Training settings
        self.batch_size = 16
        self.num_workers = 4
        self.learning_rate = 0.0001
        self.weight_decay = 0.0001
        self.num_epochs = 100
        self.lr = 1e-3
        self.lr_bert = 2e-5
        self.lr_drop = 70
        self.gradient_clip_val = 1.0
        
        # Output settings
        self.output_dir = 'output'
        
        # Create directories
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        os.makedirs(self.visualization_dir, exist_ok=True) 