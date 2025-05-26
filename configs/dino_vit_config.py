"""
DINO ViT Configuration for TransVG
This configuration uses the DINO ViT backbone for visual grounding
"""

from configs.model_config import ModelConfig

class DinoVitConfig(ModelConfig):
    """Configuration for TransVG with DINO ViT backbone"""
    
    def __init__(self):
        super().__init__()
        
        # Use DINO ViT as backbone
        self.vision_backbone = 'dino_vit'
        
        # Vision settings specific to ViT
        self.image_size = 384  # SWAG E2E V1 ViT-B/16 uses 384x384 input size
        
        # Adjust hidden dimension to match ViT features
        self.hidden_dim = 256
        
        # DINO specific settings
        self.pretrained = True
        self.freeze_backbone = True  # Initially freeze to test feature quality
        self.partial_freeze_vision = True  # Only unfreeze last few layers
        
        # Adjust learning rate
        self.lr = 5e-5  # Lower learning rate due to pretrained backbone
        
        # Output directory
        self.output_dir = 'output/dino_vit' 