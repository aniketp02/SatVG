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
        
        # Enhanced loss functions
        self.l1_weight = 5.0
        self.giou_weight = 2.0
        self.center_weight = 1.0
        self.use_center_loss = True  # Enable center prediction loss
        self.use_focal_loss = True   # Enable focal loss
        
        # Data augmentation settings
        self.use_augmentation = True
        self.aug_scale = True      # Random scaling
        self.aug_crop = True       # Random cropping
        self.aug_translate = True  # Random translation
        self.aug_color = True      # Color jitter
        self.aug_blur = False      # Gaussian blur (disabled by default)
        self.aug_erase = False     # Random erasing (disabled by default)
        
        # Augmentation intensity (light)
        self.aug_scale_factor = 0.1    # Scale within 10% of original size
        self.aug_brightness = 0.1      # Brightness adjustment
        self.aug_contrast = 0.1        # Contrast adjustment
        self.aug_saturation = 0.1      # Saturation adjustment
        self.aug_hue = 0.05            # Hue adjustment
        self.aug_translate_percent = 0.05  # Translation percentage 