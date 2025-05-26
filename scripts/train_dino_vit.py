"""
Training script for TransVG with DINO ViT backbone
"""

import torch
import argparse
import os
import sys

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from configs.dino_vit_config import DinoVitConfig
from models.transvg import build_model

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train TransVG with DINO ViT backbone')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint')
    parser.add_argument('--checkpoint', type=str, default=None, help='Checkpoint path')
    parser.add_argument('--batch_size', type=int, default=None, help='Batch size')
    parser.add_argument('--lr', type=float, default=None, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=None, help='Number of epochs')
    parser.add_argument('--freeze', action='store_true', help='Freeze backbone')
    parser.add_argument('--partial_freeze', action='store_true', help='Partially freeze backbone')
    args = parser.parse_args()
    
    # Create configuration
    config = DinoVitConfig()
    
    # Override config with command line arguments if provided
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.lr = args.lr
    if args.epochs:
        config.num_epochs = args.epochs
    if args.freeze:
        config.freeze_backbone = True
        config.partial_freeze_vision = False
    if args.partial_freeze:
        config.freeze_backbone = True
        config.partial_freeze_vision = True
    
    # Print memory usage warning
    print("\nNOTE: The DINO ViT model uses 384x384 image size which requires more GPU memory.")
    print("If you encounter CUDA out of memory errors, reduce the batch size.\n")
    
    # Create model
    model = build_model(config)
    
    # Print model info
    print(f"\nTransVG with DINO ViT backbone:")
    print(f"- Vision backbone: {config.vision_backbone}")
    print(f"- Image size: {config.image_size}")
    print(f"- Hidden dimension: {config.hidden_dim}")
    print(f"- Freeze backbone: {config.freeze_backbone}")
    print(f"- Partial freeze: {config.partial_freeze_vision}")
    print(f"- Batch size: {config.batch_size}")
    print(f"- Learning rate: {config.lr}")
    print(f"- Epochs: {config.num_epochs}")
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"- Total parameters: {total_params:,}")
    print(f"- Trainable parameters: {trainable_params:,}")
    print(f"- Percentage trainable: {trainable_params / total_params * 100:.2f}%\n")
    
    # Here you would add the training loop code
    # For simplicity, this example just shows the model setup
    
    print("Model initialized successfully. Ready for training.")
    
if __name__ == "__main__":
    main() 