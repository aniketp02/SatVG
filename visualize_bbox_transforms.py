"""
Script to visualize bounding box transformations without needing a model

This script helps debug the bounding box coordinate transformations
by showing how boxes are converted between different formats.
"""

import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from temp.dataloader import build_dataloaders
from bbox_diagnostics import visualize_bbox_transformations
from configs.model_config import ModelConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize bounding box transformations")
    parser.add_argument("--output_dir", type=str, default="bbox_transforms", help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    parser.add_argument("--img_size", type=int, default=224, help="Image size")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--use_pin_memory", action="store_true", help="Use pin_memory in data loading")
    
    return parser.parse_args()


def main():
    """Main function to visualize bounding box transformations"""
    # Parse arguments
    args = parse_args()
    
    # Set the device explicitly
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = ModelConfig()
    config.batch_size = args.batch_size
    config.image_size = args.img_size
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print some info
    print(f"Visualizing bounding box transformations for {args.num_samples} samples from {args.split} split")
    print(f"Output directory: {output_dir}")
    
    # Build dataloaders with text included for better visualization
    dataloaders = build_dataloaders(config, include_text=True, use_pin_memory=args.use_pin_memory)
    
    # Get the specified dataloader
    dataloader = dataloaders[args.split]
    
    # Get a batch of data
    batch = next(iter(dataloader))
    
    # Visualize bounding box transformations for each sample
    vis_paths = []
    for i in range(min(len(batch['img']), args.num_samples)):
        img = batch['img'][i]
        bbox = batch['original_bbox'][i]
        image_id = batch['image_id'][i]
        text = batch['text'][i]
        
        # Visualize transformations
        output_path = visualize_bbox_transformations(img, bbox, config.image_size)
        vis_paths.append(output_path)
        
        print(f"Visualized transformations for sample {i+1}/{args.num_samples} - {image_id}")
        print(f"  Query: {text}")
        print(f"  Original bbox: {bbox.tolist()}")
        print(f"  Output saved to: {output_path}")
        print()
    
    print(f"All visualizations saved to {output_dir}")


if __name__ == "__main__":
    main() 