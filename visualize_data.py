"""
Script to visualize ground truth bounding boxes from the dataset
This helps verify that the bounding boxes are correctly loaded and transformed
"""
import os
import sys
import torch
import argparse
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from temp.dataloader import build_dataloaders
from utils.visualization import visualize_dataset_samples
from configs.model_config import ModelConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize dataset samples")
    parser.add_argument("--output_dir", type=str, default="data_visualization", help="Directory to save visualizations")
    parser.add_argument("--num_samples", type=int, default=16, help="Number of samples to visualize")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size")
    
    return parser.parse_args()


def main():
    """Main function to visualize dataset samples"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = ModelConfig()
    config.batch_size = args.batch_size
    
    # Ensure output directory exists
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Print some info
    print(f"Visualizing {args.num_samples} samples from {args.split} split")
    print(f"Output directory: {output_dir}")
    
    # Build dataloaders
    dataloaders = build_dataloaders(config)
    
    # Get the specified dataloader
    dataloader = dataloaders[args.split]
    
    # Visualize dataset samples
    visualize_dataset_samples(dataloader, output_dir, args.num_samples)
    
    print(f"Visualizations saved to {output_dir}")


if __name__ == "__main__":
    main() 