"""
Debug Model Predictions

This script loads a trained model from a checkpoint, runs inference on a few samples,
and visualizes the results using our enhanced visualization functions.
"""

import os
import sys
import torch
import argparse
import matplotlib.pyplot as plt
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bbox_diagnostics import compare_model_bbox_outputs
from temp.dataloader import build_dataloaders
from models.transvg import build_model
from configs.model_config import ModelConfig

def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Debug model predictions")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="model_debug", help="Directory to save results")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size")
    parser.add_argument("--use_pin_memory", action="store_true", help="Use pin_memory in DataLoader")
    
    return parser.parse_args()

def debug_model_predictions(args):
    """
    Debug model predictions
    
    Args:
        args: Command line arguments
    """
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = ModelConfig()
    config.batch_size = args.batch_size
    
    # Build dataloaders
    dataloaders = build_dataloaders(config, include_text=True, use_pin_memory=args.use_pin_memory)
    dataloader = dataloaders[args.split]
    
    print(f"Loaded {len(dataloader.dataset)} samples from {args.split} split")
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    try:
        print(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        # Check what keys are in the checkpoint
        print("Checkpoint keys:", checkpoint.keys())
        
        # Try different key patterns that might exist in the checkpoint
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        elif 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
        elif 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            # Try loading the checkpoint directly
            model.load_state_dict(checkpoint)
            
        print("Checkpoint loaded successfully")
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        print("Using randomly initialized model for visualization purposes")
    
    # Set model to evaluation mode
    model.eval()
    
    # Get a batch from the dataloader
    batch = next(iter(dataloader))
    
    # Process samples
    visualizations = []
    with torch.no_grad():
        # Move inputs to device
        img = batch['img'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_mask = batch['text_mask'].to(device)
        
        # Forward pass
        predictions = model(img, text_tokens, text_mask)
        
        # Create visualizations for each sample in the batch
        for i in range(min(len(img), args.num_samples)):
            # Get sample data
            sample_img = batch['img'][i]
            text = batch['text'][i]
            target_bbox = batch['original_bbox'][i]
            pred_bbox = predictions[i].cpu()
            image_id = batch['image_id'][i]
            
            print(f"\nProcessing sample {i+1}/{args.num_samples} - {image_id}")
            print(f"Text query: {text}")
            print(f"Original bbox: {target_bbox.tolist()}")
            print(f"Predicted bbox: {pred_bbox.tolist()}")
            
            # Create comparison visualization
            output_path = compare_model_bbox_outputs(
                sample_img, text, target_bbox, pred_bbox, image_id
            )
            visualizations.append(output_path)
            print(f"Visualization saved to: {output_path}")
    
    print(f"\nAll {len(visualizations)} visualizations saved to directory: {os.path.dirname(visualizations[0])}")
    return visualizations

if __name__ == "__main__":
    args = parse_args()
    debug_model_predictions(args) 