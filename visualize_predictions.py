"""
Visualization script for TransVG model predictions

This script loads model checkpoints from different epochs and visualizes predictions
on the same set of examples to confirm the model is learning.
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image
import random

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transvg import build_model
from models.losses import box_cxcywh_to_xyxy
from models.custom_dataloader import DiorRsvgDataset, collate_fn
from configs.model_config import ModelConfig
from configs.dino_vit_config import DinoVitConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Visualize TransVG model predictions")
    parser.add_argument("--checkpoints", type=str, nargs='+', required=True, 
                        help="Paths to model checkpoints")
    parser.add_argument("--data_root", type=str, default="/home/pokle/Trans-VG/visual_grounding/dior-rsvg",
                        help="Path to dataset root")
    parser.add_argument("--split", type=str, default="val", 
                        choices=["train", "val", "test"], 
                        help="Dataset split to visualize")
    parser.add_argument("--num_samples", type=int, default=5, 
                        help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations", 
                        help="Directory to save visualizations")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cuda or cuda:0, cuda:1, etc.)")
    parser.add_argument("--image_size", type=int, default=384, 
                        help="Image size for model input")
    parser.add_argument("--seed", type=int, default=42, 
                        help="Random seed for reproducibility")
    
    return parser.parse_args()


def visualize_predictions(models, sample_indices, dataset, device, output_dir, epoch_labels):
    """
    Visualize model predictions across different epochs
    
    Args:
        models: List of models from different epochs
        sample_indices: List of sample indices to visualize
        dataset: Dataset for visualization
        device: Device to use
        output_dir: Directory to save visualizations
        epoch_labels: Labels for each model (e.g., epoch numbers)
    """
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up colors for different epochs (using matplotlib color cycle)
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    for idx in sample_indices:
        # Get sample
        sample = dataset[idx]
        
        # Create batch with single sample
        batch = collate_fn([sample])
        
        # Get image and ground truth box
        img = batch['img'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_mask = batch['text_mask'].to(device)
        target = batch['target'].to(device)
        
        # Original image (without normalization)
        orig_img_path = os.path.join(dataset.img_dir, sample['image_name'])
        orig_img = Image.open(orig_img_path).convert('RGB')
        orig_img = orig_img.resize((dataset.img_size, dataset.img_size))
        
        # Get ground truth box
        gt_box = box_cxcywh_to_xyxy(target[0]).cpu().numpy() if target[0].shape[0] == 4 else target[0].cpu().numpy()
        gt_box = gt_box * dataset.img_size  # Scale to image size
        
        # Get the query text
        query_text = sample['text']
        
        # Create figure with subplot for each model
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.imshow(orig_img)
        
        # Draw ground truth box
        rect_gt = patches.Rectangle(
            (gt_box[0], gt_box[1]),
            gt_box[2] - gt_box[0],
            gt_box[3] - gt_box[1],
            linewidth=2,
            edgecolor='black',
            linestyle='--',
            facecolor='none',
            label='Ground Truth'
        )
        ax.add_patch(rect_gt)
        
        # Iterate through models to get predictions
        for model_idx, model in enumerate(models):
            # Set model to evaluation mode
            model.eval()
            
            # Forward pass
            with torch.no_grad():
                output = model(img, text_tokens, text_mask)
            
            # Get predicted box
            pred_box = output[0].cpu().numpy() if output[0].shape[0] == 4 else output[0].cpu().numpy()
            pred_box = pred_box * dataset.img_size  # Scale to image size
            
            # Draw predicted box
            rect_pred = patches.Rectangle(
                (pred_box[0], pred_box[1]),
                pred_box[2] - pred_box[0],
                pred_box[3] - pred_box[1],
                linewidth=2,
                edgecolor=colors[model_idx % len(colors)],
                facecolor='none',
                label=f'Epoch {epoch_labels[model_idx]}'
            )
            ax.add_patch(rect_pred)
        
        # Add title with query text
        ax.set_title(f"Query: {query_text}", fontsize=12)
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Remove axis ticks
        ax.set_xticks([])
        ax.set_yticks([])
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f'sample_{idx}.png'), dpi=200)
        plt.close(fig)
    
    print(f"Saved visualizations to {output_dir}")


def extract_epoch_from_checkpoint(checkpoint_path):
    """Extract epoch number from checkpoint path"""
    filename = os.path.basename(checkpoint_path)
    parts = filename.split('_')
    for part in parts:
        if part.startswith('epoch'):
            return part[5:]  # Remove 'epoch' prefix
    return "unknown"


def main():
    """Main function"""
    args = parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    # Create dataset
    config = DinoVitConfig()
    config.image_size = args.image_size
    config.data_root = args.data_root
    
    dataset = DiorRsvgDataset(
        data_root=args.data_root,
        split=args.split,
        max_query_len=40,
        bert_model='bert-base-uncased',
        img_size=args.image_size,
        use_augmentation=False,
        config=config
    )
    
    # Get random sample indices
    sample_indices = random.sample(range(len(dataset)), min(args.num_samples, len(dataset)))
    
    # Load models from checkpoints
    models = []
    epoch_labels = []
    
    for checkpoint_path in args.checkpoints:
        print(f"Loading checkpoint: {checkpoint_path}")
        
        # Extract epoch number from checkpoint path
        epoch_label = extract_epoch_from_checkpoint(checkpoint_path)
        epoch_labels.append(epoch_label)
        
        # Build model
        model = build_model(config)
        
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        
        # Move model to device
        model = model.to(device)
        model.eval()
        
        models.append(model)
    
    # Visualize predictions
    visualize_predictions(models, sample_indices, dataset, device, output_dir, epoch_labels)
    
    print("Done!")


if __name__ == "__main__":
    main() 