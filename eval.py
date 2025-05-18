"""
Evaluation and visualization script for TransVG model
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transvg import build_model
from models.losses import box_cxcywh_to_xyxy
from temp.dataloader import build_dataloaders
from utils.logger import get_logger
from utils.metrics import calculate_metrics
from configs.model_config import ModelConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Evaluate TransVG model")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"], help="Dataset split to evaluate on")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--output_dir", type=str, default="visualizations", help="Directory to save visualizations")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for evaluation")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use (cuda or cuda:0, cuda:1, etc.)")
    parser.add_argument("--include_text", action="store_true", help="Include query text in the evaluation")
    
    return parser.parse_args()


def visualize_prediction(img, text, pred_box, target_box, image_id, output_path, pred_box_cxcywh=None):
    """
    Visualize image with predicted and ground truth boxes
    
    Args:
        img: Input image tensor
        text: Text query
        pred_box: Predicted bounding box [x1, y1, x2, y2]
        target_box: Target bounding box [x1, y1, x2, y2]
        image_id: Image identifier
        output_path: Path to save visualization
        pred_box_cxcywh: Original predicted box in [cx, cy, w, h] format before conversion
    """
    # Convert tensor to numpy array
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Create figure with larger dimensions to accommodate annotations outside image
    fig, ax = plt.subplots(1, figsize=(16, 16))
    
    # Calculate the range needed for axes based on the box coordinates
    pred_box_np = pred_box.cpu().numpy()
    target_box_np = target_box.cpu().numpy()
    
    # The image is always 224x224, so normalize ground truth to this size if needed
    # Check if ground truth box goes beyond 224x224
    image_size = 224
    if np.any(target_box_np > image_size):
        # Calculate the scaling factor to bring target box within bounds
        scale_x = image_size / max(target_box_np[0], target_box_np[2])
        scale_y = image_size / max(target_box_np[1], target_box_np[3])
        scale = min(scale_x, scale_y, 1.0)  # Don't upscale, only downscale if needed
        
        if scale < 1.0:
            print(f"Warning: Ground truth box exceeds image bounds for {image_id}. Scaling by {scale:.3f}")
            target_box_np = target_box_np * scale
    
    # Find min and max of all coordinates to set plot limits
    all_coords = np.concatenate([pred_box_np, target_box_np])
    min_x, min_y = np.min(all_coords.reshape(-1, 2), axis=0) - 20
    max_x, max_y = np.max(all_coords.reshape(-1, 2), axis=0) + 20
    
    # Ensure original image is still fully visible
    min_x = min(min_x, -10)
    min_y = min(min_y, -10)
    max_x = max(max_x, image_size + 10)
    max_y = max(max_y, image_size + 10)
    
    # Display the image
    ax.imshow(img_np, extent=[0, image_size, image_size, 0])  # Extent ensures image coordinates are correct
    
    # Set extended limits to show boxes outside the image
    ax.set_xlim(min_x, max_x)
    ax.set_ylim(max_y, min_y)  # Invert y-axis for image coordinates
    
    # Draw image border for reference
    ax.plot([0, image_size, image_size, 0, 0], [0, 0, image_size, image_size, 0], 'k-', linewidth=1)
    
    # Use specific colors for each coordinate representation
    xyxy_color = 'red'         # For [x1, y1, x2, y2] representation
    cxcywh_scaled_color = 'purple'   # For scaled [cx, cy, w, h] representation
    cxcywh_unscaled_color = 'orange' # For unscaled [cx, cy, w, h] representation
    gt_color = 'green'        # For ground truth
    
    # Draw predicted box (x1, y1, x2, y2) in red
    pred_rect = patches.Rectangle(
        (pred_box_np[0], pred_box_np[1]),
        pred_box_np[2] - pred_box_np[0],
        pred_box_np[3] - pred_box_np[1],
        linewidth=3,
        edgecolor=xyxy_color,
        facecolor='none',
        label='Pred [x1,y1,x2,y2]'
    )
    ax.add_patch(pred_rect)
    
    # Draw target box (green)
    target_rect = patches.Rectangle(
        (target_box_np[0], target_box_np[1]),
        target_box_np[2] - target_box_np[0],
        target_box_np[3] - target_box_np[1],
        linewidth=3,
        edgecolor=gt_color,
        facecolor='none',
        label='Ground Truth'
    )
    ax.add_patch(target_rect)
    
    # If we have the original [cx, cy, w, h] coordinates before conversion
    if pred_box_cxcywh is not None:
        # Get unscaled [cx, cy, w, h] values
        cx, cy, w, h = pred_box_cxcywh.cpu().numpy()
        
        # Calculate scaled [cx, cy, w, h] values
        scaled_cx, scaled_cy, scaled_w, scaled_h = cx * image_size, cy * image_size, w * image_size, h * image_size
        
        # Draw the scaled [cx, cy, w, h] bounding box in purple
        # Convert to [x1, y1, w, h] for Rectangle
        x1_cxcywh = scaled_cx - scaled_w / 2
        y1_cxcywh = scaled_cy - scaled_h / 2
        
        cxcywh_rect = patches.Rectangle(
            (x1_cxcywh, y1_cxcywh),
            scaled_w,
            scaled_h,
            linewidth=3,
            edgecolor=cxcywh_scaled_color,
            linestyle='--',
            facecolor='none',
            label='Pred [cx,cy,w,h] scaled'
        )
        ax.add_patch(cxcywh_rect)
        
        # Draw unscaled center as a cross
        ax.plot(cx * image_size, cy * image_size, 'x', color=cxcywh_unscaled_color, markersize=10, markeredgewidth=3)
        
        # Draw box with unscaled normalized coordinates (scaled to image size)
        unscaled_rect = patches.Rectangle(
            (cx * image_size - w * image_size / 2, cy * image_size - h * image_size / 2),
            w * image_size,
            h * image_size,
            linewidth=2,
            edgecolor=cxcywh_unscaled_color,
            linestyle=':',
            facecolor='none',
            label='Pred [cx,cy,w,h] unscaled'
        )
        ax.add_patch(unscaled_rect)
    
    # Add query text
    ax.set_title(f"Query: {text}", fontsize=16)
    
    # Display coordinate values with appropriate colors
    if pred_box_cxcywh is not None:
        # Unscaled [cx, cy, w, h] in orange
        cx, cy, w, h = pred_box_cxcywh.cpu().numpy()
        ax.text(
            10, max_y - 25, 
            f"Pred [cx,cy,w,h] (unscaled): [{cx:.4f}, {cy:.4f}, {w:.4f}, {h:.4f}]",
            color=cxcywh_unscaled_color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7)
        )
        
        # Scaled [cx, cy, w, h] in purple
        scaled_cx, scaled_cy, scaled_w, scaled_h = cx * image_size, cy * image_size, w * image_size, h * image_size
        ax.text(
            10, max_y - 45, 
            f"Pred [cx,cy,w,h] (scaled): [{scaled_cx:.1f}, {scaled_cy:.1f}, {scaled_w:.1f}, {scaled_h:.1f}]",
            color=cxcywh_scaled_color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7)
        )
    
    # [x1, y1, x2, y2] in red
    ax.text(
        10, max_y - 65, 
        f"Pred [x1,y1,x2,y2]: [{pred_box_np[0]:.1f}, {pred_box_np[1]:.1f}, {pred_box_np[2]:.1f}, {pred_box_np[3]:.1f}]",
        color=xyxy_color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Ground truth in green
    ax.text(
        10, max_y - 85, 
        f"GT [x1,y1,x2,y2]: [{target_box_np[0]:.1f}, {target_box_np[1]:.1f}, {target_box_np[2]:.1f}, {target_box_np[3]:.1f}]",
        color=gt_color, fontsize=12, bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Add an info box indicating image size
    ax.text(
        10, max_y - 105,
        f"Image size: {image_size}x{image_size} pixels",
        fontsize=12, bbox=dict(facecolor='white', alpha=0.7)
    )
    
    # Add legend with larger font
    ax.legend(loc='upper right', fontsize=12)
    
    # Remove axes ticks but keep border
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
    
    # Save figure with tight layout
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def evaluate(model, data_loader, device, output_dir, num_samples=5):
    """
    Evaluate model and visualize predictions
    
    Args:
        model: Model to evaluate
        data_loader: Data loader for evaluation
        device: Device to use
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    
    # Create output directory if it doesn't exist
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Collect all predictions and targets for metric calculation
    all_pred_boxes = []
    all_target_boxes = []
    
    # Collect samples for visualization
    vis_samples = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Move data to device
            img = batch['img'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            target = batch['target'].to(device)
            original_bbox = batch['original_bbox'].to(device)
            
            # Get raw query text
            query_text = batch['text'] if 'text' in batch else batch.get('query', ["Unknown query"] * len(img))
            
            # Forward pass
            pred_boxes = model(img, text_tokens, text_mask)
            
            # Save the original predictions in [cx, cy, w, h] format
            pred_boxes_cxcywh = pred_boxes.clone()
            
            # Convert normalized [cx, cy, w, h] to [x1, y1, x2, y2] for metric calculation
            pred_boxes_xyxy = box_cxcywh_to_xyxy(pred_boxes)
            
            # Scale to original image size (224 pixels in this case)
            pred_boxes_xyxy = pred_boxes_xyxy * 224
            
            # Collect predictions and targets
            all_pred_boxes.append(pred_boxes_xyxy.cpu())
            all_target_boxes.append(original_bbox.cpu())
            
            # Collect samples for visualization
            if len(vis_samples) < num_samples:
                # Get number of samples we can add from this batch
                samples_to_add = min(num_samples - len(vis_samples), len(img))
                
                for j in range(samples_to_add):
                    vis_samples.append({
                        'img': img[j],
                        'pred_box': pred_boxes_xyxy[j],
                        'target_box': original_bbox[j],
                        'image_id': batch['image_id'][j],
                        'text': query_text[j] if isinstance(query_text, list) else query_text,
                        'pred_box_cxcywh': pred_boxes_cxcywh[j]
                    })
    
    # Concatenate all predictions and targets
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_target_boxes = torch.cat(all_target_boxes, dim=0)
    
    # Calculate metrics
    metrics = calculate_metrics(all_pred_boxes, all_target_boxes)
    
    # Print metrics
    print("Evaluation metrics:")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    
    # Visualize samples
    print(f"Visualizing {len(vis_samples)} samples...")
    for i, sample in enumerate(vis_samples):
        output_path = output_dir / f"sample_{i}.png"
        visualize_prediction(
            sample['img'],
            sample['text'],
            sample['pred_box'],
            sample['target_box'],
            sample['image_id'],
            output_path,
            sample['pred_box_cxcywh']
        )
    
    return metrics


def main():
    """Main evaluation function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = ModelConfig()
    config.batch_size = args.batch_size
    
    # Set up logger
    logger = get_logger(config, "eval")
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build data loaders
    logger.info("Building data loader...")
    dataloaders = build_dataloaders(config, include_text=args.include_text)
    data_loader = dataloaders[args.split]
    logger.info(f"{args.split.capitalize()} dataset size: {len(data_loader.dataset)}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    logger.info(f"Loading checkpoint from {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    
    # Evaluate model
    logger.info(f"Evaluating model on {args.split} split...")
    metrics = evaluate(model, data_loader, device, args.output_dir, args.num_samples)
    
    # Log metrics
    logger.log_metrics(metrics, split=args.split)
    
    # Finish logging
    logger.finish()


if __name__ == "__main__":
    main() 