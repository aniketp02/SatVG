"""
Script to analyze model performance issues and diagnose problems
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import argparse
from tqdm import tqdm

from models.transvg import build_model
from models.losses import box_cxcywh_to_xyxy
from utils.metrics import compute_iou
from utils.logger import get_logger
from configs.model_config import ModelConfig


def analyze_model_predictions(model, data_loader, device, output_dir, max_samples=50):
    """
    Analyze model predictions and create visualizations to diagnose issues
    
    Args:
        model: Trained model
        data_loader: Data loader containing examples to analyze
        device: Device to run model on
        output_dir: Output directory for visualizations
        max_samples: Maximum number of samples to analyze
        
    Returns:
        Dictionary of analysis results
    """
    model.eval()
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Analysis metrics
    all_ious = []
    all_center_errors = []
    all_size_errors = []
    good_examples = []
    bad_examples = []
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(data_loader)):
            # Limit number of samples
            if i >= max_samples:
                break
                
            # Move data to device
            img = batch['img'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            target = batch['target'].to(device)
            original_bbox = batch['original_bbox'].to(device)
            
            # Get image IDs if available
            image_ids = batch.get('image_id', [f"sample_{i}_{j}" for j in range(len(img))])
            
            # Forward pass - model now outputs directly in [xmin, ymin, xmax, ymax] format
            pred_boxes = model(img, text_tokens, text_mask)
            
            # Scale to original image size
            image_size = img.shape[2]
            pred_boxes_scaled = pred_boxes * image_size
            
            # Calculate IoU for each example
            for j in range(len(img)):
                pred_box = pred_boxes_scaled[j].cpu()
                target_box = original_bbox[j].cpu()
                
                # Get image and text
                image = img[j].cpu()
                
                # Calculate IoU
                iou = compute_iou(pred_box.unsqueeze(0), target_box.unsqueeze(0)).item()
                all_ious.append(iou)
                
                # Calculate center error
                pred_center_x = (pred_box[0] + pred_box[2]) / 2
                pred_center_y = (pred_box[1] + pred_box[3]) / 2
                target_center_x = (target_box[0] + target_box[2]) / 2
                target_center_y = (target_box[1] + target_box[3]) / 2
                
                center_error = np.sqrt((pred_center_x - target_center_x)**2 + 
                                       (pred_center_y - target_center_y)**2)
                all_center_errors.append(center_error)
                
                # Calculate size error
                pred_width = pred_box[2] - pred_box[0]
                pred_height = pred_box[3] - pred_box[1]
                target_width = target_box[2] - target_box[0]
                target_height = target_box[3] - target_box[1]
                
                width_error = abs(pred_width - target_width) / target_width
                height_error = abs(pred_height - target_height) / target_height
                size_error = (width_error + height_error) / 2
                all_size_errors.append(size_error)
                
                # Create visualization
                create_visualization(
                    image, 
                    pred_box, 
                    target_box, 
                    iou, 
                    os.path.join(output_dir, f"sample_{i}_{j}.png")
                )
                
                # Collect good and bad examples
                example_data = {
                    'image': image,
                    'pred_box': pred_box,
                    'target_box': target_box,
                    'iou': iou,
                    'image_id': image_ids[j] if isinstance(image_ids, list) else image_ids,
                    'center_error': center_error,
                    'size_error': size_error
                }
                
                if iou > 0.7:
                    good_examples.append(example_data)
                elif iou < 0.3:
                    bad_examples.append(example_data)
    
    # Convert to numpy arrays
    all_ious = np.array(all_ious)
    all_center_errors = np.array(all_center_errors)
    all_size_errors = np.array(all_size_errors)
    
    # Create histogram plots
    create_histogram(all_ious, 'IoU', os.path.join(output_dir, 'iou_histogram.png'))
    create_histogram(all_center_errors, 'Center Error', os.path.join(output_dir, 'center_error_histogram.png'))
    create_histogram(all_size_errors, 'Size Error', os.path.join(output_dir, 'size_error_histogram.png'))
    
    # Create scatter plot of center error vs. size error
    plt.figure(figsize=(10, 8))
    plt.scatter(all_center_errors, all_size_errors, alpha=0.6)
    plt.xlabel('Center Error')
    plt.ylabel('Size Error')
    plt.title('Center Error vs. Size Error')
    plt.colorbar(label='IoU')
    plt.grid(alpha=0.3)
    plt.savefig(os.path.join(output_dir, 'error_scatter.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # Calculate statistics
    metrics = {
        'mean_iou': all_ious.mean(),
        'median_iou': np.median(all_ious),
        'acc@0.5': (all_ious >= 0.5).mean(),
        'acc@0.75': (all_ious >= 0.75).mean(),
        'mean_center_error': all_center_errors.mean(),
        'mean_size_error': all_size_errors.mean()
    }
    
    # Save metrics to file
    with open(os.path.join(output_dir, 'metrics.txt'), 'w') as f:
        f.write("Model Performance Analysis\n")
        f.write("========================\n\n")
        for key, value in metrics.items():
            f.write(f"{key}: {value:.4f}\n")
    
    return {
        'metrics': metrics,
        'good_examples': good_examples,
        'bad_examples': bad_examples
    }


def create_visualization(image, pred_box, target_box, iou, output_path):
    """Create visualization of predicted and target bounding boxes"""
    # Convert tensor to numpy and denormalize
    img_np = image.permute(1, 2, 0).cpu().numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 10))
    
    # Display image
    ax.imshow(img_np)
    
    # Draw predicted box
    x1, y1, x2, y2 = pred_box.tolist()
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, 
                         linewidth=2, edgecolor='b', facecolor='none',
                         label='Prediction')
    ax.add_patch(rect)
    
    # Draw target box
    x1, y1, x2, y2 = target_box.tolist()
    width = x2 - x1
    height = y2 - y1
    rect = patches.Rectangle((x1, y1), width, height, 
                         linewidth=2, edgecolor='r', facecolor='none',
                         label='Ground Truth')
    ax.add_patch(rect)
    
    # Add IoU to title
    ax.set_title(f'IoU: {iou:.4f}')
    
    # Remove axes
    ax.axis('off')
    
    # Add legend
    ax.legend()
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_histogram(data, label, output_path, bins=20):
    """Create histogram of data"""
    plt.figure(figsize=(10, 6))
    plt.hist(data, bins=bins, alpha=0.7, color='skyblue', edgecolor='black')
    plt.xlabel(label)
    plt.ylabel('Frequency')
    plt.title(f'{label} Distribution')
    plt.grid(alpha=0.3)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Analyze model performance")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--output_dir", type=str, default="analysis_output", help="Output directory for analysis")
    parser.add_argument("--max_samples", type=int, default=50, help="Maximum number of samples to analyze")
    parser.add_argument("--device", type=str, default=None, help="Device to run analysis on")
    
    return parser.parse_args()


def main():
    """Main analysis function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = ModelConfig()
    
    # Set device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    
    # Set up logger
    logger = get_logger(config, "model_analysis")
    logger.info(f"Using device: {device}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if os.path.isfile(args.checkpoint):
        logger.info(f"Loading checkpoint from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        # Check if checkpoint contains full model or just state dict
        if 'model' in checkpoint:
            model.load_state_dict(checkpoint['model'])
        else:
            model.load_state_dict(checkpoint)
    else:
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        return
    
    # Build data loader
    from temp.dataloader import build_dataloaders
    logger.info("Building data loaders...")
    dataloaders = build_dataloaders(config, include_text=True)
    
    # Run analysis
    logger.info("Running analysis...")
    output_dir = args.output_dir
    os.makedirs(output_dir, exist_ok=True)
    
    results = analyze_model_predictions(
        model,
        dataloaders['val'],
        device,
        output_dir,
        max_samples=args.max_samples
    )
    
    # Print analysis results
    logger.info("Analysis completed!")
    logger.info(f"Results saved to {output_dir}")
    logger.info("Metrics:")
    for key, value in results['metrics'].items():
        logger.info(f"{key}: {value:.4f}")


if __name__ == "__main__":
    main() 