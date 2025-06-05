#!/usr/bin/env python3
"""
Inference script for improved TransVG model with DINO ViT backbone
Generates annotated visualizations showing:
- Original image
- Text description
- Ground truth bounding box (green)
- Predicted bounding box (red)
- IoU score and confidence metrics
- Model configuration and checkpoint information
"""

import os
import sys
import torch
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
import json
from datetime import datetime
from torchvision import transforms

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transvg import build_model
from models.custom_dataloader import build_dataloaders
from utils.metrics import compute_iou, calculate_metrics
from configs.dino_vit_config import DinoVitConfig
from create_comparison_gallery import create_comparison_gallery, create_performance_histogram


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Generate annotated inferences from TransVG model")
    parser.add_argument("--checkpoint_path", type=str, 
                        default="/home/pokle/Trans-VG/visual_grounding/checkpoints/all_data_dino_improved_epoch54.pth",
                        help="Path to model checkpoint")
    parser.add_argument("--data_root", type=str, default="/home/pokle/Trans-VG/visual_grounding/dior-rsvg", 
                        help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="inference_results", 
                        help="Output directory for inference results")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test"], 
                        help="Dataset split to run inference on")
    parser.add_argument("--num_samples", type=int, default=50, 
                        help="Number of samples to visualize (0 for all)")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cpu, cuda:0, cuda:1, etc)")
    parser.add_argument("--save_metrics", action="store_true", default=True,
                        help="Save detailed metrics to JSON file")
    parser.add_argument("--create_summary", action="store_true", default=True,
                        help="Create a summary report with key statistics")
    parser.add_argument("--create_gallery", action="store_true", default=True,
                        help="Create comparison gallery of best and worst samples")
    
    return parser.parse_args()


def load_model_and_config(checkpoint_path, device):
    """
    Load trained model from checkpoint
    
    Args:
        checkpoint_path: Path to checkpoint file
        device: Device to load model on
        
    Returns:
        model: Loaded model
        config: Model configuration
        checkpoint_info: Information about the checkpoint
    """
    # Load configuration
    config = DinoVitConfig()
    
    # Build model
    model = build_model(config)
    model = model.to(device)
    
    # Load checkpoint
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    print(f"Loading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load model state
    model.load_state_dict(checkpoint['model'])
    model.eval()
    
    # Extract checkpoint information
    checkpoint_info = {
        'epoch': checkpoint.get('epoch', 'Unknown'),
        'metrics': checkpoint.get('metrics', {}),
        'path': checkpoint_path,
        'file_size_mb': os.path.getsize(checkpoint_path) / (1024 * 1024)
    }
    
    print(f"Loaded model from epoch {checkpoint_info['epoch']}")
    if checkpoint_info['metrics']:
        print(f"Checkpoint metrics: {checkpoint_info['metrics']}")
    
    return model, config, checkpoint_info


def run_inference(model, dataloader, device, num_samples=0):
    """
    Run inference on dataset
    
    Args:
        model: Trained model
        dataloader: Data loader
        device: Device to run on
        num_samples: Number of samples to process (0 for all)
        
    Returns:
        results: List of inference results
    """
    model.eval()
    results = []
    
    with torch.no_grad():
        for i, batch in enumerate(dataloader):
            if num_samples > 0 and len(results) >= num_samples:
                break
            
            # Move data to device
            img = batch['img'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            target = batch['target'].to(device)  # Normalized [0,1]
            original_bbox = batch['original_bbox'].to(device)  # Original pixel coordinates
            orig_img_size = batch['orig_img_size'].to(device)  # Original image dimensions
            
            # Forward pass - model outputs normalized coordinates [0,1]
            pred_boxes = model(img, text_tokens, text_mask)
            
            # Process each sample in the batch
            batch_size = img.shape[0]
            for j in range(batch_size):
                if num_samples > 0 and len(results) >= num_samples:
                    break
                
                # Get original image dimensions
                img_w, img_h = orig_img_size[j].cpu().numpy()
                
                # Convert normalized predictions to pixel coordinates
                pred_box_pixels = torch.zeros_like(pred_boxes[j])
                pred_box_pixels[0] = pred_boxes[j, 0] * img_w  # x1
                pred_box_pixels[1] = pred_boxes[j, 1] * img_h  # y1
                pred_box_pixels[2] = pred_boxes[j, 2] * img_w  # x2
                pred_box_pixels[3] = pred_boxes[j, 3] * img_h  # y2
                
                # Calculate IoU
                iou = compute_iou(pred_box_pixels.unsqueeze(0), original_bbox[j].unsqueeze(0)).item()
                
                # Store result
                result = {
                    'image_tensor': img[j].cpu(),
                    'text': batch['text'][j],
                    'image_name': batch['image_name'][j],
                    'pred_box_normalized': pred_boxes[j].cpu().numpy(),
                    'pred_box_pixels': pred_box_pixels.cpu().numpy(),
                    'target_box_normalized': target[j].cpu().numpy(),
                    'target_box_pixels': original_bbox[j].cpu().numpy(),
                    'original_img_size': (img_w, img_h),
                    'iou': iou,
                    'sample_idx': len(results)
                }
                
                results.append(result)
                
                if (len(results)) % 10 == 0:
                    print(f"Processed {len(results)} samples...")
    
    return results


def create_annotated_visualization(result, output_path, checkpoint_info, config):
    """
    Create annotated visualization for a single result
    
    Args:
        result: Inference result dictionary
        output_path: Path to save the visualization
        checkpoint_info: Information about the model checkpoint
        config: Model configuration
    """
    # Convert tensor to PIL image
    img_tensor = result['image_tensor']
    # Denormalize the image
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = img_tensor * std + mean
    img_tensor = torch.clamp(img_tensor, 0, 1)
    
    # Convert to PIL image
    img_pil = transforms.ToPILImage()(img_tensor)
    
    # Create figure with subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    # Original image with normalized coordinates visualization
    ax1.imshow(img_pil)
    ax1.set_title("Model Input (Resized)", fontsize=14, fontweight='bold')
    
    # Scale bounding boxes to the resized image dimensions (config.image_size)
    img_size = getattr(config, 'image_size', 384)
    
    # Ground truth box (green) - normalized to resized image
    gt_box_resized = result['target_box_normalized'] * img_size
    gt_rect = patches.Rectangle(
        (gt_box_resized[0], gt_box_resized[1]), 
        gt_box_resized[2] - gt_box_resized[0], 
        gt_box_resized[3] - gt_box_resized[1],
        linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth'
    )
    ax1.add_patch(gt_rect)
    
    # Predicted box (red) - normalized to resized image
    pred_box_resized = result['pred_box_normalized'] * img_size
    pred_rect = patches.Rectangle(
        (pred_box_resized[0], pred_box_resized[1]), 
        pred_box_resized[2] - pred_box_resized[0], 
        pred_box_resized[3] - pred_box_resized[1],
        linewidth=3, edgecolor='red', facecolor='none', label='Prediction'
    )
    ax1.add_patch(pred_rect)
    
    ax1.legend(loc='upper right')
    ax1.set_xlim(0, img_size)
    ax1.set_ylim(img_size, 0)  # Flip y-axis for image coordinates
    
    # Load and display original image
    img_path = os.path.join("/home/pokle/Trans-VG/visual_grounding/dior-rsvg/JPEGImages", 
                           result['image_name'])
    if os.path.exists(img_path):
        original_img = Image.open(img_path).convert('RGB')
        ax2.imshow(original_img)
        ax2.set_title("Original Image with Predictions", fontsize=14, fontweight='bold')
        
        # Ground truth box (green) - original pixel coordinates
        gt_box_orig = result['target_box_pixels']
        gt_rect_orig = patches.Rectangle(
            (gt_box_orig[0], gt_box_orig[1]), 
            gt_box_orig[2] - gt_box_orig[0], 
            gt_box_orig[3] - gt_box_orig[1],
            linewidth=3, edgecolor='green', facecolor='none', label='Ground Truth'
        )
        ax2.add_patch(gt_rect_orig)
        
        # Predicted box (red) - original pixel coordinates
        pred_box_orig = result['pred_box_pixels']
        pred_rect_orig = patches.Rectangle(
            (pred_box_orig[0], pred_box_orig[1]), 
            pred_box_orig[2] - pred_box_orig[0], 
            pred_box_orig[3] - pred_box_orig[1],
            linewidth=3, edgecolor='red', facecolor='none', label='Prediction'
        )
        ax2.add_patch(pred_rect_orig)
        
        ax2.legend(loc='upper right')
        ax2.set_xlim(0, result['original_img_size'][0])
        ax2.set_ylim(result['original_img_size'][1], 0)  # Flip y-axis
    else:
        ax2.text(0.5, 0.5, f"Original image not found:\n{img_path}", 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title("Original Image (Not Found)", fontsize=14)
    
    # Add text and metrics information
    info_text = f"""Text Query: "{result['text']}"

IoU Score: {result['iou']:.4f}

Ground Truth (pixels): [{gt_box_orig[0]:.1f}, {gt_box_orig[1]:.1f}, {gt_box_orig[2]:.1f}, {gt_box_orig[3]:.1f}]
Prediction (pixels): [{pred_box_orig[0]:.1f}, {pred_box_orig[1]:.1f}, {pred_box_orig[2]:.1f}, {pred_box_orig[3]:.1f}]

Model: TransVG with DINO ViT backbone
Checkpoint: {os.path.basename(checkpoint_info['path'])} (Epoch {checkpoint_info['epoch']})
Image: {result['image_name']} ({result['original_img_size'][0]}x{result['original_img_size'][1]})"""
    
    fig.suptitle(f"Sample {result['sample_idx'] + 1}: Visual Grounding Result", 
                fontsize=16, fontweight='bold')
    
    # Add text box with information
    fig.text(0.02, 0.02, info_text, fontsize=10, 
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8),
            verticalalignment='bottom')
    
    plt.tight_layout()
    plt.subplots_adjust(bottom=0.25)  # Make room for text
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def create_summary_report(results, checkpoint_info, config, output_dir):
    """
    Create a summary report with key statistics
    
    Args:
        results: List of inference results
        checkpoint_info: Information about the model checkpoint
        config: Model configuration
        output_dir: Output directory
    """
    # Calculate overall metrics
    pred_boxes = np.array([r['pred_box_pixels'] for r in results])
    target_boxes = np.array([r['target_box_pixels'] for r in results])
    
    metrics = calculate_metrics(torch.tensor(pred_boxes), torch.tensor(target_boxes))
    
    # Convert metrics to native Python types for JSON serialization
    metrics = {k: float(v) if isinstance(v, (np.floating, torch.Tensor)) else v for k, v in metrics.items()}
    
    # Calculate IoU distribution
    ious = [r['iou'] for r in results]
    iou_stats = {
        'mean': float(np.mean(ious)),
        'median': float(np.median(ious)),
        'std': float(np.std(ious)),
        'min': float(np.min(ious)),
        'max': float(np.max(ious))
    }
    
    # Create summary report
    report = {
        'experiment_info': {
            'model': 'TransVG with DINO ViT backbone',
            'checkpoint_path': checkpoint_info['path'],
            'checkpoint_epoch': int(checkpoint_info['epoch']) if isinstance(checkpoint_info['epoch'], (np.integer, int)) else checkpoint_info['epoch'],
            'checkpoint_metrics': {k: float(v) if isinstance(v, (np.floating, np.integer)) else v for k, v in checkpoint_info['metrics'].items()} if checkpoint_info['metrics'] else {},
            'inference_date': datetime.now().isoformat(),
            'num_samples_evaluated': len(results)
        },
        'model_config': {
            'image_size': getattr(config, 'image_size', 384),
            'hidden_dim': getattr(config, 'hidden_dim', 256),
            'vision_backbone': getattr(config, 'vision_backbone', 'dino_vit'),
            'use_center_loss': getattr(config, 'use_center_loss', True),
            'use_focal_loss': getattr(config, 'use_focal_loss', True),
            'use_augmentation': getattr(config, 'use_augmentation', True)
        },
        'performance_metrics': metrics,
        'iou_statistics': iou_stats,
        'sample_breakdown': {
            'excellent_ious_0.75+': sum(1 for iou in ious if iou >= 0.75),
            'good_ious_0.5_to_0.75': sum(1 for iou in ious if 0.5 <= iou < 0.75),
            'moderate_ious_0.25_to_0.5': sum(1 for iou in ious if 0.25 <= iou < 0.5),
            'poor_ious_below_0.25': sum(1 for iou in ious if iou < 0.25)
        }
    }
    
    # Save report as JSON
    report_path = os.path.join(output_dir, 'inference_summary.json')
    with open(report_path, 'w') as f:
        json.dump(report, f, indent=2)
    
    # Create readable text summary
    summary_path = os.path.join(output_dir, 'inference_summary.txt')
    with open(summary_path, 'w') as f:
        f.write("TransVG Model Inference Summary Report\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Information:\n")
        f.write(f"  Architecture: {report['experiment_info']['model']}\n")
        f.write(f"  Checkpoint: {os.path.basename(checkpoint_info['path'])}\n")
        f.write(f"  Training Epoch: {checkpoint_info['epoch']}\n")
        f.write(f"  Image Size: {report['model_config']['image_size']}x{report['model_config']['image_size']}\n")
        f.write(f"  Inference Date: {report['experiment_info']['inference_date']}\n\n")
        
        f.write(f"Evaluation Results ({len(results)} samples):\n")
        f.write(f"  Mean IoU: {iou_stats['mean']:.4f} ± {iou_stats['std']:.4f}\n")
        f.write(f"  Median IoU: {iou_stats['median']:.4f}\n")
        f.write(f"  IoU Range: [{iou_stats['min']:.4f}, {iou_stats['max']:.4f}]\n\n")
        
        f.write("Accuracy at IoU Thresholds:\n")
        for key, value in metrics.items():
            if key.startswith('Acc@'):
                f.write(f"  {key}: {value:.4f} ({value*100:.1f}%)\n")
        f.write("\n")
        
        f.write("Performance Breakdown:\n")
        breakdown = report['sample_breakdown']
        total = len(results)
        f.write(f"  Excellent (IoU ≥ 0.75): {breakdown['excellent_ious_0.75+']} ({breakdown['excellent_ious_0.75+']/total*100:.1f}%)\n")
        f.write(f"  Good (0.5 ≤ IoU < 0.75): {breakdown['good_ious_0.5_to_0.75']} ({breakdown['good_ious_0.5_to_0.75']/total*100:.1f}%)\n")
        f.write(f"  Moderate (0.25 ≤ IoU < 0.5): {breakdown['moderate_ious_0.25_to_0.5']} ({breakdown['moderate_ious_0.25_to_0.5']/total*100:.1f}%)\n")
        f.write(f"  Poor (IoU < 0.25): {breakdown['poor_ious_below_0.25']} ({breakdown['poor_ious_below_0.25']/total*100:.1f}%)\n")
    
    print(f"Summary report saved to: {summary_path}")
    return report


def main():
    """Main inference function"""
    args = parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    try:
        model, config, checkpoint_info = load_model_and_config(args.checkpoint_path, device)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("Please check the checkpoint path and try again.")
        return
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    # Set data configuration
    config.data_root = args.data_root
    config.use_augmentation = False  # No augmentation for inference
    
    # Build data loader
    print(f"Loading {args.split} dataset...")
    try:
        dataloaders = build_dataloaders(config, use_pin_memory=True)
        dataloader = dataloaders[args.split]
        print(f"Dataset size: {len(dataloader.dataset)}")
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Run inference
    print(f"Running inference on {args.num_samples if args.num_samples > 0 else 'all'} samples...")
    results = run_inference(model, dataloader, device, args.num_samples)
    print(f"Inference completed on {len(results)} samples")
    
    if len(results) == 0:
        print("No results generated. Check dataset and model configuration.")
        return
    
    # Create visualizations
    print("Creating annotated visualizations...")
    for i, result in enumerate(results):
        output_path = output_dir / f"sample_{i+1:03d}_iou_{result['iou']:.3f}.png"
        create_annotated_visualization(result, output_path, checkpoint_info, config)
        
        if (i + 1) % 10 == 0:
            print(f"Created {i + 1}/{len(results)} visualizations")
    
    # Save detailed metrics if requested
    if args.save_metrics:
        metrics_path = output_dir / "detailed_metrics.json"
        detailed_metrics = {
            'samples': [
                {
                    'image_name': r['image_name'],
                    'text': r['text'],
                    'iou': r['iou'],
                    'pred_box': r['pred_box_pixels'].tolist(),
                    'target_box': r['target_box_pixels'].tolist()
                }
                for r in results
            ]
        }
        with open(metrics_path, 'w') as f:
            json.dump(detailed_metrics, f, indent=2)
        print(f"Detailed metrics saved to: {metrics_path}")
    
    # Create summary report if requested
    if args.create_summary:
        summary_report = create_summary_report(results, checkpoint_info, config, output_dir)
    
    # Create comparison gallery if requested
    if args.create_gallery:
        try:
            gallery_path = output_dir / 'comparison_gallery.png'
            histogram_path = output_dir / 'performance_histogram.png'
            create_comparison_gallery(str(output_dir), str(gallery_path))
            create_performance_histogram(str(output_dir), str(histogram_path))
            print(f"Comparison gallery saved to: {gallery_path}")
            print(f"Performance histogram saved to: {histogram_path}")
        except Exception as e:
            print(f"Warning: Could not create comparison gallery: {e}")
    
    print(f"\nInference complete!")
    print(f"Results saved to: {output_dir}")
    print(f"Generated {len(results)} annotated visualizations")
    
    # Print quick statistics
    ious = [r['iou'] for r in results]
    print(f"\nQuick Statistics:")
    print(f"  Mean IoU: {np.mean(ious):.4f}")
    print(f"  Acc@0.5: {sum(1 for iou in ious if iou >= 0.5) / len(ious):.4f}")
    print(f"  Acc@0.75: {sum(1 for iou in ious if iou >= 0.75) / len(ious):.4f}")
    
    # Summary for academic presentation
    print(f"\n" + "="*60)
    print("ACADEMIC PRESENTATION SUMMARY")
    print("="*60)
    print(f"Model: TransVG with DINO ViT backbone")
    print(f"Checkpoint: {os.path.basename(args.checkpoint_path)} (Epoch {checkpoint_info['epoch']})")
    print(f"Dataset: {args.split} split ({len(results)} samples)")
    print(f"Average IoU: {np.mean(ious):.4f} ± {np.std(ious):.4f}")
    print(f"Accuracy@0.5: {sum(1 for iou in ious if iou >= 0.5) / len(ious)*100:.1f}%")
    print(f"Accuracy@0.75: {sum(1 for iou in ious if iou >= 0.75) / len(ious)*100:.1f}%")
    print(f"\nOutputs generated:")
    print(f"  - {len(results)} annotated sample visualizations")
    if args.save_metrics:
        print(f"  - Detailed metrics (JSON)")
    if args.create_summary:
        print(f"  - Summary report (TXT)")
    if args.create_gallery:
        print(f"  - Comparison gallery (best/worst samples)")
        print(f"  - Performance histogram")
    print("="*60)


if __name__ == "__main__":
    main() 