"""
Test script for bounding box visualization

This script creates test visualizations with known values to ensure
bounding boxes are correctly displayed and text queries are visible.
"""

import os
import sys
import torch
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from bbox_diagnostics import visualize_bbox_transformations, compare_model_bbox_outputs
from temp.dataloader import build_dataloaders
from configs.model_config import ModelConfig

def create_test_visualization():
    """Create test visualizations with hardcoded values"""
    # Set up output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox_test_viz")
    os.makedirs(output_dir, exist_ok=True)
    
    # Load a few real samples from dataset
    config = ModelConfig()
    config.batch_size = 4
    dataloaders = build_dataloaders(config, include_text=True, use_pin_memory=False)
    
    # Get a batch from the train dataloader
    batch = next(iter(dataloaders['train']))
    
    # Create visualizations for each sample in the batch
    for i in range(len(batch['img'])):
        img = batch['img'][i]
        text = batch['text'][i]
        bbox = batch['original_bbox'][i]
        image_id = batch['image_id'][i]
        
        print(f"\nProcessing sample {i+1} - {image_id}")
        print(f"Original bbox: {bbox.tolist()}")
        print(f"Text query: {text}")
        
        # Ensure the bbox is correctly formed - this is critical
        # Format: [x1, y1, x2, y2]
        if bbox[0] > bbox[2] or bbox[1] > bbox[3]:
            print(f"WARNING: Invalid bbox format detected: {bbox.tolist()}")
            # Fix the coordinates if needed
            x1, y1, x2, y2 = bbox.tolist()
            if x1 > x2:
                x1, x2 = x2, x1
            if y1 > y2:
                y1, y2 = y2, y1
            bbox = torch.tensor([x1, y1, x2, y2])
            print(f"Fixed bbox: {bbox.tolist()}")
        
        # Create a simulated prediction (shifted slightly from ground truth)
        # Convert to cx, cy, w, h format
        x1, y1, x2, y2 = bbox.tolist()
        w = x2 - x1
        h = y2 - y1
        cx = x1 + w/2
        cy = y1 + h/2
        
        # Add some noise to prediction (10-20% shift)
        cx_pred = cx * (1.0 + 0.1 * (i % 3))
        cy_pred = cy * (1.0 - 0.1 * ((i+1) % 3))
        w_pred = w * (0.8 + 0.2 * ((i+2) % 3))
        h_pred = h * (0.8 + 0.2 * ((i+3) % 3))
        
        pred_bbox = torch.tensor([cx_pred, cy_pred, w_pred, h_pred])
        
        # Generate visualizations
        # 1. Transformation visualization
        print("Generating bbox transformation visualization...")
        transform_path = visualize_bbox_transformations(
            img, bbox, config.image_size, text_query=text
        )
        print(f"Saved to: {transform_path}")
        
        # 2. Comparison visualization
        print("Generating bbox comparison visualization...")
        compare_path = compare_model_bbox_outputs(
            img, text, bbox, pred_bbox, image_id
        )
        print(f"Saved to: {compare_path}")
        
        # 3. Simple direct matplotlib visualization as a fallback
        print("Generating direct matplotlib visualization...")
        
        # Convert image tensor to numpy and denormalize
        img_np = img.permute(1, 2, 0).cpu().numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 1)
        
        # Get display size
        display_h, display_w = img_np.shape[:2]
        
        # Estimate original image size from target coordinates
        original_img_size = max(x2, y2) * 1.1  # Add some margin
        
        # Calculate scale factor from original to display
        scale_w = display_w / original_img_size
        scale_h = display_h / original_img_size
        
        # Scale bounding box coordinates to match displayed image
        gt_x1 = x1 * scale_w
        gt_y1 = y1 * scale_h
        gt_x2 = x2 * scale_w
        gt_y2 = y2 * scale_h
        
        # Scale predicted coordinates
        pred_x1 = (cx_pred - w_pred/2) * scale_w
        pred_y1 = (cy_pred - h_pred/2) * scale_h
        pred_x2 = (cx_pred + w_pred/2) * scale_w
        pred_y2 = (cy_pred + h_pred/2) * scale_h
        
        # Create a figure large enough to show everything
        fig, ax = plt.subplots(1, figsize=(16, 16))
        
        # Add a visible title with the query
        ax.set_title(f"Query: {text}", fontsize=16, fontweight='bold')
        
        # Display the full image
        ax.imshow(img_np)
        
        # Create a rectangle patch for the ground truth box
        from matplotlib.patches import Rectangle
        gt_rect = Rectangle((gt_x1, gt_y1), gt_x2-gt_x1, gt_y2-gt_y1, linewidth=3, 
                           edgecolor='r', facecolor='none', label='Ground Truth')
        ax.add_patch(gt_rect)
        
        # Create a rectangle for the prediction
        pred_rect = Rectangle((pred_x1, pred_y1), pred_x2-pred_x1, pred_y2-pred_y1, linewidth=3,
                           edgecolor='b', facecolor='none', label='Prediction',
                           linestyle='--')
        ax.add_patch(pred_rect)
        
        # Make sure the axes include all bounding boxes by setting limits
        # Add some padding to make boxes fully visible
        padding = 20
        x_min = max(0, min(gt_x1, pred_x1) - padding)
        y_min = max(0, min(gt_y1, pred_y1) - padding)
        x_max = min(display_w, max(gt_x2, pred_x2) + padding)
        y_max = min(display_h, max(gt_y2, pred_y2) + padding)
        
        # Set the axis limits to show the region with the bounding boxes
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_max, y_min)  # Inverted y-axis for image coordinates
        
        # Add a legend
        ax.legend(loc='upper right', fontsize=12)
        
        # Add coordinate info
        bbox_info = (
            f"Ground Truth [x1,y1,x2,y2] (original): [{x1:.1f}, {y1:.1f}, {x2:.1f}, {y2:.1f}]\n"
            f"Prediction [x1,y1,x2,y2] (display): [{pred_x1:.1f}, {pred_y1:.1f}, {pred_x2:.1f}, {pred_y2:.1f}]"
        )
        # Use figure transform for text
        plt.figtext(0.5, 0.01, bbox_info, fontsize=12,
               ha='center', va='bottom', bbox=dict(facecolor='white', alpha=0.8))
        
        # Save the direct visualization
        direct_path = os.path.join(output_dir, f"direct_viz_{image_id}.png")
        plt.tight_layout()
        plt.savefig(direct_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved direct visualization to: {direct_path}")
        
        # Create a second visualization showing the full image with the region highlighted
        fig, ax = plt.subplots(1, figsize=(10, 10))
        ax.set_title(f"Full Image with Zoomed Region Highlighted\nQuery: {text}", fontsize=14)
        ax.imshow(img_np)
        
        # Draw ground truth box
        ax.add_patch(Rectangle((gt_x1, gt_y1), gt_x2-gt_x1, gt_y2-gt_y1, linewidth=3, 
                              edgecolor='r', facecolor='none', label='Ground Truth'))
        
        # Draw predicted box
        ax.add_patch(Rectangle((pred_x1, pred_y1), pred_x2-pred_x1, pred_y2-pred_y1, linewidth=3, 
                              edgecolor='b', facecolor='none', label='Prediction',
                              linestyle='--'))
        
        # Add scaling information
        scale_info = (
            f"Original size (est): {original_img_size:.1f}x{original_img_size:.1f}\n"
            f"Display size: {display_w}x{display_h}\n"
            f"Scale factors: {scale_w:.3f}, {scale_h:.3f}"
        )
        ax.text(0.02, 0.02, scale_info, transform=ax.transAxes, fontsize=10,
              verticalalignment='bottom', horizontalalignment='left',
              bbox=dict(facecolor='white', alpha=0.7))
        
        ax.legend(loc='upper right')
        
        # Save the full image visualization
        full_path = os.path.join(output_dir, f"full_viz_{image_id}.png")
        plt.tight_layout()
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Saved full visualization to: {full_path}")

if __name__ == "__main__":
    print("Running bbox visualization test...")
    create_test_visualization()
    print("\nTest complete. Check the saved images to ensure bounding boxes and text are visible.") 