"""
Bounding Box Diagnostics

This script provides utilities to debug bounding box transformations 
and visualize them at different stages of the model pipeline.
"""

import os
import sys
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def xyxy_to_cxcywh(bbox):
    """
    Convert [x1, y1, x2, y2] format to [cx, cy, w, h] format
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        
    Returns:
        Bounding box in [cx, cy, w, h] format
    """
    x1, y1, x2, y2 = bbox
    return [(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1]

def cxcywh_to_xyxy(bbox):
    """
    Convert [cx, cy, w, h] format to [x1, y1, x2, y2] format
    
    Args:
        bbox: Bounding box in [cx, cy, w, h] format
        
    Returns:
        Bounding box in [x1, y1, x2, y2] format
    """
    cx, cy, w, h = bbox
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]

def normalize_bbox(bbox, img_size):
    """
    Normalize bounding box coordinates to [0, 1] range
    
    Args:
        bbox: Bounding box in [x1, y1, x2, y2] format
        img_size: Image size (assumed square)
        
    Returns:
        Normalized bounding box in [x1, y1, x2, y2] format
    """
    # Original images might be larger than the img_size (e.g., 800x800 normalized to 224x224)
    # Make sure the coordinates are normalized to [0, 1] range
    return [min(coord / img_size, 1.0) for coord in bbox]

def denormalize_bbox(bbox, img_size):
    """
    Denormalize bounding box coordinates from [0, 1] range to pixel values
    
    Args:
        bbox: Normalized bounding box in [x1, y1, x2, y2] format
        img_size: Image size (assumed square)
        
    Returns:
        Denormalized bounding box in [x1, y1, x2, y2] format
    """
    return [coord * img_size for coord in bbox]

def visualize_bbox_transformations(img, original_bbox, img_size=224, text_query=None):
    """
    Visualize various bounding box transformations to debug coordinate issues
    
    Args:
        img: Input image tensor (C, H, W)
        original_bbox: Original bounding box in [x1, y1, x2, y2] format
        img_size: Image size (assumed square)
        text_query: Text query associated with the image
        
    Returns:
        Path to the saved visualization
    """
    # Convert to numpy arrays for visualization
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Get actual display size of the image
    display_h, display_w = img_np.shape[:2]
    
    # Calculate the scale factors to match original coordinates to displayed image
    scale_factor = min(display_w, display_h) / img_size
    
    # Original bounding box
    bbox_xyxy = original_bbox.cpu().numpy().copy()
    
    # Create transformations
    # 1. Original [x1, y1, x2, y2]
    # We'll rescale these for display, but keep the original for the coordinate text
    bbox_xyxy_orig = bbox_xyxy.copy()
    
    # 2. Convert to [cx, cy, w, h]
    bbox_cxcywh = xyxy_to_cxcywh(bbox_xyxy_orig)
    
    # 3. Normalize to [0, 1]
    # First, calculate the scale to original image dimensions
    original_img_size = max(bbox_xyxy_orig[2], bbox_xyxy_orig[3])  # Rough estimate of original size
    bbox_xyxy_norm = [coord / original_img_size for coord in bbox_xyxy_orig]
    
    # 4. Normalized [cx, cy, w, h]
    bbox_cxcywh_norm = [coord / original_img_size for coord in bbox_cxcywh]
    
    # 5. Back to [x1, y1, x2, y2] from normalized [cx, cy, w, h]
    bbox_xyxy_from_cxcywh_norm = cxcywh_to_xyxy(bbox_cxcywh_norm)
    
    # 6. Denormalized back from normalized [cx, cy, w, h]
    bbox_xyxy_denorm = [coord * original_img_size for coord in bbox_xyxy_from_cxcywh_norm]
    
    # Create figure with a query text banner at the top
    fig = plt.figure(figsize=(18, 14))
    
    # Add query text as a banner at the top if provided
    if text_query:
        plt.figtext(0.5, 0.96, f"Query: \"{text_query}\"", 
                   fontsize=16, ha='center', fontweight='bold',
                   bbox=dict(facecolor='blue', alpha=0.2, boxstyle='round,pad=0.5'))
    
    # Create subplots for the transformations
    gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3, top=0.92)
    axs = [fig.add_subplot(gs[i//3, i%3]) for i in range(6)]
    
    titles = [
        "Original [x1, y1, x2, y2]",
        "Converted to [cx, cy, w, h]",
        "Normalized [x1, y1, x2, y2]",
        "Normalized [cx, cy, w, h]",
        "Back to [x1, y1, x2, y2] from normalized [cx, cy, w, h]",
        "Denormalized from [cx, cy, w, h]"
    ]
    
    bboxes = [
        bbox_xyxy_orig,
        bbox_cxcywh,
        bbox_xyxy_norm,
        bbox_cxcywh_norm,
        bbox_xyxy_from_cxcywh_norm,
        bbox_xyxy_denorm
    ]
    
    colors = ['red', 'blue', 'green', 'purple', 'orange', 'cyan']
    
    for i, (ax, title, bbox, color) in enumerate(zip(axs, titles, bboxes, colors)):
        # Show image
        ax.imshow(img_np)
        
        # For visualization, convert all to [x1, y1, x2, y2] format
        if i == 1 or i == 3:  # These are in [cx, cy, w, h] format
            vis_bbox = cxcywh_to_xyxy(bbox)
        else:
            vis_bbox = bbox
        
        # For normalized coordinates, scale to display size
        if i == 2 or i == 3 or i == 4:
            # Scale normalized coordinates to display size
            vis_bbox = [coord * display_w if j % 2 == 0 else coord * display_h 
                      for j, coord in enumerate(vis_bbox)]
        elif i == 5 or i == 0:
            # Scale original or denormalized coordinates to display size
            # Original is already in absolute coordinates for the original image
            # Scale down to match the displayed image size
            vis_bbox = [coord * display_w / original_img_size if j % 2 == 0 else 
                       coord * display_h / original_img_size
                       for j, coord in enumerate(vis_bbox)]
        else:
            # For other formats, estimate display scale
            # Convert to normalized first, then scale to display
            scale = min(display_w, display_h) / max(bbox)
            vis_bbox = [coord * scale for coord in vis_bbox]
        
        # Draw bounding box using Rectangle patch - more visible with thicker line
        if i == 1 or i == 3:  # For [cx, cy, w, h] format
            # Need to convert for Rectangle which expects [x, y, width, height]
            cx, cy, w, h = vis_bbox
            rect = patches.Rectangle(
                (cx - w/2, cy - h/2), w, h,
                linewidth=4,
                edgecolor=color,
                facecolor='none',
                linestyle='-' if i % 2 == 0 else '--'
            )
        else:  # For [x1, y1, x2, y2] format
            x1, y1, x2, y2 = vis_bbox
            rect = patches.Rectangle(
                (x1, y1), x2-x1, y2-y1,
                linewidth=4,
                edgecolor=color,
                facecolor='none',
                linestyle='-' if i % 2 == 0 else '--'
            )
        ax.add_patch(rect)
        
        # Also draw the box using lines for even more visibility
        if i == 1 or i == 3:  # For [cx, cy, w, h] format
            cx, cy, w, h = vis_bbox
            x1, y1 = cx - w/2, cy - h/2
            x2, y2 = cx + w/2, cy + h/2
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=2)
        else:  # For [x1, y1, x2, y2] format
            x1, y1, x2, y2 = vis_bbox
            ax.plot([x1, x2, x2, x1, x1], [y1, y1, y2, y2, y1], color=color, linewidth=2)
        
        # Add title with contrasting background for visibility
        ax.set_title(title, fontsize=13, fontweight='bold', 
                   bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
        
        # Add coordinate values with better visibility - show the actual values, not display values
        coord_text = f"Coords: {[round(c, 3) for c in bbox]}"
        ax.text(
            0.05, 0.95, 
            coord_text, 
            transform=ax.transAxes, 
            fontsize=11,
            fontweight='bold',
            verticalalignment='top',
            horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.8, boxstyle='round,pad=0.3')
        )
        
        # Remove axes
        ax.axis('off')
    
    # Add explanation text at the bottom
    explanation = (
        "Input coordinates [xmin, ymin, xmax, ymax] are scaled to match the displayed image size.\n"
        f"Original image size estimate: {original_img_size}x{original_img_size}, "
        f"Display size: {display_w}x{display_h}"
    )
    plt.figtext(0.5, 0.01, explanation, ha='center', fontsize=12, 
               bbox=dict(facecolor='white', alpha=0.8))
    
    # Save figure
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"bbox_transformations_{int(torch.rand(1)[0] * 10000)}.png")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return output_path

def compare_model_bbox_outputs(img, text, target_bbox, pred_bbox, image_id=None):
    """
    Compare ground truth and predicted bounding boxes
    
    Args:
        img: Input image tensor (C, H, W)
        text: Text query
        target_bbox: Target bounding box in [x1, y1, x2, y2] format
        pred_bbox: Predicted bounding box in [cx, cy, w, h] format
        image_id: Image identifier
    
    Returns:
        Path to the saved visualization
    """
    # Convert tensors to numpy arrays
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Get the display size of the image
    display_h, display_w = img_np.shape[:2]
    
    # Convert bounding boxes to numpy arrays
    target_bbox_np = target_bbox.cpu().numpy()
    pred_bbox_np = pred_bbox.cpu().numpy()
    
    # Estimate original image size from target coordinates
    original_img_size = max(target_bbox_np[2], target_bbox_np[3]) * 1.1  # Add some margin
    
    # Calculate the scale factor from original to display size
    scale_w = display_w / original_img_size
    scale_h = display_h / original_img_size
    
    # Scale target bbox to display size
    target_bbox_display = [
        target_bbox_np[0] * scale_w,  # x1
        target_bbox_np[1] * scale_h,  # y1
        target_bbox_np[2] * scale_w,  # x2
        target_bbox_np[3] * scale_h   # y2
    ]
    
    # Convert predicted bbox from [cx, cy, w, h] to [x1, y1, x2, y2]
    pred_bbox_xyxy = cxcywh_to_xyxy(pred_bbox_np)
    
    # Scale predicted bbox to display size
    pred_bbox_display = [
        pred_bbox_xyxy[0] * original_img_size * scale_w,  # x1
        pred_bbox_xyxy[1] * original_img_size * scale_h,  # y1
        pred_bbox_xyxy[2] * original_img_size * scale_w,  # x2
        pred_bbox_xyxy[3] * original_img_size * scale_h   # y2
    ]
    
    # Compute IoU
    def compute_iou(box1, box2):
        # Box format: [x1, y1, x2, y2]
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        # Calculate intersection area
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        # Calculate union area
        box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
        box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = box1_area + box2_area - intersection
        
        # Calculate IoU
        iou = intersection / union if union > 0 else 0
        return iou
    
    iou = compute_iou(target_bbox_np, pred_bbox_xyxy)
    
    # Extract coordinates for easier access
    gt_x1, gt_y1, gt_x2, gt_y2 = target_bbox_display
    pred_x1, pred_y1, pred_x2, pred_y2 = pred_bbox_display
    
    # Calculate the region to focus on (with padding)
    padding = max(display_w, display_h) * 0.1  # 10% padding
    x_min = max(0, min(gt_x1, pred_x1) - padding)
    y_min = max(0, min(gt_y1, pred_y1) - padding)
    x_max = min(display_w, max(gt_x2, pred_x2) + padding)
    y_max = min(display_h, max(gt_y2, pred_y2) + padding)
    
    # Create output directory
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Part 1: Create a zoomed-in visualization
    fig = plt.figure(figsize=(12, 12))
    
    # Add query text as a banner at the top - make it very prominent
    plt.figtext(0.5, 0.97, f"Query: \"{text}\"", 
               fontsize=18, ha='center', fontweight='bold',
               bbox=dict(facecolor='blue', alpha=0.2, boxstyle='round,pad=0.5'))
    
    # Create main image subplot
    ax = fig.add_subplot(1, 1, 1)
    ax.imshow(img_np)
    
    # Draw ground truth box (green, thicker, solid line) - using Rectangle
    gt_rect = patches.Rectangle(
        (gt_x1, gt_y1),
        gt_x2 - gt_x1,
        gt_y2 - gt_y1,
        linewidth=4,
        edgecolor='limegreen',
        facecolor='none',
        label='Ground Truth',
        linestyle='-'
    )
    ax.add_patch(gt_rect)
    
    # Draw predicted box (red, thicker, dashed line) - using Rectangle
    pred_rect = patches.Rectangle(
        (pred_x1, pred_y1),
        pred_x2 - pred_x1,
        pred_y2 - pred_y1,
        linewidth=4,
        edgecolor='red',
        facecolor='none',
        label='Predicted',
        linestyle='--'
    )
    ax.add_patch(pred_rect)
    
    # Zoom to the region of interest
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_max, y_min)  # Inverted y-axis for image coordinates
    
    # Add image ID as subtitle if available
    if image_id:
        ax.set_title(f"Image ID: {image_id}", fontsize=14, fontweight='bold')
    
    # Add IoU score in a highly visible box
    iou_text = f"IoU: {iou:.4f}"
    ax.text(0.02, 0.02, iou_text, transform=ax.transAxes, fontsize=16,
            fontweight='bold', verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add legend with larger font and distinct background
    legend = ax.legend(loc='upper right', fontsize=14, framealpha=0.9)
    legend.get_frame().set_facecolor('white')
    
    # Add coordinate values in a separated box at the bottom
    bbox_info = (
        f"Ground Truth [x1,y1,x2,y2] (original): {[round(c, 1) for c in target_bbox_np]}\n"
        f"Predicted [cx,cy,w,h] (normalized): {[round(c, 3) for c in pred_bbox_np]}\n"
        f"Predicted [x1,y1,x2,y2] (original scale): {[round(c, 1) for c in pred_bbox_xyxy]}"
    )
    
    plt.figtext(0.5, 0.02, bbox_info, fontsize=12, ha='center', fontweight='bold',
                bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.5'))
    
    # Add scaling information
    scaling_info = (
        f"Original image size (estimated): {original_img_size:.1f}x{original_img_size:.1f}\n"
        f"Display size: {display_w}x{display_h}, Scale factors: {scale_w:.3f}, {scale_h:.3f}"
    )
    plt.figtext(0.5, 0.06, scaling_info, fontsize=10, ha='center',
                bbox=dict(facecolor='white', alpha=0.7))
    
    # Remove axes
    ax.axis('off')
    
    # Save zoomed visualization
    zoomed_path = os.path.join(output_dir, f"zoomed_comparison_{image_id if image_id else int(torch.rand(1)[0] * 10000)}.png")
    plt.tight_layout()
    plt.savefig(zoomed_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Part 2: Create a full-image visualization showing context
    fig, ax = plt.subplots(1, figsize=(12, 12))
    ax.set_title(f"Full Image Context\nQuery: \"{text}\"", fontsize=14, fontweight='bold')
    ax.imshow(img_np)
    
    # Draw ground truth box
    ax.add_patch(patches.Rectangle(
        (gt_x1, gt_y1), gt_x2-gt_x1, gt_y2-gt_y1, linewidth=3, 
        edgecolor='limegreen', facecolor='none', label='Ground Truth'))
    
    # Draw predicted box
    ax.add_patch(patches.Rectangle(
        (pred_x1, pred_y1), pred_x2-pred_x1, pred_y2-pred_y1, linewidth=3, 
        edgecolor='red', facecolor='none', label='Prediction', linestyle='--'))
    
    # Add IoU information
    ax.text(0.02, 0.02, f"IoU: {iou:.4f}", transform=ax.transAxes, fontsize=14,
           verticalalignment='bottom', horizontalalignment='left',
           bbox=dict(facecolor='white', alpha=0.8))
    
    ax.legend(loc='upper right')
    ax.axis('off')
    
    # Save full visualization
    full_path = os.path.join(output_dir, f"full_comparison_{image_id if image_id else int(torch.rand(1)[0] * 10000)}.png")
    plt.tight_layout()
    plt.savefig(full_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return zoomed_path

def trace_bbox_through_model(model, batch, device, trace_layers=False):
    """
    Trace a batch through the model and record intermediate bounding box representations
    
    Args:
        model: TransVG model
        batch: Batch of samples
        device: Device to run the model on
        trace_layers: Whether to trace intermediate layer representations
        
    Returns:
        Dictionary of tracing results
    """
    # Move inputs to device
    img = batch['img'].to(device)
    text_tokens = batch['text_tokens'].to(device)
    text_mask = batch['text_mask'].to(device)
    target = batch['target'].to(device)
    
    # Get text queries for display
    texts = batch.get('text', ['Query not available' for _ in range(len(img))])
    
    # Create results dictionary
    results = {
        'image': img.cpu(),
        'text': texts,
        'target': target.cpu(),
        'original_bbox': batch['original_bbox'],
        'image_id': batch['image_id']
    }
    
    # Put model in eval mode
    model.eval()
    
    # Forward pass with gradient tracking if trace_layers is True
    if trace_layers:
        # This would require model modification to return intermediate values
        # As a placeholder, we'll just get the final output
        with torch.no_grad():
            output = model(img, text_tokens, text_mask)
    else:
        with torch.no_grad():
            output = model(img, text_tokens, text_mask)
    
    # Store predictions
    results['pred'] = output.cpu()
    
    return results

def run_diagnostics(model, dataloader, device, num_samples=5, output_dir=None):
    """
    Run diagnostic tests on the model and dataset
    
    Args:
        model: TransVG model
        dataloader: DataLoader containing dataset samples
        device: Device to run the model on
        num_samples: Number of samples to test
        output_dir: Optional output directory for visualizations
        
    Returns:
        Dictionary of diagnostic results
    """
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "bbox_diagnostics")
    os.makedirs(output_dir, exist_ok=True)
    
    # Set model to eval mode
    model.eval()
    
    # Get a batch from the dataloader
    batch = next(iter(dataloader))
    
    # Trace batch through model
    trace_results = trace_bbox_through_model(model, batch, device)
    
    # Visualize samples
    vis_paths = []
    for i in range(min(len(batch['img']), num_samples)):
        # Extract sample components
        img = batch['img'][i]
        text = trace_results['text'][i]
        target_bbox = batch['original_bbox'][i]
        pred_bbox = trace_results['pred'][i]
        image_id = batch['image_id'][i]
        
        # Visualize box transformations
        transform_path = visualize_bbox_transformations(img, target_bbox, text_query=text)
        vis_paths.append(transform_path)
        
        # Compare target and predicted boxes
        compare_path = compare_model_bbox_outputs(img, text, target_bbox, pred_bbox, image_id)
        vis_paths.append(compare_path)
    
    # Calculate statistics for the entire dataloader
    num_samples_large = min(100, len(dataloader.dataset))
    stats_samples = []
    
    with torch.no_grad():
        for _ in range(num_samples_large // dataloader.batch_size + 1):
            try:
                batch = next(iter(dataloader))
                trace_results = trace_bbox_through_model(model, batch, device)
                
                # Extract predictions and targets
                preds = trace_results['pred']
                targets = batch['target']
                
                # Store statistics
                for i in range(len(preds)):
                    stats_samples.append({
                        'pred': preds[i].cpu().numpy(),
                        'target': targets[i].cpu().numpy()
                    })
                
                if len(stats_samples) >= num_samples_large:
                    break
            except StopIteration:
                break
    
    # Calculate statistics
    stats = {}
    
    # 1. Mean absolute error for each coordinate
    pred_coords = np.array([sample['pred'] for sample in stats_samples])
    target_coords = np.array([sample['target'] for sample in stats_samples])
    stats['mae_per_coord'] = np.mean(np.abs(pred_coords - target_coords), axis=0)
    
    # 2. Convert to [x1, y1, x2, y2] and calculate IoU
    ious = []
    for sample in stats_samples:
        pred_xyxy = cxcywh_to_xyxy(sample['pred'])
        target_xyxy = cxcywh_to_xyxy(sample['target'])
        
        # Calculate IoU
        x1 = max(pred_xyxy[0], target_xyxy[0])
        y1 = max(pred_xyxy[1], target_xyxy[1])
        x2 = min(pred_xyxy[2], target_xyxy[2])
        y2 = min(pred_xyxy[3], target_xyxy[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        pred_area = (pred_xyxy[2] - pred_xyxy[0]) * (pred_xyxy[3] - pred_xyxy[1])
        target_area = (target_xyxy[2] - target_xyxy[0]) * (target_xyxy[3] - target_xyxy[1])
        union = pred_area + target_area - intersection
        
        iou = intersection / union if union > 0 else 0
        ious.append(iou)
    
    stats['mean_iou'] = np.mean(ious)
    stats['median_iou'] = np.median(ious)
    
    # 3. Calculate accuracy at different IoU thresholds
    for threshold in [0.25, 0.5, 0.75]:
        stats[f'acc_{threshold}'] = np.mean([iou >= threshold for iou in ious])
    
    # Save statistics to file
    stats_path = os.path.join(output_dir, "bbox_statistics.txt")
    with open(stats_path, 'w') as f:
        f.write("Bounding Box Diagnostic Statistics\n")
        f.write("================================\n\n")
        
        f.write("Mean Absolute Error per Coordinate:\n")
        f.write(f"  cx: {stats['mae_per_coord'][0]:.4f}\n")
        f.write(f"  cy: {stats['mae_per_coord'][1]:.4f}\n")
        f.write(f"  w:  {stats['mae_per_coord'][2]:.4f}\n")
        f.write(f"  h:  {stats['mae_per_coord'][3]:.4f}\n\n")
        
        f.write("IoU Statistics:\n")
        f.write(f"  Mean IoU: {stats['mean_iou']:.4f}\n")
        f.write(f"  Median IoU: {stats['median_iou']:.4f}\n\n")
        
        f.write("Accuracy at IoU Thresholds:\n")
        f.write(f"  Acc@0.25: {stats['acc_0.25']:.4f}\n")
        f.write(f"  Acc@0.5: {stats['acc_0.5']:.4f}\n")
        f.write(f"  Acc@0.75: {stats['acc_0.75']:.4f}\n")
    
    return {
        'stats': stats,
        'visualizations': vis_paths,
        'stats_path': stats_path
    }

if __name__ == "__main__":
    import argparse
    from temp.dataloader import build_dataloaders
    from models.transvg import build_model
    from configs.model_config import ModelConfig
    
    parser = argparse.ArgumentParser(description="Run bounding box diagnostics")
    parser.add_argument("--output_dir", type=str, default="bbox_diagnostics", help="Directory to save diagnostics")
    parser.add_argument("--num_samples", type=int, default=5, help="Number of samples to visualize")
    parser.add_argument("--split", type=str, default="val", choices=["train", "val", "test"], help="Dataset split to use")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--device", type=str, default="cuda:1", help="Device to use (cuda:0, cuda:1, cpu)")
    parser.add_argument("--use_pin_memory", action="store_true", help="Use pin_memory in data loading")
    args = parser.parse_args()
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load configuration
    config = ModelConfig()
    
    # Build dataloaders (with text)
    dataloaders = build_dataloaders(config, include_text=True, use_pin_memory=args.use_pin_memory)
    
    # Build model
    model = build_model(config)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model = model.to(device)
    
    # Run diagnostics
    output_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), args.output_dir)
    diagnostic_results = run_diagnostics(model, dataloaders[args.split], device, args.num_samples, output_dir)
    
    print(f"Diagnostic results saved to {output_dir}")
    print(f"Statistics file: {diagnostic_results['stats_path']}")
    print(f"Number of visualizations: {len(diagnostic_results['visualizations'])}") 