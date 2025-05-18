"""
Evaluation metrics for TransVG model
Primarily focused on IoU metrics for visual grounding
"""

import torch
import numpy as np


def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes
    
    Args:
        box1: tensor of shape (..., 4) with [x1, y1, x2, y2] format
        box2: tensor of shape (..., 4) with [x1, y1, x2, y2] format
        
    Returns:
        iou: tensor of shape (...) with IoU values
    """
    # Calculate intersection area
    xmin = torch.max(box1[..., 0], box2[..., 0])
    ymin = torch.max(box1[..., 1], box2[..., 1])
    xmax = torch.min(box1[..., 2], box2[..., 2])
    ymax = torch.min(box1[..., 3], box2[..., 3])
    
    w = torch.clamp(xmax - xmin, min=0)
    h = torch.clamp(ymax - ymin, min=0)
    inter = w * h
    
    # Calculate union area
    area1 = (box1[..., 2] - box1[..., 0]) * (box1[..., 3] - box1[..., 1])
    area2 = (box2[..., 2] - box2[..., 0]) * (box2[..., 3] - box2[..., 1])
    union = area1 + area2 - inter
    
    # Calculate IoU
    iou = inter / (union + 1e-6)  # Add small epsilon to avoid division by zero
    
    return iou


def compute_accuracy(pred_boxes, target_boxes, thresholds=[0.25, 0.5, 0.75]):
    """
    Compute accuracy at different IoU thresholds
    
    Args:
        pred_boxes: Predicted boxes in [x1, y1, x2, y2] format (N, 4)
        target_boxes: Target boxes in [x1, y1, x2, y2] format (N, 4)
        thresholds: List of IoU thresholds for accuracy calculation
        
    Returns:
        accuracies: Dictionary of accuracy values at different thresholds
    """
    # Convert tensors to numpy arrays if needed
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.detach().cpu().numpy()
    if isinstance(target_boxes, torch.Tensor):
        target_boxes = target_boxes.detach().cpu().numpy()
    
    # Ensure inputs are numpy arrays
    pred_boxes = np.asarray(pred_boxes)
    target_boxes = np.asarray(target_boxes)
    
    # Compute IoU for all boxes
    ious = []
    for i in range(len(pred_boxes)):
        # Ensure positive width and height
        pred_box = pred_boxes[i].copy()
        target_box = target_boxes[i].copy()
        
        # Make sure x2 > x1 and y2 > y1
        pred_box[[0, 2]] = sorted(pred_box[[0, 2]])
        pred_box[[1, 3]] = sorted(pred_box[[1, 3]])
        target_box[[0, 2]] = sorted(target_box[[0, 2]])
        target_box[[1, 3]] = sorted(target_box[[1, 3]])
        
        # Calculate IoU
        xmin = max(pred_box[0], target_box[0])
        ymin = max(pred_box[1], target_box[1])
        xmax = min(pred_box[2], target_box[2])
        ymax = min(pred_box[3], target_box[3])
        
        w = max(0, xmax - xmin)
        h = max(0, ymax - ymin)
        inter = w * h
        
        area1 = (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1])
        area2 = (target_box[2] - target_box[0]) * (target_box[3] - target_box[1])
        union = area1 + area2 - inter
        
        iou = inter / (union + 1e-6)
        ious.append(iou)
    
    ious = np.array(ious)
    
    # Calculate accuracy at different thresholds
    accuracies = {}
    for threshold in thresholds:
        accuracies[f'Acc@{threshold}'] = (ious >= threshold).mean()
    
    # Add mean IoU as a metric
    accuracies['mIoU'] = ious.mean()
    
    # Add median IoU as a metric
    accuracies['medianIoU'] = np.median(ious)
    
    return accuracies


def calculate_metrics(pred_boxes, target_boxes):
    """
    Calculate all evaluation metrics
    
    Args:
        pred_boxes: Predicted boxes in [x1, y1, x2, y2] format (N, 4)
        target_boxes: Target boxes in [x1, y1, x2, y2] format (N, 4)
        
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    # Calculate accuracy at different IoU thresholds
    accuracies = compute_accuracy(pred_boxes, target_boxes)
    
    # Calculate coordinate-wise errors
    if isinstance(pred_boxes, torch.Tensor):
        pred_boxes = pred_boxes.detach().cpu().numpy()
    if isinstance(target_boxes, torch.Tensor):
        target_boxes = target_boxes.detach().cpu().numpy()
    
    # Calculate absolute error for each coordinate
    abs_errors = np.abs(pred_boxes - target_boxes)
    coord_names = ['x1', 'y1', 'x2', 'y2']
    
    # Add coordinate errors to metrics
    for i, coord in enumerate(coord_names):
        accuracies[f'mae_{coord}'] = abs_errors[:, i].mean()
    
    # Calculate box center and size errors
    pred_centers_x = (pred_boxes[:, 0] + pred_boxes[:, 2]) / 2
    pred_centers_y = (pred_boxes[:, 1] + pred_boxes[:, 3]) / 2
    target_centers_x = (target_boxes[:, 0] + target_boxes[:, 2]) / 2
    target_centers_y = (target_boxes[:, 1] + target_boxes[:, 3]) / 2
    
    pred_widths = pred_boxes[:, 2] - pred_boxes[:, 0]
    pred_heights = pred_boxes[:, 3] - pred_boxes[:, 1]
    target_widths = target_boxes[:, 2] - target_boxes[:, 0]
    target_heights = target_boxes[:, 3] - target_boxes[:, 1]
    
    # Calculate relative size errors
    width_error = np.abs(pred_widths - target_widths) / (target_widths + 1e-6)
    height_error = np.abs(pred_heights - target_heights) / (target_heights + 1e-6)
    
    accuracies['center_x_error'] = np.abs(pred_centers_x - target_centers_x).mean()
    accuracies['center_y_error'] = np.abs(pred_centers_y - target_centers_y).mean()
    accuracies['width_error'] = width_error.mean()
    accuracies['height_error'] = height_error.mean()
    
    return accuracies 