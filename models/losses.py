"""
Loss functions for TransVG model
Includes L1 loss and GIoU loss for bounding box regression
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def box_cxcywh_to_xyxy(x):
    """
    Convert bounding box from [cx, cy, w, h] to [x1, y1, x2, y2] format
    
    Args:
        x: tensor of shape (..., 4) with [cx, cy, w, h] format
        
    Returns:
        y: tensor of shape (..., 4) with [x1, y1, x2, y2] format
    """
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h),
         (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)


def box_xyxy_to_cxcywh(x):
    """
    Convert bounding box from [x1, y1, x2, y2] to [cx, cy, w, h] format
    
    Args:
        x: tensor of shape (..., 4) with [x1, y1, x2, y2] format
        
    Returns:
        y: tensor of shape (..., 4) with [cx, cy, w, h] format
    """
    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2,
         (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)


def generalized_box_iou(boxes1, boxes2):
    """
    Compute the generalized IoU between two sets of boxes.
    
    Args:
        boxes1: Tensor of shape (B, 4) in [x1, y1, x2, y2] format
        boxes2: Tensor of shape (B, 4) in [x1, y1, x2, y2] format
        
    Returns:
        giou: Tensor of shape (B,) containing the generalized IoU for each pair of boxes
    """
    # Get coordinates
    x1, y1, x2, y2 = boxes1.unbind(-1)
    x1g, y1g, x2g, y2g = boxes2.unbind(-1)
    
    # Calculate areas
    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x2g - x1g) * (y2g - y1g)
    
    # Calculate intersection
    xmin = torch.max(x1, x1g)
    ymin = torch.max(y1, y1g)
    xmax = torch.min(x2, x2g)
    ymax = torch.min(y2, y2g)
    
    # Ensure intersection is valid (xmax > xmin, ymax > ymin)
    w = (xmax - xmin).clamp(min=0)
    h = (ymax - ymin).clamp(min=0)
    inter = w * h
    
    # Calculate union
    union = area1 + area2 - inter
    
    # Calculate IoU
    iou = inter / union
    
    # Calculate enclosing box
    xmin_c = torch.min(x1, x1g)
    ymin_c = torch.min(y1, y1g)
    xmax_c = torch.max(x2, x2g)
    ymax_c = torch.max(y2, y2g)
    
    # Calculate area of enclosing box
    w_c = xmax_c - xmin_c
    h_c = ymax_c - ymin_c
    area_c = w_c * h_c
    
    # Calculate GIoU
    giou = iou - (area_c - union) / area_c
    
    return giou


class FocalLoss(nn.Module):
    """
    Focal Loss for imbalanced classification
    Helps focus on hard examples
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        # Binary focal loss for bounding box confidence
        bce_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-bce_loss)  # Probability of the correct class
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class TransVGLoss(nn.Module):
    """
    Loss function for TransVG
    Combines L1 loss, GIoU loss, and optionally center prediction loss and focal loss
    """
    def __init__(self, config):
        super().__init__()
        self.l1_weight = getattr(config, 'l1_weight', 5.0)
        self.giou_weight = getattr(config, 'giou_weight', 2.0)
        self.center_weight = getattr(config, 'center_weight', 1.0)
        self.use_focal_loss = getattr(config, 'use_focal_loss', False)
        self.use_center_loss = getattr(config, 'use_center_loss', False)
        
        # Create focal loss if needed
        if self.use_focal_loss:
            self.focal_loss = FocalLoss(alpha=0.25, gamma=2.0)
    
    def forward(self, pred_boxes, target_boxes):
        """
        Args:
            pred_boxes: Predicted boxes in [xmin, ymin, xmax, ymax] format (B, 4)
            target_boxes: Target boxes in [xmin, ymin, xmax, ymax] format (B, 4)
            
        Returns:
            loss: Total loss
            loss_dict: Dictionary with individual loss terms
        """
        # Both pred_boxes and target_boxes are already in [xmin, ymin, xmax, ymax] format
        
        # Calculate L1 loss
        l1_loss = F.l1_loss(pred_boxes, target_boxes, reduction='none')
        l1_loss = l1_loss.sum(dim=1).mean()
        
        # Calculate GIoU loss
        giou = generalized_box_iou(pred_boxes, target_boxes)
        giou_loss = 1 - giou.mean()
        
        # Calculate center point loss if enabled
        center_loss = torch.tensor(0.0, device=pred_boxes.device)
        if self.use_center_loss:
            # Convert to center format for center loss
            pred_cxcywh = box_xyxy_to_cxcywh(pred_boxes)
            target_cxcywh = box_xyxy_to_cxcywh(target_boxes)
            
            # Only use center coordinates (cx, cy)
            pred_center = pred_cxcywh[:, :2]
            target_center = target_cxcywh[:, :2]
            
            # Center prediction loss (strong weight on center prediction)
            center_loss = F.mse_loss(pred_center, target_center)
        
        # Combine losses
        loss = self.l1_weight * l1_loss + self.giou_weight * giou_loss
        
        # Add center loss if enabled
        if self.use_center_loss:
            loss += self.center_weight * center_loss
        
        # Create loss dictionary for logging
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'giou_loss': giou_loss.item(),
            'total_loss': loss.item()
        }
        
        if self.use_center_loss:
            loss_dict['center_loss'] = center_loss.item()
        
        return loss, loss_dict 