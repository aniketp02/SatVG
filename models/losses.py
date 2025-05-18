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
    Compute Generalized IoU (GIoU) between boxes
    
    Args:
        boxes1: tensor of shape (N, 4) with [x1, y1, x2, y2] format
        boxes2: tensor of shape (N, 4) with [x1, y1, x2, y2] format
        
    Returns:
        giou: tensor of shape (N,) with GIoU values
    """
    # Convert to [x1, y1, x2, y2] format if needed
    assert boxes1.shape[-1] == 4 and boxes2.shape[-1] == 4
    
    # Get area of boxes
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
    
    # Get coordinates of intersection
    lt = torch.max(boxes1[:, :2], boxes2[:, :2])  # [N, 2]
    rb = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # [N, 2]
    
    # Calculate intersection area
    wh = (rb - lt).clamp(min=0)  # [N, 2]
    inter = wh[:, 0] * wh[:, 1]  # [N]
    
    # Calculate union area
    union = area1 + area2 - inter
    
    # Calculate IoU
    iou = inter / union
    
    # Calculate coordinates of enclosing box
    lt_c = torch.min(boxes1[:, :2], boxes2[:, :2])
    rb_c = torch.max(boxes1[:, 2:], boxes2[:, 2:])
    
    # Calculate area of enclosing box
    wh_c = (rb_c - lt_c).clamp(min=0)
    area_c = wh_c[:, 0] * wh_c[:, 1]
    
    # Calculate GIoU
    giou = iou - (area_c - union) / area_c
    
    return giou


class TransVGLoss(nn.Module):
    """
    Loss function for TransVG
    Combines L1 loss and GIoU loss for bounding box regression
    """
    def __init__(self, config):
        super().__init__()
        self.l1_weight = config.l1_weight
        self.giou_weight = config.giou_weight
    
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
        # No need for conversion
        
        # Calculate L1 loss
        l1_loss = F.l1_loss(pred_boxes, target_boxes, reduction='none')
        l1_loss = l1_loss.sum(dim=1).mean()
        
        # Calculate GIoU loss
        giou = generalized_box_iou(pred_boxes, target_boxes)
        giou_loss = 1 - giou.mean()
        
        # Combine losses
        loss = self.l1_weight * l1_loss + self.giou_weight * giou_loss
        
        # Create loss dictionary for logging
        loss_dict = {
            'l1_loss': l1_loss.item(),
            'giou_loss': giou_loss.item(),
            'total_loss': loss.item()
        }
        
        return loss, loss_dict 