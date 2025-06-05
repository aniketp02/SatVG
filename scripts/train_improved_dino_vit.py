#!/usr/bin/env python3
"""
Training script for improved TransVG model with DINO ViT backbone
Features:
- Center prediction loss
- Focal loss
- Light data augmentation
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.transvg import build_model
from models.losses import TransVGLoss
from models.custom_dataloader import build_dataloaders
from utils.logger import get_logger
from utils.metrics import calculate_metrics
from configs.dino_vit_config import DinoVitConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train improved TransVG model with DINO ViT backbone")
    parser.add_argument("--data_root", type=str, default="/home/pokle/Trans-VG/visual_grounding/dior-rsvg", 
                        help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="output/improved_dino_vit", 
                        help="Output directory for logs and checkpoints")
    parser.add_argument("--log_name", type=str, default=None, 
                        help="Name for log files")
    parser.add_argument("--epochs", type=int, default=200, 
                        help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=24, 
                        help="Batch size")
    parser.add_argument("--lr", type=float, default=5e-5, 
                        help="Learning rate")
    parser.add_argument("--lr_bert", type=float, default=2e-5, 
                        help="Learning rate for BERT")
    parser.add_argument("--weight_decay", type=float, default=1e-4, 
                        help="Weight decay")
    parser.add_argument("--lr_drop", type=int, default=70, 
                        help="Epoch at which to drop learning rate")
    parser.add_argument("--resume", type=str, default="", 
                        help="Resume from checkpoint")
    parser.add_argument("--device", type=str, default="cuda", 
                        help="Device to use (cpu, cuda:0, cuda:1, etc)")
    parser.add_argument("--num_workers", type=int, default=4, 
                        help="Number of workers for data loading")
    parser.add_argument("--scheduler", type=str, default="step", choices=["step", "cosine"], 
                        help="LR scheduler type")
    
    # Center prediction loss weight
    parser.add_argument("--center_weight", type=float, default=1.0, 
                        help="Weight for center prediction loss")
    
    # Data augmentation flags
    parser.add_argument("--no_augmentation", action="store_true", 
                        help="Disable data augmentation")
    parser.add_argument("--no_center_loss", action="store_true", 
                        help="Disable center prediction loss")
    parser.add_argument("--no_focal_loss", action="store_true", 
                        help="Disable focal loss")
    
    return parser.parse_args()


def train_one_epoch(model, data_loader, criterion, optimizer, device, logger, epoch, config):
    """
    Train for one epoch
    
    Args:
        model: Model to train
        data_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        logger: Logger
        epoch: Current epoch
        config: Model configuration
        
    Returns:
        epoch_loss: Average loss for the epoch
    """
    model.train()
    total_loss = 0
    total_samples = 0
    
    for i, batch in enumerate(data_loader):
        # Move data to device
        img = batch['img'].to(device)
        text_tokens = batch['text_tokens'].to(device)
        text_mask = batch['text_mask'].to(device)
        target = batch['target'].to(device)  # This is already normalized [0,1]
        
        # Forward pass - model outputs normalized coordinates [0,1]
        pred_boxes = model(img, text_tokens, text_mask)
        
        # Calculate loss
        loss, loss_dict = criterion(pred_boxes, target)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Apply gradient clipping to prevent exploding gradients
        if hasattr(config, 'gradient_clip_val') and config.gradient_clip_val > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.gradient_clip_val)
        
        optimizer.step()
        
        # Track metrics
        batch_size = img.shape[0]
        total_loss += loss.item() * batch_size
        total_samples += batch_size
        
        # Log every 50 batches
        if i % 50 == 0:
            logger.info(f"Epoch {epoch}, Batch {i}/{len(data_loader)}, Loss: {loss.item():.4f}")
            log_str = f"Train metrics: l1_loss: {loss_dict['l1_loss']:.4f} | giou_loss: {loss_dict['giou_loss']:.4f}"
            if 'center_loss' in loss_dict:
                log_str += f" | center_loss: {loss_dict['center_loss']:.4f}"
            log_str += f" | total_loss: {loss_dict['total_loss']:.4f}"
            logger.info(log_str)
    
    epoch_loss = total_loss / total_samples
    return epoch_loss


def validate(model, data_loader, criterion, device, logger, epoch):
    """
    Validate model
    
    Args:
        model: Model to validate
        data_loader: Validation data loader
        criterion: Loss function
        device: Device to use
        logger: Logger
        epoch: Current epoch
        
    Returns:
        metrics: Dictionary of validation metrics
    """
    model.eval()
    total_loss = 0
    total_samples = 0
    
    # Collect all predictions and targets for metric calculation
    all_pred_boxes = []
    all_target_boxes = []
    
    with torch.no_grad():
        for i, batch in enumerate(data_loader):
            # Move data to device
            img = batch['img'].to(device)
            text_tokens = batch['text_tokens'].to(device)
            text_mask = batch['text_mask'].to(device)
            target = batch['target'].to(device)  # Normalized [0,1]
            original_bbox = batch['original_bbox'].to(device)  # Original pixel coordinates
            orig_img_size = batch['orig_img_size'].to(device)  # Original image dimensions
            
            # Forward pass - model outputs normalized coordinates [0,1]
            pred_boxes = model(img, text_tokens, text_mask)
            
            # Calculate loss using normalized coordinates
            loss, loss_dict = criterion(pred_boxes, target)
            
            # Track metrics
            batch_size = img.shape[0]
            total_loss += loss.item() * batch_size
            total_samples += batch_size
            
            # Convert normalized predictions to pixel coordinates for original image size
            # This ensures correct comparison with original_bbox
            pred_boxes_scaled = torch.zeros_like(pred_boxes)
            for j in range(batch_size):
                img_w, img_h = orig_img_size[j]
                pred_boxes_scaled[j, 0] = pred_boxes[j, 0] * img_w  # x1
                pred_boxes_scaled[j, 1] = pred_boxes[j, 1] * img_h  # y1
                pred_boxes_scaled[j, 2] = pred_boxes[j, 2] * img_w  # x2
                pred_boxes_scaled[j, 3] = pred_boxes[j, 3] * img_h  # y2
            
            # Collect predictions and targets
            all_pred_boxes.append(pred_boxes_scaled.cpu())
            all_target_boxes.append(original_bbox.cpu())
    
    # Calculate validation loss
    val_loss = total_loss / total_samples
    
    # Concatenate all predictions and targets
    all_pred_boxes = torch.cat(all_pred_boxes, dim=0)
    all_target_boxes = torch.cat(all_target_boxes, dim=0)
    
    # Log first few examples for debugging
    indices = [0, len(all_pred_boxes)//2, len(all_pred_boxes)-1]  # Start, middle, end
    for i, idx in enumerate(indices):
        logger.info(f"Sample {i+1} - Pred: {all_pred_boxes[idx].tolist()}, Target: {all_target_boxes[idx].tolist()}")
    
    # Calculate metrics
    metrics = calculate_metrics(all_pred_boxes, all_target_boxes)
    metrics['loss'] = val_loss
    
    # Log metrics
    logger.info(f"Validation Epoch {epoch}, Loss: {val_loss:.4f}")
    metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
    logger.info(f"Val metrics: {metrics_str}")
    
    return metrics


def main():
    """Main training function"""
    # Parse arguments
    args = parse_args()
    
    # Load configuration
    config = DinoVitConfig()
    
    # Override config with command line arguments
    config.epochs = args.epochs
    config.batch_size = args.batch_size
    config.lr = args.lr
    config.lr_bert = args.lr_bert
    config.weight_decay = args.weight_decay
    config.lr_drop = args.lr_drop
    config.output_dir = args.output_dir
    config.data_root = args.data_root
    config.num_workers = args.num_workers
    
    # Set center prediction loss weight
    config.center_weight = args.center_weight
    
    # Data augmentation settings - enable by default, can be disabled with flags
    config.use_augmentation = not args.no_augmentation
    config.use_center_loss = not args.no_center_loss
    config.use_focal_loss = not args.no_focal_loss
    
    # Set up logging
    logger = get_logger(config, args.log_name)
    
    # Log configuration
    logger.info(f"Running with configuration: {vars(args)}")
    logger.info(f"Using DINO ViT backbone | Data augmentation: {config.use_augmentation}, Center loss: {config.use_center_loss}, Focal loss: {config.use_focal_loss}")
    
    # Set random seed for reproducibility
    seed = 17
    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    
    # Set device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Build data loaders with our custom dataloader
    logger.info("Building data loaders...")
    dataloaders = build_dataloaders(config, use_pin_memory=True)
    logger.info(f"Train dataset size: {len(dataloaders['train'].dataset)}")
    logger.info(f"Val dataset size: {len(dataloaders['val'].dataset)}")
    logger.info(f"Test dataset size: {len(dataloaders['test'].dataset)}")
    
    # Build model
    logger.info("Building model...")
    model = build_model(config)
    model = model.to(device)
    
    # Define loss function
    criterion = TransVGLoss(config)
    
    # Define optimizer
    optimizer = AdamW([
        {'params': model.vision_encoder.parameters(), 'lr': config.lr},
        {'params': model.language_encoder.parameters(), 'lr': config.lr_bert},
        {'params': model.bbox_head.parameters(), 'lr': config.lr},
        {'params': model.cross_encoder.parameters(), 'lr': config.lr},
        {'params': model.global_token, 'lr': config.lr}
    ], weight_decay=config.weight_decay)
    
    # Define learning rate scheduler
    if args.scheduler == 'cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=config.epochs, eta_min=1e-6)
        logger.info(f"Using cosine annealing scheduler with T_max={config.epochs}")
    else:
        scheduler = StepLR(optimizer, step_size=config.lr_drop, gamma=0.1)
        logger.info(f"Using step scheduler with step_size={config.lr_drop}, gamma=0.1")
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_metric = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint from {args.resume}")
            checkpoint = torch.load(args.resume, map_location=device)
            model.load_state_dict(checkpoint['model'])
            start_epoch = checkpoint['epoch'] + 1
            if 'metrics' in checkpoint and 'Acc@0.5' in checkpoint['metrics']:
                best_val_metric = checkpoint['metrics']['Acc@0.5']
        else:
            logger.warning(f"Checkpoint not found: {args.resume}")
    
    # Training loop
    logger.info("Starting training...")
    for epoch in range(start_epoch, config.epochs):
        logger.info(f"Epoch {epoch}/{config.epochs}")
        
        # Train
        train_loss = train_one_epoch(model, dataloaders['train'], criterion, optimizer, device, logger, epoch, config)
        logger.info(f"Train Epoch {epoch}, Loss: {train_loss:.4f}")
        
        # Validate
        val_metrics = validate(model, dataloaders['val'], criterion, device, logger, epoch)
        
        # Update scheduler
        scheduler.step()
        
        # Save checkpoint
        is_best = val_metrics['Acc@0.5'] > best_val_metric
        if is_best:
            best_val_metric = val_metrics['Acc@0.5']
        
        # Save model checkpoint
        logger.log_model(model, epoch, val_metrics)
        
        # Log best model so far
        if is_best:
            logger.info(f"New best model! Acc@0.5: {best_val_metric:.4f}")
    
    # Final evaluation on test set
    logger.info("Final evaluation on test set...")
    test_metrics = validate(model, dataloaders['test'], criterion, device, logger, config.epochs)
    logger.info(f"Test metrics: {test_metrics}")
    
    # Finish logging
    logger.finish()


if __name__ == "__main__":
    main()