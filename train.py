"""
Training script for TransVG model
"""

import os
import sys
import torch
import argparse
import numpy as np
from pathlib import Path
from torch.optim.adam import Adam
from torch.optim.adamw import AdamW
from torch.optim.lr_scheduler import StepLR

# Add parent directory to path to import modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from models.transvg import build_model
from models.losses import TransVGLoss
from models.custom_dataloader import build_dataloaders
from utils.logger import get_logger
from utils.metrics import calculate_metrics
from configs.model_config import ModelConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description="Train TransVG model")
    parser.add_argument("--config", type=str, default="", help="Path to config file")
    parser.add_argument("--log_name", type=str, default=None, help="Name for log files")
    parser.add_argument("--use_wandb", action="store_true", help="Use Weights & Biases for logging")
    parser.add_argument("--epochs", type=int, default=None, help="Number of epochs to train")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size")
    parser.add_argument("--lr", type=float, default=None, help="Learning rate")
    parser.add_argument("--resume", type=str, default="", help="Resume from checkpoint")
    parser.add_argument("--use_pin_memory", action="store_true", help="Use pin_memory in data loading")
    parser.add_argument("--device", type=str, default=None, help="Device to use (cpu, cuda:0, cuda:1, etc)")
    parser.add_argument("--freeze_both", action="store_true", help="Freeze both visual and linguistic backbones")
    parser.add_argument("--partial_freeze_vision", action="store_true", help="Partially freeze vision backbone (early layers only)")
    parser.add_argument("--partial_freeze_linguistic", action="store_true", help="Partially freeze linguistic backbone (embeddings and early layers only)")
    
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
        config: Model configuration for additional parameters like gradient clipping
        
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
            step = epoch * len(data_loader) + i
            logger.info(f"Epoch {epoch}, Batch {i}/{len(data_loader)}, Loss: {loss.item():.4f}")
            logger.info(f"Train metrics: l1_loss: {loss_dict['l1_loss']:.4f} | giou_loss: {loss_dict['giou_loss']:.4f} | total_loss: {loss_dict['total_loss']:.4f}")
    
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
    logger.info(f"Pred box sample: {all_pred_boxes[0].tolist()}")
    logger.info(f"Target box sample: {all_target_boxes[0].tolist()}")
    
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
    config = ModelConfig()
    
    # Override config with command line arguments
    if args.epochs is not None:
        config.epochs = args.epochs
    if args.batch_size is not None:
        config.batch_size = args.batch_size
    if args.lr is not None:
        config.lr = args.lr
    if args.use_wandb:
        config.use_wandb = True
    if args.freeze_both:
        config.freeze_backbone = True
        config.freeze_linguistic = True
        logger_msg = "Both backbones will be frozen"
    else:
        logger_msg = f"Visual backbone freeze: {config.freeze_backbone}, Linguistic backbone freeze: {config.freeze_linguistic}"
    
    # Handle partial freezing options
    if args.partial_freeze_vision:
        config.partial_freeze_vision = True
        logger_msg += " (vision: partially)"
    
    if args.partial_freeze_linguistic:
        config.partial_freeze_linguistic = True
        logger_msg += " (linguistic: partially)"
    
    # Set up logging
    logger = get_logger(config, args.log_name)
    
    # Set random seed for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Set device
    device = torch.device(args.device if args.device else ("cuda" if torch.cuda.is_available() else "cpu"))
    logger.info(f"Using device: {device}")
    logger.info(logger_msg)
    
    # Build data loaders with our custom dataloader
    logger.info("Building data loaders...")
    dataloaders = build_dataloaders(config, use_pin_memory=args.use_pin_memory)
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
        {'params': model.bbox_head.parameters(), 'lr': config.lr}
    ], weight_decay=config.weight_decay)
    
    # Define learning rate scheduler
    scheduler = StepLR(optimizer, step_size=config.lr_drop, gamma=0.1)
    
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