"""
Logging utilities for TransVG model
Supports both terminal logging and (optionally) W&B logging
"""

import os
import json
import torch
import logging
import datetime
from pathlib import Path

# Flag to determine if wandb is available
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class Logger:
    """
    Logger class for TransVG training
    Can log to terminal, file, and optionally W&B
    """
    def __init__(self, config, log_name=None):
        """
        Initialize logger
        
        Args:
            config: Model configuration with logging parameters
            log_name: Optional log name, otherwise uses timestamp
        """
        self.config = config
        self.use_wandb = config.use_wandb and WANDB_AVAILABLE
        
        # Create log directory if it doesn't exist
        log_dir = Path(config.log_dir)
        log_dir.mkdir(exist_ok=True, parents=True)
        
        # Create checkpoint directory if it doesn't exist
        checkpoint_dir = Path(config.checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True, parents=True)
        
        # Create log name
        if log_name is None:
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            log_name = f"transvg_{timestamp}"
        
        self.log_name = log_name
        self.log_path = log_dir / f"{log_name}.log"
        
        # Set up logging to file and terminal
        logger = logging.getLogger(log_name)
        logger.setLevel(logging.INFO)
        
        # Create file handler which logs even debug messages
        fh = logging.FileHandler(self.log_path)
        fh.setLevel(logging.INFO)
        
        # Create console handler with a higher log level
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        
        # Create formatter and add it to the handlers
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)
        
        # Add the handlers to the logger
        logger.addHandler(fh)
        logger.addHandler(ch)
        
        self.logger = logger
        
        # Initialize W&B if enabled
        if self.use_wandb:
            wandb.init(
                project=config.project_name,
                name=log_name,
                config=config.__dict__
            )
        
        # Log model configuration
        self.info(f"Initialized logger: {log_name}")
        self.info(f"Configuration: {json.dumps(config.__dict__, default=str, indent=2)}")
    
    def info(self, message):
        """Log info message"""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message"""
        self.logger.warning(message)
    
    def error(self, message):
        """Log error message"""
        self.logger.error(message)
    
    def log_metrics(self, metrics, step=None, split='train'):
        """
        Log metrics to terminal and wandb if enabled
        
        Args:
            metrics: Dictionary of metrics to log
            step: Current step (optional)
            split: Data split (train, val, test)
        """
        # Add split prefix to metrics
        prefixed_metrics = {f"{split}/{k}": v for k, v in metrics.items()}
        
        # Format metrics for terminal logging
        metrics_str = " | ".join([f"{k}: {v:.4f}" for k, v in metrics.items()])
        self.info(f"{split.capitalize()} metrics: {metrics_str}")
        
        # Log to wandb if enabled
        if self.use_wandb:
            wandb.log(prefixed_metrics, step=step)
    
    def log_model(self, model, epoch, metrics):
        """
        Save model checkpoint
        
        Args:
            model: Model to save
            epoch: Current epoch
            metrics: Dictionary of metrics
        """
        # Create checkpoint path
        checkpoint_dir = Path(self.config.checkpoint_dir)
        checkpoint_path = checkpoint_dir / f"{self.log_name}_epoch{epoch}.pth"
        
        # Save model checkpoint
        checkpoint = {
            'epoch': epoch,
            'model': model.state_dict(),
            'metrics': metrics
        }
        
        torch.save(checkpoint, checkpoint_path)
        self.info(f"Saved model checkpoint to {checkpoint_path}")
        
        # Log model to wandb if enabled
        if self.use_wandb:
            wandb.save(str(checkpoint_path))
    
    def finish(self):
        """Finish logging (important for wandb)"""
        if self.use_wandb:
            wandb.finish()
        self.info("Finished logging session")


def get_logger(config, log_name=None):
    """
    Get logger instance
    
    Args:
        config: Model configuration
        log_name: Optional log name
        
    Returns:
        logger: Logger instance
    """
    return Logger(config, log_name) 