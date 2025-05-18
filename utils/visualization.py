"""
Visualization utilities for debugging and analysis
"""

import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
from pathlib import Path

def visualize_ground_truth(img, text, bbox, image_id, output_path):
    """
    Visualize image with ground truth bounding box
    
    Args:
        img: Input image tensor (C, H, W)
        text: Text query
        bbox: Ground truth bounding box [x1, y1, x2, y2]
        image_id: Image identifier
        output_path: Path to save visualization
    """
    # Convert tensor to numpy array
    img_np = img.permute(1, 2, 0).cpu().numpy()
    
    # Denormalize image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = img_np * std + mean
    img_np = np.clip(img_np, 0, 1)
    
    # Create figure
    fig, ax = plt.subplots(1, figsize=(10, 10))
    ax.imshow(img_np)
    
    # Get box coordinates
    bbox = bbox.cpu().numpy()
    
    # Draw ground truth box (green)
    rect = patches.Rectangle(
        (bbox[0], bbox[1]),
        bbox[2] - bbox[0],
        bbox[3] - bbox[1],
        linewidth=2,
        edgecolor='g',
        facecolor='none',
        label='Ground Truth'
    )
    ax.add_patch(rect)
    
    # Add query text and image ID
    ax.set_title(f"Query: {text}\nImage ID: {image_id}", fontsize=12)
    
    # Add legend
    ax.legend()
    
    # Remove axes
    ax.axis('off')
    
    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def visualize_batch(batch, output_dir, max_samples=16):
    """
    Visualize a batch of samples with ground truth bounding boxes
    
    Args:
        batch: Batch of samples from dataloader
        output_dir: Directory to save visualizations
        max_samples: Maximum number of samples to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract data from batch
    images = batch['img']
    targets = batch['original_bbox']
    image_ids = batch['image_id']
    
    # Try to get the raw text if available in the batch
    texts = batch.get('text', [f"Query #{i}" for i in range(len(images))])
    
    # Cap the number of samples to visualize
    num_samples = min(len(images), max_samples)
    
    # Visualize each sample
    for i in range(num_samples):
        output_path = os.path.join(output_dir, f"gt_sample_{i}_{image_ids[i]}.png")
        visualize_ground_truth(
            images[i],
            texts[i] if isinstance(texts, list) else f"Query #{i}",
            targets[i],
            image_ids[i],
            output_path
        )
    
    print(f"Visualized {num_samples} samples in {output_dir}")

def visualize_dataset_samples(dataloader, output_dir, num_samples=16):
    """
    Visualize random samples from the dataset with ground truth bounding boxes
    
    Args:
        dataloader: DataLoader containing dataset samples
        output_dir: Directory to save visualizations
        num_samples: Number of samples to visualize
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get a batch from the dataloader
    for batch in dataloader:
        # Visualize the batch
        visualize_batch(batch, output_dir, num_samples)
        break  # Only visualize one batch 