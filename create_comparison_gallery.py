#!/usr/bin/env python3
"""
Create a comparison gallery showing best and worst performing samples
for quick academic presentation overview
"""

import os
import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image
from pathlib import Path


def load_inference_results(results_dir):
    """Load inference results from directory"""
    detailed_metrics_path = os.path.join(results_dir, 'detailed_metrics.json')
    
    if not os.path.exists(detailed_metrics_path):
        raise FileNotFoundError(f"Detailed metrics not found: {detailed_metrics_path}")
    
    with open(detailed_metrics_path, 'r') as f:
        data = json.load(f)
    
    return data['samples']


def create_comparison_gallery(results_dir, output_path, num_best=5, num_worst=5):
    """
    Create a comparison gallery showing best and worst performing samples
    
    Args:
        results_dir: Directory containing inference results
        output_path: Path to save the gallery
        num_best: Number of best samples to show
        num_worst: Number of worst samples to show
    """
    # Load results
    samples = load_inference_results(results_dir)
    
    # Sort by IoU
    samples_sorted = sorted(samples, key=lambda x: x['iou'], reverse=True)
    
    # Get best and worst samples
    best_samples = samples_sorted[:num_best]
    worst_samples = samples_sorted[-num_worst:]
    
    # Create figure
    fig, axes = plt.subplots(2, max(num_best, num_worst), figsize=(20, 8))
    
    # Handle single row case
    if max(num_best, num_worst) == 1:
        axes = axes.reshape(2, 1)
    
    # Plot best samples
    for i in range(num_best):
        sample = best_samples[i]
        ax = axes[0, i]
        
        # Load and display image
        img_path = f"/home/pokle/Trans-VG/visual_grounding/dior-rsvg/JPEGImages/{sample['image_name']}"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            
            # Add bounding boxes
            pred_box = sample['pred_box']
            target_box = sample['target_box']
            
            # Ground truth (green)
            gt_rect = patches.Rectangle(
                (target_box[0], target_box[1]), 
                target_box[2] - target_box[0], 
                target_box[3] - target_box[1],
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(gt_rect)
            
            # Prediction (red)
            pred_rect = patches.Rectangle(
                (pred_box[0], pred_box[1]), 
                pred_box[2] - pred_box[0], 
                pred_box[3] - pred_box[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(pred_rect)
            
            # Title with IoU
            ax.set_title(f"Best #{i+1}\nIoU: {sample['iou']:.3f}", fontsize=10, fontweight='bold')
            
            # Add text query as subtitle (truncated)
            text_truncated = sample['text'][:40] + "..." if len(sample['text']) > 40 else sample['text']
            ax.text(0.5, -0.1, f'"{text_truncated}"', transform=ax.transAxes, 
                   ha='center', va='top', fontsize=8, style='italic')
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Best #{i+1}\nIoU: {sample['iou']:.3f}", fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Plot worst samples
    for i in range(num_worst):
        sample = worst_samples[i]
        ax = axes[1, i]
        
        # Load and display image
        img_path = f"/home/pokle/Trans-VG/visual_grounding/dior-rsvg/JPEGImages/{sample['image_name']}"
        if os.path.exists(img_path):
            img = Image.open(img_path).convert('RGB')
            ax.imshow(img)
            
            # Add bounding boxes
            pred_box = sample['pred_box']
            target_box = sample['target_box']
            
            # Ground truth (green)
            gt_rect = patches.Rectangle(
                (target_box[0], target_box[1]), 
                target_box[2] - target_box[0], 
                target_box[3] - target_box[1],
                linewidth=2, edgecolor='green', facecolor='none'
            )
            ax.add_patch(gt_rect)
            
            # Prediction (red)
            pred_rect = patches.Rectangle(
                (pred_box[0], pred_box[1]), 
                pred_box[2] - pred_box[0], 
                pred_box[3] - pred_box[1],
                linewidth=2, edgecolor='red', facecolor='none'
            )
            ax.add_patch(pred_rect)
            
            # Title with IoU
            ax.set_title(f"Worst #{i+1}\nIoU: {sample['iou']:.3f}", fontsize=10, fontweight='bold')
            
            # Add text query as subtitle (truncated)
            text_truncated = sample['text'][:40] + "..." if len(sample['text']) > 40 else sample['text']
            ax.text(0.5, -0.1, f'"{text_truncated}"', transform=ax.transAxes, 
                   ha='center', va='top', fontsize=8, style='italic')
        else:
            ax.text(0.5, 0.5, "Image not found", ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Worst #{i+1}\nIoU: {sample['iou']:.3f}", fontsize=10)
        
        ax.set_xticks([])
        ax.set_yticks([])
    
    # Hide unused subplots
    total_plots = max(num_best, num_worst)
    if num_best < total_plots:
        for i in range(num_best, total_plots):
            axes[0, i].set_visible(False)
    if num_worst < total_plots:
        for i in range(num_worst, total_plots):
            axes[1, i].set_visible(False)
    
    # Add overall title and legend
    fig.suptitle('TransVG Model Performance: Best vs Worst Predictions\nGreen: Ground Truth, Red: Prediction', 
                fontsize=16, fontweight='bold')
    
    # Add row labels
    fig.text(0.02, 0.75, 'BEST\nPERFORMING', rotation=90, ha='center', va='center', 
            fontsize=14, fontweight='bold', color='darkgreen')
    fig.text(0.02, 0.25, 'WORST\nPERFORMING', rotation=90, ha='center', va='center', 
            fontsize=14, fontweight='bold', color='darkred')
    
    plt.tight_layout()
    plt.subplots_adjust(left=0.05, top=0.85)
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparison gallery saved to: {output_path}")


def create_performance_histogram(results_dir, output_path):
    """Create histogram of IoU performance distribution"""
    samples = load_inference_results(results_dir)
    ious = [s['iou'] for s in samples]
    
    plt.figure(figsize=(10, 6))
    plt.hist(ious, bins=30, alpha=0.7, edgecolor='black')
    plt.axvline(float(np.mean(ious)), color='red', linestyle='--', linewidth=2, label=f'Mean: {np.mean(ious):.3f}')
    plt.axvline(float(np.median(ious)), color='green', linestyle='--', linewidth=2, label=f'Median: {np.median(ious):.3f}')
    
    plt.xlabel('IoU Score')
    plt.ylabel('Frequency')
    plt.title('Distribution of IoU Scores')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Add performance thresholds
    plt.axvline(0.25, color='orange', linestyle=':', alpha=0.7, label='IoU=0.25')
    plt.axvline(0.5, color='blue', linestyle=':', alpha=0.7, label='IoU=0.5')
    plt.axvline(0.75, color='purple', linestyle=':', alpha=0.7, label='IoU=0.75')
    
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Performance histogram saved to: {output_path}")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python create_comparison_gallery.py <results_directory>")
        sys.exit(1)
    
    results_dir = sys.argv[1]
    
    if not os.path.exists(results_dir):
        print(f"Error: Results directory not found: {results_dir}")
        sys.exit(1)
    
    # Create comparison gallery
    gallery_path = os.path.join(results_dir, 'comparison_gallery.png')
    create_comparison_gallery(results_dir, gallery_path)
    
    # Create performance histogram
    histogram_path = os.path.join(results_dir, 'performance_histogram.png')
    create_performance_histogram(results_dir, histogram_path)
    
    print(f"Created additional visualizations in: {results_dir}") 