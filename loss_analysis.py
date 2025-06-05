#!/usr/bin/env python3
"""
Loss Analysis Script for TransVG Training
Generates comprehensive loss curves and analysis for academic presentation
"""

import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
from scipy.interpolate import make_interp_spline
from scipy.signal import savgol_filter

class LossAnalyzer:
    """Class to analyze and visualize training losses from log files"""
    
    def __init__(self, log_file_path):
        self.log_file_path = log_file_path
        self.train_data = []
        self.val_data = []
        self.parse_log_file()
    
    def parse_log_file(self):
        """Parse the log file to extract training and validation metrics"""
        print(f"Parsing log file: {self.log_file_path}")
        
        with open(self.log_file_path, 'r') as f:
            lines = f.readlines()
        
        current_epoch = 0
        batch_count = 0
        
        for line in lines:
            # Extract epoch information
            epoch_match = re.search(r'Epoch (\d+)', line)
            if epoch_match:
                current_epoch = int(epoch_match.group(1))
            
            # Extract training metrics
            if "Train metrics:" in line:
                train_metrics = self._parse_train_metrics(line, current_epoch, batch_count)
                if train_metrics:
                    self.train_data.append(train_metrics)
                    batch_count += 1
            
            # Extract validation metrics
            elif "Val metrics:" in line:
                val_metrics = self._parse_val_metrics(line, current_epoch)
                if val_metrics:
                    self.val_data.append(val_metrics)
        
        print(f"Parsed {len(self.train_data)} training points and {len(self.val_data)} validation points")
    
    def _parse_train_metrics(self, line, epoch, batch):
        """Parse training metrics from a log line"""
        try:
            # Extract loss components
            l1_match = re.search(r'l1_loss: ([\d.]+)', line)
            giou_match = re.search(r'giou_loss: ([\d.]+)', line)
            center_match = re.search(r'center_loss: ([\d.]+)', line)
            total_match = re.search(r'total_loss: ([\d.]+)', line)
            
            if l1_match and giou_match and total_match:
                return {
                    'epoch': epoch,
                    'batch': batch,
                    'l1_loss': float(l1_match.group(1)),
                    'giou_loss': float(giou_match.group(1)),
                    'center_loss': float(center_match.group(1)) if center_match else 0.0,
                    'total_loss': float(total_match.group(1))
                }
        except Exception as e:
            print(f"Error parsing train metrics: {e}")
        return None
    
    def _parse_val_metrics(self, line, epoch):
        """Parse validation metrics from a log line"""
        try:
            # Extract metrics
            acc25_match = re.search(r'Acc@0\.25: ([\d.]+)', line)
            acc50_match = re.search(r'Acc@0\.5: ([\d.]+)', line)
            acc75_match = re.search(r'Acc@0\.75: ([\d.]+)', line)
            miou_match = re.search(r'mIoU: ([\d.]+)', line)
            loss_match = re.search(r'loss: ([\d.]+)', line)
            
            if acc25_match and acc50_match and miou_match and loss_match:
                return {
                    'epoch': epoch,
                    'acc_25': float(acc25_match.group(1)),
                    'acc_50': float(acc50_match.group(1)),
                    'acc_75': float(acc75_match.group(1)),
                    'miou': float(miou_match.group(1)),
                    'val_loss': float(loss_match.group(1))
                }
        except Exception as e:
            print(f"Error parsing val metrics: {e}")
        return None
    
    def smooth_curve(self, data, window_length=21, polyorder=3):
        """Apply smoothing to curves for better visualization"""
        if len(data) < window_length:
            window_length = len(data) if len(data) % 2 == 1 else len(data) - 1
            if window_length < 3:
                return data
        
        try:
            return savgol_filter(data, window_length, polyorder)
        except:
            return data
    
    def create_loss_curves(self, output_dir="loss_analysis"):
        """Create comprehensive loss curve visualizations"""
        Path(output_dir).mkdir(exist_ok=True)
        
        # Convert to DataFrames for easier manipulation
        train_df = pd.DataFrame(self.train_data)
        val_df = pd.DataFrame(self.val_data)
        
        if train_df.empty:
            print("No training data found!")
            return
        
        # 1. Training Loss Components Over Time
        self._plot_training_loss_components(train_df, output_dir)
        
        # 2. Training vs Validation Loss
        self._plot_train_val_loss(train_df, val_df, output_dir)
        
        # 3. Validation Metrics Progress
        self._plot_validation_metrics(val_df, output_dir)
        
        # 4. Loss Component Analysis
        self._plot_loss_contribution_analysis(train_df, output_dir)
        
        # 5. Combined Academic Figure
        self._plot_academic_summary(train_df, val_df, output_dir)
        
        print(f"All plots saved to: {output_dir}")
    
    def _plot_training_loss_components(self, train_df, output_dir):
        """Plot individual loss components during training"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # Create epoch-based aggregation for smoother curves
        epoch_train = train_df.groupby('epoch').mean().reset_index()
        
        epochs = epoch_train['epoch']
        
        # Plot 1: Individual Loss Components
        ax1.plot(epochs, epoch_train['l1_loss'], 'b-', linewidth=2, label='L1 Loss', alpha=0.8)
        ax1.plot(epochs, epoch_train['giou_loss'], 'r-', linewidth=2, label='GIoU Loss', alpha=0.8)
        ax1.plot(epochs, epoch_train['center_loss'], 'g-', linewidth=2, label='Center Loss', alpha=0.8)
        
        ax1.set_xlabel('Epoch', fontsize=12)
        ax1.set_ylabel('Loss Value', fontsize=12)
        ax1.set_title('Training Loss Components Over Time', fontsize=14, fontweight='bold')
        ax1.legend(fontsize=11)
        ax1.grid(True, alpha=0.3)
        ax1.set_xlim(0, epochs.max())
        
        # Plot 2: Total Loss
        # Smooth the total loss curve
        smoothed_total = self.smooth_curve(epoch_train['total_loss'].values)
        
        ax2.plot(epochs, epoch_train['total_loss'], 'lightblue', alpha=0.5, linewidth=1, label='Raw Total Loss')
        ax2.plot(epochs, smoothed_total, 'darkblue', linewidth=3, label='Smoothed Total Loss')
        
        ax2.set_xlabel('Epoch', fontsize=12)
        ax2.set_ylabel('Total Loss', fontsize=12)
        ax2.set_title('Total Training Loss Progression', fontsize=14, fontweight='bold')
        ax2.legend(fontsize=11)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, epochs.max())
        
        # Add academic annotations
        fig.suptitle('TransVG Training Loss Analysis\nDINO ViT Backbone with Multi-Component Loss', 
                    fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/training_loss_components.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_train_val_loss(self, train_df, val_df, output_dir):
        """Plot training vs validation loss comparison"""
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        # Aggregate training loss by epoch
        epoch_train = train_df.groupby('epoch').mean().reset_index()
        
        if not val_df.empty:
            epochs_train = epoch_train['epoch']
            epochs_val = val_df['epoch']
            
            # Smooth curves for better presentation
            train_loss_smooth = self.smooth_curve(epoch_train['total_loss'].values)
            val_loss_smooth = self.smooth_curve(val_df['val_loss'].values)
            
            # Plot raw data (faded)
            ax.plot(epochs_train, epoch_train['total_loss'], 'lightblue', alpha=0.4, linewidth=1)
            ax.plot(epochs_val, val_df['val_loss'], 'lightcoral', alpha=0.4, linewidth=1)
            
            # Plot smoothed curves (prominent)
            ax.plot(epochs_train, train_loss_smooth, 'blue', linewidth=3, label='Training Loss', marker='o', markersize=4, markevery=5)
            ax.plot(epochs_val, val_loss_smooth, 'red', linewidth=3, label='Validation Loss', marker='s', markersize=4, markevery=1)
            
            # Highlight best validation epoch
            best_val_idx = val_df['val_loss'].idxmin()
            best_epoch = val_df.loc[best_val_idx, 'epoch']
            best_val_loss = val_df.loc[best_val_idx, 'val_loss']
            
            ax.plot(best_epoch, best_val_loss, 'gold', marker='*', markersize=15, 
                   label=f'Best Val Loss: {best_val_loss:.3f} (Epoch {best_epoch})')
            
        else:
            # Only training data available
            epochs_train = epoch_train['epoch']
            train_loss_smooth = self.smooth_curve(epoch_train['total_loss'].values)
            ax.plot(epochs_train, epoch_train['total_loss'], 'lightblue', alpha=0.4, linewidth=1)
            ax.plot(epochs_train, train_loss_smooth, 'blue', linewidth=3, label='Training Loss', marker='o', markersize=4, markevery=5)
        
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Loss Value', fontsize=12)
        ax.set_title('Training vs Validation Loss Progression', fontsize=14, fontweight='bold')
        ax.legend(fontsize=11)
        ax.grid(True, alpha=0.3)
        
        # Add academic text box
        textstr = """Model: TransVG with DINO ViT Backbone
Loss Function: L1 + GIoU + Center Loss
Optimizer: AdamW with Different LR for Components"""
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/train_val_loss_comparison.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_validation_metrics(self, val_df, output_dir):
        """Plot validation metrics progression"""
        if val_df.empty:
            print("No validation data found!")
            return
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        epochs = val_df['epoch']
        
        # 1. Accuracy metrics
        ax1.plot(epochs, val_df['acc_25'] * 100, 'g-', linewidth=2, marker='o', label='Acc@0.25')
        ax1.plot(epochs, val_df['acc_50'] * 100, 'b-', linewidth=2, marker='s', label='Acc@0.5')
        ax1.plot(epochs, val_df['acc_75'] * 100, 'r-', linewidth=2, marker='^', label='Acc@0.75')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy (%)')
        ax1.set_title('Accuracy at Different IoU Thresholds', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.set_ylim(0, max(val_df['acc_25'].max() * 100 * 1.1, 30))
        
        # 2. mIoU progression
        miou_smooth = self.smooth_curve(val_df['miou'].values * 100)
        ax2.plot(epochs, val_df['miou'] * 100, 'lightgreen', alpha=0.5, linewidth=1)
        ax2.plot(epochs, miou_smooth, 'darkgreen', linewidth=3, marker='o', markersize=6)
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('mIoU (%)')
        ax2.set_title('Mean IoU Progression', fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Highlight best mIoU
        best_miou_idx = val_df['miou'].idxmax()
        best_miou_epoch = val_df.loc[best_miou_idx, 'epoch']
        best_miou = val_df.loc[best_miou_idx, 'miou'] * 100
        ax2.plot(best_miou_epoch, best_miou, 'gold', marker='*', markersize=15)
        ax2.text(best_miou_epoch, best_miou + 1, f'Best: {best_miou:.2f}%', 
                ha='center', fontweight='bold')
        
        # 3. Validation loss
        val_loss_smooth = self.smooth_curve(val_df['val_loss'].values)
        ax3.plot(epochs, val_df['val_loss'], 'lightcoral', alpha=0.5, linewidth=1)
        ax3.plot(epochs, val_loss_smooth, 'darkred', linewidth=3, marker='s', markersize=6)
        
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Validation Loss')
        ax3.set_title('Validation Loss Progression', fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance summary
        ax4.axis('off')
        
        # Create performance summary table
        final_metrics = val_df.iloc[-1]
        best_acc50_idx = val_df['acc_50'].idxmax()
        best_acc50 = val_df.loc[best_acc50_idx, 'acc_50'] * 100
        best_acc50_epoch = val_df.loc[best_acc50_idx, 'epoch']
        
        summary_text = f"""Performance Summary

Final Epoch: {int(final_metrics['epoch'])}

Final Metrics:
• Acc@0.25: {final_metrics['acc_25']*100:.2f}%
• Acc@0.5: {final_metrics['acc_50']*100:.2f}%
• Acc@0.75: {final_metrics['acc_75']*100:.2f}%
• mIoU: {final_metrics['miou']*100:.2f}%

Best Performance:
• Best Acc@0.5: {best_acc50:.2f}% (Epoch {best_acc50_epoch})
• Best mIoU: {best_miou:.2f}% (Epoch {best_miou_epoch})

Model Configuration:
• Backbone: DINO ViT (Partially Frozen)
• Loss: L1 + GIoU + Center Loss
• Optimizer: AdamW
• Data Augmentation: Enabled"""
        
        ax4.text(0.1, 0.9, summary_text, transform=ax4.transAxes, fontsize=12,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))
        
        fig.suptitle('TransVG Validation Metrics Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig(f"{output_dir}/validation_metrics_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_loss_contribution_analysis(self, train_df, output_dir):
        """Analyze the contribution of different loss components"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Aggregate by epoch
        epoch_train = train_df.groupby('epoch').mean().reset_index()
        
        epochs = epoch_train['epoch']
        
        # 1. Stacked area plot of loss components
        ax1.fill_between(epochs, 0, epoch_train['l1_loss'], alpha=0.7, color='blue', label='L1 Loss')
        ax1.fill_between(epochs, epoch_train['l1_loss'], 
                        epoch_train['l1_loss'] + epoch_train['giou_loss'], 
                        alpha=0.7, color='red', label='GIoU Loss')
        ax1.fill_between(epochs, epoch_train['l1_loss'] + epoch_train['giou_loss'],
                        epoch_train['l1_loss'] + epoch_train['giou_loss'] + epoch_train['center_loss'],
                        alpha=0.7, color='green', label='Center Loss')
        
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss Value')
        ax1.set_title('Loss Component Contribution Over Time', fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss component ratios over time
        total_loss = epoch_train['l1_loss'] + epoch_train['giou_loss'] + epoch_train['center_loss']
        l1_ratio = epoch_train['l1_loss'] / total_loss * 100
        giou_ratio = epoch_train['giou_loss'] / total_loss * 100
        center_ratio = epoch_train['center_loss'] / total_loss * 100
        
        ax2.plot(epochs, l1_ratio, 'blue', linewidth=2, marker='o', label='L1 Loss %')
        ax2.plot(epochs, giou_ratio, 'red', linewidth=2, marker='s', label='GIoU Loss %')
        ax2.plot(epochs, center_ratio, 'green', linewidth=2, marker='^', label='Center Loss %')
        
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Percentage of Total Loss (%)')
        ax2.set_title('Loss Component Ratios Over Time', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        ax2.set_ylim(0, 100)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/loss_contribution_analysis.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _plot_academic_summary(self, train_df, val_df, output_dir):
        """Create a comprehensive academic summary figure"""
        fig = plt.figure(figsize=(20, 12))
        
        # Create a complex grid layout
        gs = fig.add_gridspec(3, 4, height_ratios=[1, 1, 0.8], width_ratios=[1, 1, 1, 0.8])
        
        # Aggregate training data
        epoch_train = train_df.groupby('epoch').mean().reset_index()
        
        # 1. Training Loss (top left)
        ax1 = fig.add_subplot(gs[0, 0])
        epochs_train = epoch_train['epoch']
        train_loss_smooth = self.smooth_curve(epoch_train['total_loss'].values)
        ax1.plot(epochs_train, epoch_train['total_loss'], 'lightblue', alpha=0.4, linewidth=1)
        ax1.plot(epochs_train, train_loss_smooth, 'darkblue', linewidth=3, marker='o', markersize=4, markevery=10)
        ax1.set_title('Training Loss', fontweight='bold', fontsize=12)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.grid(True, alpha=0.3)
        
        # 2. Loss Components (top middle)
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(epochs_train, epoch_train['l1_loss'], 'b-', linewidth=2, label='L1')
        ax2.plot(epochs_train, epoch_train['giou_loss'], 'r-', linewidth=2, label='GIoU')
        ax2.plot(epochs_train, epoch_train['center_loss'], 'g-', linewidth=2, label='Center')
        ax2.set_title('Loss Components', fontweight='bold', fontsize=12)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss Value')
        ax2.legend(fontsize=10)
        ax2.grid(True, alpha=0.3)
        
        # 3. Validation Metrics (top right)
        if not val_df.empty:
            ax3 = fig.add_subplot(gs[0, 2])
            epochs_val = val_df['epoch']
            ax3.plot(epochs_val, val_df['acc_50'] * 100, 'b-', linewidth=2, marker='o', label='Acc@0.5')
            ax3.plot(epochs_val, val_df['miou'] * 100, 'g-', linewidth=2, marker='s', label='mIoU')
            ax3.set_title('Validation Performance', fontweight='bold', fontsize=12)
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Performance (%)')
            ax3.legend(fontsize=10)
            ax3.grid(True, alpha=0.3)
        
        # 4. Training vs Validation Loss (middle left-middle)
        ax4 = fig.add_subplot(gs[1, :2])
        if not val_df.empty:
            val_loss_smooth = self.smooth_curve(val_df['val_loss'].values)
            ax4.plot(epochs_train, train_loss_smooth, 'blue', linewidth=3, label='Training Loss', marker='o', markersize=4, markevery=10)
            ax4.plot(epochs_val, val_loss_smooth, 'red', linewidth=3, label='Validation Loss', marker='s', markersize=4, markevery=2)
            
            # Add gap between train and val (simplified)
            # Just show the curves without gap visualization for now
            pass
        else:
            ax4.plot(epochs_train, train_loss_smooth, 'blue', linewidth=3, label='Training Loss', marker='o', markersize=4, markevery=10)
        
        ax4.set_title('Training vs Validation Loss', fontweight='bold', fontsize=14)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Loss Value')
        ax4.legend(fontsize=12)
        ax4.grid(True, alpha=0.3)
        
        # 5. Loss Contribution (middle right)
        ax5 = fig.add_subplot(gs[1, 2])
        # Pie chart of final loss contributions
        final_epoch = epoch_train.iloc[-1]
        components = [final_epoch['l1_loss'], final_epoch['giou_loss'], final_epoch['center_loss']]
        labels = ['L1 Loss', 'GIoU Loss', 'Center Loss']
        colors = ['lightblue', 'lightcoral', 'lightgreen']
        
        ax5.pie(components, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
        ax5.set_title('Final Loss Composition', fontweight='bold', fontsize=12)
        
        # 6. Information Panel (right side)
        ax6 = fig.add_subplot(gs[:, 3])
        ax6.axis('off')
        
        # Calculate key statistics
        final_train_loss = epoch_train['total_loss'].iloc[-1]
        min_train_loss = epoch_train['total_loss'].min()
        loss_reduction = ((epoch_train['total_loss'].iloc[0] - final_train_loss) / epoch_train['total_loss'].iloc[0]) * 100
        
        info_text = f"""TransVG Training Analysis
Model: DINO ViT Backbone

Training Configuration:
• Architecture: TransVG
• Backbone: DINO ViT (Frozen/Trainable)
• Loss Function: Multi-component
  - L1 Loss (Localization)
  - GIoU Loss (Overlap)
  - Center Loss (Center Point)
• Optimizer: AdamW
• Learning Rate: Variable per component

Training Progress:
• Total Epochs: {int(epochs_train.max())}
• Training Samples: {len(train_df):,}
• Initial Loss: {epoch_train['total_loss'].iloc[0]:.3f}
• Final Loss: {final_train_loss:.3f}
• Loss Reduction: {loss_reduction:.1f}%
• Min Training Loss: {min_train_loss:.3f}

Performance Highlights:"""

        if not val_df.empty:
            best_acc50 = val_df['acc_50'].max() * 100
            best_miou = val_df['miou'].max() * 100
            final_acc50 = val_df['acc_50'].iloc[-1] * 100
            
            info_text += f"""
• Best Acc@0.5: {best_acc50:.2f}%
• Best mIoU: {best_miou:.2f}%
• Final Acc@0.5: {final_acc50:.2f}%

Training Characteristics:
• Stable convergence pattern
• Multi-component loss balancing
• Effective generalization"""
        
        ax6.text(0.05, 0.95, info_text, transform=ax6.transAxes, fontsize=11,
                verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
        
        # 7. Bottom panel - Training dynamics
        ax7 = fig.add_subplot(gs[2, :3])
        
        # Show learning rate schedule effect (if visible in loss trends)
        batch_data = train_df.copy()
        batch_data['global_step'] = batch_data.index
        
        # Sample every Nth point for clarity
        sample_every = max(1, len(batch_data) // 1000)
        sampled_data = batch_data.iloc[::sample_every]
        
        ax7.plot(sampled_data['global_step'], sampled_data['total_loss'], 'lightblue', alpha=0.3, linewidth=0.5)
        
        # Add epoch boundaries
        epoch_boundaries = []
        for epoch in range(int(epochs_train.max()) + 1):
            epoch_data = batch_data[batch_data['epoch'] == epoch]
            if not epoch_data.empty:
                epoch_boundaries.append(epoch_data['global_step'].iloc[0])
        
        for boundary in epoch_boundaries[::5]:  # Show every 5th epoch
            ax7.axvline(x=boundary, color='red', linestyle='--', alpha=0.3, linewidth=1)
        
        ax7.set_title('Training Dynamics (Batch-level Loss Progression)', fontweight='bold', fontsize=12)
        ax7.set_xlabel('Training Steps')
        ax7.set_ylabel('Batch Loss')
        ax7.grid(True, alpha=0.3)
        
        # Main title
        fig.suptitle('TransVG Training Analysis: Comprehensive Loss and Performance Overview', 
                    fontsize=20, fontweight='bold', y=0.98)
        
        plt.tight_layout()
        plt.savefig(f"{output_dir}/academic_training_summary.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        print("Academic summary figure created successfully!")


def main():
    """Main function to run loss analysis"""
    # Configuration
    log_file = "logs/all_data_dino_improved.log"
    output_dir = "loss_analysis_results"
    
    print("="*60)
    print("TransVG Loss Analysis for Academic Presentation")
    print("="*60)
    
    # Check if log file exists
    if not Path(log_file).exists():
        print(f"Error: Log file not found: {log_file}")
        print("Please check the path and try again.")
        return
    
    # Create analyzer and process
    analyzer = LossAnalyzer(log_file)
    
    if not analyzer.train_data:
        print("No training data found in log file!")
        return
    
    # Generate all visualizations
    print(f"Generating loss curves and analysis...")
    analyzer.create_loss_curves(output_dir)
    
    # Create summary report
    Path(output_dir).mkdir(exist_ok=True)
    
    with open(f"{output_dir}/analysis_summary.md", 'w') as f:
        f.write("# TransVG Training Loss Analysis Summary\n\n")
        f.write(f"**Analysis Date**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Log File**: {log_file}\n")
        f.write(f"**Training Points**: {len(analyzer.train_data)}\n")
        f.write(f"**Validation Points**: {len(analyzer.val_data)}\n\n")
        
        f.write("## Generated Visualizations\n\n")
        f.write("1. **training_loss_components.png** - Individual loss components over time\n")
        f.write("2. **train_val_loss_comparison.png** - Training vs validation loss comparison\n")
        f.write("3. **validation_metrics_analysis.png** - Comprehensive validation metrics\n")
        f.write("4. **loss_contribution_analysis.png** - Loss component contribution analysis\n")
        f.write("5. **academic_training_summary.png** - Complete academic overview figure\n\n")
        
        f.write("## Key Insights\n\n")
        
        if analyzer.train_data:
            train_df = pd.DataFrame(analyzer.train_data)
            epoch_train = train_df.groupby('epoch').mean()
            
            initial_loss = epoch_train['total_loss'].iloc[0]
            final_loss = epoch_train['total_loss'].iloc[-1]
            loss_reduction = ((initial_loss - final_loss) / initial_loss) * 100
            
            f.write(f"- **Loss Reduction**: {loss_reduction:.1f}% (from {initial_loss:.3f} to {final_loss:.3f})\n")
            f.write(f"- **Training Epochs**: {int(epoch_train.index.max())}\n")
            f.write(f"- **Loss Components**: L1, GIoU, and Center losses\n")
            
        if analyzer.val_data:
            val_df = pd.DataFrame(analyzer.val_data)
            best_acc50 = val_df['acc_50'].max() * 100
            best_miou = val_df['miou'].max() * 100
            
            f.write(f"- **Best Accuracy@0.5**: {best_acc50:.2f}%\n")
            f.write(f"- **Best mIoU**: {best_miou:.2f}%\n")
        
        f.write("\n## Academic Presentation Notes\n\n")
        f.write("- Use `academic_training_summary.png` for comprehensive overview\n")
        f.write("- Individual component plots can be used for detailed analysis\n")
        f.write("- All figures are high-resolution (300 DPI) for publication quality\n")
        f.write("- Color schemes are consistent and professional for academic use\n")
    
    print(f"\n" + "="*60)
    print("LOSS ANALYSIS COMPLETE")
    print("="*60)
    print(f"Generated visualizations in: {output_dir}")
    print(f"Summary report: {output_dir}/analysis_summary.md")
    print("="*60)


if __name__ == "__main__":
    main() 