# TransVG Model Inference Guide

This guide explains how to generate annotated inferences from the trained TransVG model with DINO ViT backbone for academic presentation.

## Model Information

- **Architecture**: TransVG with DINO ViT backbone
- **Checkpoint**: `all_data_dino_improved_epoch54.pth` (Epoch 54)
- **Dataset**: DIOR-RSVG (Remote Sensing Visual Grounding)
- **Image Size**: 384x384 pixels

## Quick Start

### Option 1: Quick Inference (50 samples)
```bash
cd visual_grounding
python run_inference.py
```

### Option 2: Comprehensive Inference (200 samples)
```bash
cd visual_grounding
python run_full_inference.py
```

### Option 3: Custom Inference
```bash
cd visual_grounding
python inference_annotated.py \
    --checkpoint_path /home/pokle/Trans-VG/visual_grounding/checkpoints/all_data_dino_improved_epoch54.pth \
    --data_root /home/pokle/Trans-VG/visual_grounding/dior-rsvg \
    --output_dir custom_results \
    --split test \
    --num_samples 100 \
    --save_metrics \
    --create_summary \
    --create_gallery
```

## Generated Outputs

The inference script generates the following files for academic presentation:

### 1. Annotated Sample Visualizations
- **Files**: `sample_XXX_iou_Y.YYY.png`
- **Content**: Side-by-side comparison showing:
  - Model input (resized 384x384)
  - Original image with predictions
  - Ground truth bounding box (green)
  - Predicted bounding box (red)
  - Text query and IoU score
  - Model and checkpoint information

### 2. Comparison Gallery
- **File**: `comparison_gallery.png`
- **Content**: Grid showing best and worst performing samples
- **Purpose**: Quick overview of model strengths and weaknesses

### 3. Performance Histogram
- **File**: `performance_histogram.png`
- **Content**: Distribution of IoU scores with statistics
- **Purpose**: Understanding overall performance distribution

### 4. Summary Report
- **Files**: `inference_summary.txt` and `inference_summary.json`
- **Content**: 
  - Model configuration
  - Performance metrics (Acc@0.25, Acc@0.5, Acc@0.75)
  - IoU statistics (mean, median, std, min, max)
  - Performance breakdown by IoU ranges

### 5. Detailed Metrics
- **File**: `detailed_metrics.json`
- **Content**: Per-sample results with image names, text queries, IoU scores, and bounding boxes

## Performance Metrics Explanation

### IoU (Intersection over Union)
- **Range**: 0.0 to 1.0
- **Interpretation**: 
  - 0.75+: Excellent localization
  - 0.5-0.75: Good localization
  - 0.25-0.5: Moderate localization
  - <0.25: Poor localization

### Accuracy Thresholds
- **Acc@0.25**: Percentage of predictions with IoU ≥ 0.25
- **Acc@0.5**: Percentage of predictions with IoU ≥ 0.5
- **Acc@0.75**: Percentage of predictions with IoU ≥ 0.75

## Sample Results (50 test samples)

```
Model: TransVG with DINO ViT backbone
Checkpoint: all_data_dino_improved_epoch54.pth (Epoch 54)
Dataset: test split (50 samples)
Average IoU: 0.2331 ± 0.2521
Accuracy@0.5: 18.0%
Accuracy@0.75: 4.0%

Performance Breakdown:
  Excellent (IoU ≥ 0.75): 2 (4.0%)
  Good (0.5 ≤ IoU < 0.75): 7 (14.0%)
  Moderate (0.25 ≤ IoU < 0.5): 10 (20.0%)
  Poor (IoU < 0.25): 31 (62.0%)
```

## Command Line Arguments

### Required Arguments
- `--checkpoint_path`: Path to the trained model checkpoint
- `--data_root`: Path to the DIOR-RSVG dataset root directory

### Optional Arguments
- `--output_dir`: Output directory for results (default: "inference_results")
- `--split`: Dataset split to evaluate ("test" or "val", default: "test")
- `--num_samples`: Number of samples to process (0 for all, default: 50)
- `--device`: Device to use ("cuda", "cpu", default: "cuda")
- `--save_metrics`: Save detailed metrics to JSON
- `--create_summary`: Create summary report
- `--create_gallery`: Create comparison gallery

## Academic Presentation Tips

1. **Use the comparison gallery** to show model capabilities and limitations
2. **Include performance histogram** to demonstrate score distribution
3. **Show individual annotated samples** for detailed analysis
4. **Reference the summary statistics** for quantitative evaluation
5. **Compare with baseline methods** using the same evaluation metrics

## Troubleshooting

### Common Issues
1. **CUDA out of memory**: Reduce batch size or use CPU
2. **Dataset not found**: Check data_root path
3. **Checkpoint not found**: Verify checkpoint path
4. **Import errors**: Ensure all dependencies are installed

### Dependencies
- PyTorch
- torchvision
- matplotlib
- PIL (Pillow)
- numpy
- json

## File Structure
```
visual_grounding/
├── inference_annotated.py          # Main inference script
├── run_inference.py               # Quick inference wrapper
├── run_full_inference.py          # Comprehensive inference wrapper
├── create_comparison_gallery.py   # Gallery creation utilities
├── README_INFERENCE.md            # This guide
└── inference_results_epoch54/     # Generated results
    ├── sample_001_iou_0.642.png   # Individual samples
    ├── comparison_gallery.png     # Best/worst comparison
    ├── performance_histogram.png  # IoU distribution
    ├── inference_summary.txt      # Human-readable summary
    ├── inference_summary.json     # Machine-readable summary
    └── detailed_metrics.json      # Per-sample metrics
``` 