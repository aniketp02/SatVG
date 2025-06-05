# RSVG Dataset Analysis for TransVG

## Dataset Structure

The RSVG (Remote Sensing Visual Grounding) dataset has the following structure:

```
rsvg_dataset/
├── images/                  # Contains satellite/aerial imagery
├── processed/               # Pre-processed dataset files
│   ├── rsvg_train.pth       # Training set
│   ├── rsvg_val.pth         # Validation set  
│   └── rsvg_test.pth        # Test set
├── rsvg/                    # Raw dataset files
│   ├── rsvg_train.pth
│   ├── rsvg_val.pth
│   └── rsvg_test.pth
└── rsvg_train.pth           # Duplicate or alternative format
```

## Dataset Analysis

### Data Format

The dataset is stored in PyTorch (.pth) format, which contains:
- Images: Remote sensing imagery (satellite/aerial photos)
- Text queries: Natural language descriptions of regions within images
- Bounding box annotations: Target regions corresponding to text queries

### Potential Issues

Based on previous model performance analysis, the following issues may exist in the dataset:

1. **Identical Bounding Boxes**: 
   - There's evidence suggesting many samples contain identical target bounding boxes
   - This could indicate data corruption or preprocessing errors
   - Example observed: `[207.0, 198.0, 287.0, 283.0]` appearing across multiple samples

2. **Image-Text Alignment Issues**:
   - The model consistently achieves only ~33% accuracy at IoU threshold 0.25
   - This suggests a potential misalignment between textual descriptions and visual regions

3. **Bias in Annotations**:
   - The model may be learning to predict a specific region regardless of query
   - This could be due to annotation bias or limited variety in the dataset

4. **Class Imbalance**:
   - Filename patterns suggest some categories may be overrepresented:
     - `sport_baseball`, `sport_tennis`, `man_made_storage_tank`, `junction_roundabout`
   - This imbalance could bias the model toward certain region types

## Recommended Dataset Validation Steps

1. **Data Diversity Check**:
   - Analyze distribution of bounding box coordinates
   - Verify uniqueness of annotations across different queries
   - Examine class distribution across train/val/test splits

2. **Annotation Quality Assessment**:
   - Manually inspect random samples to verify text-region alignment
   - Check for annotation errors or inconsistencies
   - Verify that different queries for the same image have different ground truth boxes

3. **Preprocessing Validation**:
   - Ensure data augmentation maintains ground truth validity
   - Verify that preprocessing doesn't introduce systematic errors
   - Check that normalization procedures are appropriate for remote sensing imagery

## Integration with TransVG

Before continuing with model improvements like focal loss and center prediction loss, the following steps are recommended:

1. **Dataset Correction**:
   - Verify and correct any duplicate or erroneous annotations
   - Ensure proper alignment between text queries and target regions

2. **Data Augmentation Strategy**:
   - Implement mild augmentations that preserve spatial relationships
   - Consider specialized augmentations for remote sensing data

3. **Model Debugging**:
   - Add visualization tools to inspect predictions across different queries
   - Implement metrics to detect prediction bias
   - Monitor diversity of model outputs during training

By addressing these dataset issues, the TransVG model's performance ceiling could potentially increase beyond the current ~33% accuracy limitation. 