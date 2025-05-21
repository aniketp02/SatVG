# TransVG Experiments Log

This document tracks the experiments conducted with the TransVG model for visual grounding, along with results and observations.

## Table of Contents

- [Baseline Model](#baseline-model)
- [Completed Experiments](#completed-experiments)
- [Ongoing Experiments](#ongoing-experiments)
- [Planned Experiments](#planned-experiments)
- [Improvement Ideas](#improvement-ideas)

## Baseline Model

Our baseline model uses a cross-modal transformer architecture with a ResNet50 visual backbone and BERT language backbone. The key configurations are:

- Image size: 224×224
- Hidden dimension: 256
- Cross-attention layers: 4
- Visual encoder layers: 6
- BERT model: bert-base-uncased
- Prediction head: MLP with 1 hidden layer (256 dim)
- Backbone: ResNet50 (partially frozen - early layers only)
- BERT: Partially frozen (last 4 layers trainable)
- Optimizer: AdamW
- Learning rate: 1e-3 (vision), 2e-5 (language)
- Batch size: 32
- Loss functions: L1 loss + GIoU loss
- Input resolution: 224×224
- Coordinate format: Normalized [0,1] in [xmin, ymin, xmax, ymax] format

### Performance

Performance on test set after 5 epochs:

| Metric       | Value  | Description |
|--------------|--------|-------------|
| Acc@0.25     | 0.3399 | Percentage of predictions with IoU > 0.25 |
| Acc@0.5      | 0.1241 | Percentage of predictions with IoU > 0.5 |
| Acc@0.75     | 0.0179 | Percentage of predictions with IoU > 0.75 |
| mIoU         | 0.1947 | Mean Intersection over Union |
| medianIoU    | 0.1049 | Median Intersection over Union |
| loss         | 4.3934 | Combined L1 + GIoU loss |

## Completed Experiments

### Experiment 1: Initial TransVG Implementation

**Date**: 2025-05-18

**Description**: Initial implementation of TransVG with both visual and linguistic backbones completely frozen.

**Observations**:
- Model failed to learn, consistently predicting the same bounding box ([224, 224, 448, 448]) for every input
- Validation metrics remained constant across epochs (Acc@0.5: 0.0066, mIoU: 0.0714)
- Loss showed minimal decrease

**Root Cause Analysis**:
- Coordinate system mismatch between predictions and targets
- Both backbones being frozen limited model capacity
- Bounding box prediction always outside normalized range

**Results**:
- Acc@0.25: 0.1033
- Acc@0.5: 0.0066
- mIoU: 0.0714

### Experiment 2: Fixed Coordinate System

**Date**: 2025-05-18

**Description**: Implemented a custom dataloader with proper coordinate normalization and fixed bounding box prediction.

**Changes**:
1. Created custom dataloader that properly normalizes bounding box coordinates
2. Fixed vision encoder to use proper coordinate scaling
3. Modified prediction head initialization to predict reasonable default boxes
4. Constrained predictions to ensure valid bounding boxes
5. Modified validation logic to handle coordinate scaling correctly
6. Partially unfroze vision backbone (kept early layers frozen)

**Observations**:
- Model successfully learned to predict diverse bounding boxes
- Loss steadily decreased from ~7.9 to ~4.3
- Performance improved across all metrics

**Results**:
- Acc@0.25: 0.3399 (+0.2366)
- Acc@0.5: 0.1241 (+0.1175)
- Acc@0.75: 0.0179 (+0.0176)
- mIoU: 0.1947 (+0.1233)
- medianIoU: 0.1049

## Ongoing Experiments

### Experiment 3: ViT DINO Backbone

**Start Date**: 2025-05-20
**Current Status**: In progress (Epoch 7/200)

**Description**: Replaced the ResNet50 visual backbone with a Vision Transformer (ViT) DINO model.

**Hypothesis**: DINO pre-training provides better contextualized visual features that should improve grounding performance.

**Implementation Details**:
1. Integrated DINO ViT as visual backbone
2. Partially frozen DINO backbone
3. Partially frozen linguistic backbone
4. Hidden dimension: 256
5. Batch size: 16 (reduced from 32 due to memory constraints)
6. Learning rate: 5e-5, BERT learning rate: 2e-5

**Current Best Performance (Epoch 5/200)**:
- Acc@0.25: 0.3288
- Acc@0.5: 0.1193
- Acc@0.75: 0.0173
- mIoU: 0.1893
- medianIoU: 0.0944
- Loss: 4.3884

**Observations**:
- Model shows consistent improvement from epochs 1-5
- Slight performance dip in epoch 6
- Training loss shows expected fluctuations but overall decreasing trend
- Performance approaching baseline model with potential for further gains

**Next Steps**:
- Continue training to at least 15-20 epochs
- Monitor validation metrics for stabilization
- Consider learning rate adjustments if performance plateaus

### Experiment 4: Increased Image Size and Cross-Attention Layers

**Start Date**: 2025-05-20
**Current Status**: In progress

**Description**: Increased input image resolution and number of cross-attention layers to improve feature representation and cross-modal interaction.

**Implementation Details**:
- Image size: 384×384 (increased from 224×224)
- Cross-attention layers: 6 (increased from 4)
- Fixed coordinate system following Experiment 2

**Current Performance**:
- Acc@0.25: 0.3150
- Acc@0.5: 0.1116
- Acc@0.75: 0.0130
- mIoU: 0.1798
- medianIoU: 0.0853
- Loss: 4.5501

**Observations**:
- Higher resolution allows for more detailed feature extraction
- Model showing promising performance comparable to baseline
- Increased cross-attention layers provide more effective cross-modal fusion
- Larger input size increases memory requirements

**Next Steps**:
- Continue training to assess full potential
- Compare performance trajectory against the DINO backbone experiment
- Evaluate if the increased complexity justifies the performance gain

## Planned Experiments

### Experiment 5: Data Augmentation

**Description**: Implement data augmentation strategies to improve model generalization.

**Hypothesis**: Augmentation will improve model generalization and prevent overfitting, especially for longer training runs.

**Implementation Plan**:
1. Add color jitter, random flips, and random crops
2. Ensure bounding box coordinates are properly adjusted for spatial augmentations
3. Compare performance with and without augmentation

**Expected Outcome**: Slower initial training but better final performance, especially on test set.

### Experiment 6: Learning Rate Schedule Optimization

**Description**: Implement learning rate warmup and cosine decay instead of step decay.

**Hypothesis**: Gradual warmup helps stabilize early training, while cosine decay provides smoother learning rate reduction.

**Implementation Plan**:
1. Implement linear warmup for first 5% of training
2. Switch to cosine decay for remainder of training
3. Test different combinations of min/max learning rates

**Expected Outcome**: More stable training and potentially better convergence.

## Improvement Ideas

1. **Architectural Improvements**:
   - Add multi-scale feature fusion from vision backbone
   - Experiment with different normalization strategies

2. **Training Strategies**:
   - Progressive unfreezing of backbone layers
   - Mixed precision training for faster iterations
   - Gradient accumulation for effectively larger batch sizes

3. **Loss Function Enhancements**:
   - Center-ness constraint to focus on center accuracy
   - Dynamic weighting of L1 and GIoU losses
   - Classification head to predict confidence score

4. **Model Interpretability**:
   - Add attention visualization for cross-modal layers
   - Track attention maps during training
   - Analyze errors by characteristics (object size, text complexity)

5. **Evaluation Metrics**:
   - Compare with human annotations to assess qualitative performance
   - Add metrics for specific object categories or query types

6. **Inference Optimization**:
   - Model pruning and quantization
   - Knowledge distillation to smaller models
   - Benchmark latency on different hardware 