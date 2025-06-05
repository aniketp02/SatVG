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

### Experiment 3: DINO ViT Backbone with Improved Training

**Date**: 2025-05-20 to 2025-05-28
**Status**: Completed (28 epochs)

**Description**: Replaced the ResNet50 visual backbone with a Vision Transformer (ViT) DINO model and implemented several training improvements.

**Implementation Details**:
1. Integrated DINO ViT as visual backbone
2. Partially frozen DINO backbone and linguistic backbone
3. Increased image size to 384×384
4. Increased cross-attention layers to 6
5. Added center loss and focal loss components
6. Implemented data augmentation (scale, translate, color jitter)
7. Batch size: 24 (adjusted from 16 due to memory constraints)
8. Learning rate: 5e-5, BERT learning rate: 2e-5

**Observations**:
- Initial performance improved rapidly (epochs 0-5)
- Performance peaked around epochs 7-9
- Training loss continued to decrease (from ~7.6 to ~3.85) over 28 epochs
- Validation metrics plateaued after epoch 9
- Component losses showed significant improvements:
  - L1 loss: ~0.96 → ~0.38-0.42
  - GIoU loss: ~1.40 → ~0.95-1.03
  - Center loss: ~0.059 → ~0.012-0.018

**Investigation Findings**:
- Initial concerns about identical target boxes were investigated
- Dataset inspection confirmed diverse boxes (32,638 total boxes, 31,318 unique)
- The identical target box issue in logs was due to always displaying the same validation sample
- Random seed configuration needed improvement for reproducibility

**Best Performance (Epoch 8)**:
- Acc@0.25: 0.2541
- Acc@0.5: 0.0959 
- Acc@0.75: 0.0143
- mIoU: 0.1500
- medianIoU: 0.0337
- Center error: ~100 pixels

**Final Performance (Epoch 27)**:
- Acc@0.25: 0.2473
- Acc@0.5: 0.0916
- Acc@0.75: 0.0127
- mIoU: 0.1449
- medianIoU: 0.0204
- Center error: ~105-110 pixels

**Key Learnings**:
- DINO ViT backbone shows competitive performance to ResNet50
- Model learns effectively but reaches a performance ceiling
- Width/height error decreases faster than center error, suggesting the model learns box sizing better than localization
- Signs of potential overfitting after extended training (training loss continues to decrease while validation metrics plateau)

## Ongoing Experiments

### Experiment 4: Improved Logging and Seed Configuration

**Start Date**: 2025-05-28
**Current Status**: In progress

**Description**: Implementing fixes for logging and random seed issues identified in previous experiments.

**Implementation Details**:
1. Fixed validation logging to show diverse examples instead of always the first sample
2. Implemented proper random seed configuration for reproducibility
3. Added more comprehensive logging of diverse samples
4. Enabled shuffling in validation dataloader for more robust evaluation

**Expected Outcomes**:
- Better assessment of model performance through diverse sample logging
- More consistent results between training runs
- More robust evaluation through validation set shuffling

## Planned Experiments

### Experiment 5: Learning Rate Schedule Optimization

**Description**: Implement learning rate warmup and cosine decay instead of step decay.

**Hypothesis**: Gradual warmup helps stabilize early training, while cosine decay provides smoother learning rate reduction.

**Implementation Plan**:
1. Implement linear warmup for first 5% of training
2. Switch to cosine decay for remainder of training
3. Test different combinations of min/max learning rates

**Expected Outcome**: More stable training and potentially better convergence.

### Experiment 6: Multi-Scale Feature Fusion

**Description**: Enhance the model's feature representation by integrating multi-scale features from the vision backbone.

**Hypothesis**: Multi-scale features will improve localization performance, especially for objects of varying sizes.

**Implementation Plan**:
1. Extract features from different layers of the DINO ViT backbone
2. Implement feature pyramid or similar architecture for fusion
3. Modify cross-attention to work with multi-scale features

**Expected Outcome**: Improved localization accuracy and better performance on small objects.

### Experiment 7: Loss Function Enhancements

**Description**: Optimize the weighting and composition of loss functions.

**Hypothesis**: Better loss function design will guide the model to focus more on reducing center error.

**Implementation Plan**:
1. Adjust weights between L1, GIoU, and center losses
2. Implement dynamic loss weighting based on training progress
3. Experiment with DIoU or CIoU loss variants

**Expected Outcome**: Reduced center error and improved overall accuracy.

## Improvement Ideas

1. **Architectural Improvements**:
   - Add multi-scale feature fusion from vision backbone
   - Experiment with different normalization strategies
   - Try alternative cross-attention mechanisms

2. **Training Strategies**:
   - Progressive unfreezing of backbone layers
   - Mixed precision training for faster iterations
   - Gradient accumulation for effectively larger batch sizes
   - Implement curriculum learning by difficulty

3. **Loss Function Enhancements**:
   - Dynamically adjust center-ness constraint weight during training
   - Progressive loss weighting that changes during training
   - Add auxiliary losses to encourage better feature representation

4. **Model Interpretability**:
   - Add attention visualization for cross-modal layers
   - Track attention maps during training
   - Analyze errors by characteristics (object size, text complexity)

5. **Evaluation Metrics**:
   - Compare with human annotations to assess qualitative performance
   - Add metrics for specific object categories or query types
   - Implement more frequent validation checks

6. **Inference Optimization**:
   - Model pruning and quantization
   - Knowledge distillation to smaller models
   - Benchmark latency on different hardware 