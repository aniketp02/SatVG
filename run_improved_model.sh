#!/bin/bash

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=$PYTHONPATH:/home/pokle/Trans-VG

# Timestamp for output directories and logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_NAME="transvg_improved_${TIMESTAMP}"
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Set device
DEVICE="cuda:1"

# Run the training script with improved model settings:
# 1. Partially frozen backbones
# 2. Better initialization
# 3. Gradient clipping
# 4. Higher learning rates
# 5. Reduced cross-attention layers (4 instead of 8)
python train.py \
  --log_name $LOG_NAME \
  --device $DEVICE \
  --freeze_both \
  --partial_freeze_vision \
  --partial_freeze_linguistic \
  --batch_size 32 \
  --lr 1e-3 \
  --epochs 5

echo "Training completed. Log saved to $LOG_DIR/$LOG_NAME.log" 