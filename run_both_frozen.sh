#!/bin/bash

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=$PYTHONPATH:/home/pokle/Trans-VG

# Timestamp for output directories and logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_NAME="transvg_both_frozen_xyxy_${TIMESTAMP}"
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Set device
DEVICE="cuda:1"

# Run the training script with both visual and linguistic backbones frozen
# Now using xyxy format directly instead of cx,cy,w,h format
python train.py \
  --log_name $LOG_NAME \
  --device $DEVICE \
  --freeze_both \
  --batch_size 32 \
  --lr 5e-4 \
  --epochs 1

echo "Training completed. Log saved to $LOG_DIR/$LOG_NAME.log" 