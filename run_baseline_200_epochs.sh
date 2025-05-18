#!/bin/bash

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=$PYTHONPATH:/home/pokle/Trans-VG

# Timestamp for output directories and logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_NAME="transvg_baseline_200ep_${TIMESTAMP}"
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Set device
DEVICE="cuda:1"

# Run the baseline model for 200 epochs:
# 1. Using our custom dataloader with proper coordinate normalization
# 2. Better bounding box prediction initialization
# 3. Proper scaling between normalized and pixel coordinates
# 4. Partially frozen vision backbone
echo "Starting TransVG baseline training for 200 epochs..."
echo "Logs will be saved to $LOG_DIR/$LOG_NAME.log"

python visual_grounding/train.py \
  --log_name $LOG_NAME \
  --device $DEVICE \
  --partial_freeze_vision \
  --batch_size 32 \
  --lr 1e-3 \
  --epochs 200

echo "Training completed. Log saved to $LOG_DIR/$LOG_NAME.log"
echo "Model checkpoints saved to $CHECKPOINT_DIR/${LOG_NAME}_epoch*.pth" 