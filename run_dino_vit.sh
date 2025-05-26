#!/bin/bash

# Set environment variables
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:32,garbage_collection_threshold:0.6
export CUDA_LAUNCH_BLOCKING=0
export PYTHONPATH=$PYTHONPATH:/home/pokle/Trans-VG:/home/pokle/Trans-VG/visual_grounding

# Timestamp for output directories and logs
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_NAME="transvg_dino_vit_${TIMESTAMP}"
LOG_DIR="logs"
CHECKPOINT_DIR="checkpoints/dino_vit"

# Create necessary directories
mkdir -p $LOG_DIR
mkdir -p $CHECKPOINT_DIR

# Set device - adjust as needed
DEVICE="cuda:0"

# Lower batch size due to larger image dimensions (384x384)
BATCH_SIZE=16

# First, run a quick model check
echo "Initializing model to check setup..."
cd /home/pokle/Trans-VG/visual_grounding
python scripts/train_dino_vit.py --batch_size $BATCH_SIZE --partial_freeze

# Now run the actual training with the main training script
echo "Starting training with DINO ViT backbone..."

python train.py \
  --log_name $LOG_NAME \
  --device $DEVICE \
  --partial_freeze_vision \
  --batch_size $BATCH_SIZE \
  --lr 5e-5 \
  --epochs 200 \
  --resume /home/pokle/Trans-VG/visual_grounding/checkpoints/transvg_dino_vit_20250520_174900_epoch6.pth \
  --dino_vit

echo "Training completed. Log saved to $LOG_DIR/$LOG_NAME.log" 