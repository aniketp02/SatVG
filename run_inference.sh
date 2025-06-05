#!/bin/bash

# Inference script for TransVG model with DINO ViT backbone
# Generates annotated visualizations for academic presentation

echo "Starting TransVG Model Inference..."
echo "=================================="

# Set parameters
CHECKPOINT_PATH="/home/pokle/Trans-VG/visual_grounding/checkpoints/all_data_dino_improved_epoch54.pth"
DATA_ROOT="/home/pokle/Trans-VG/visual_grounding/dior-rsvg"
OUTPUT_DIR="inference_results_$(date +%Y%m%d_%H%M%S)"
SPLIT="test"  # Can be changed to "val" if needed
NUM_SAMPLES=100  # Number of samples to visualize (0 for all)

# Check if checkpoint exists
if [ ! -f "$CHECKPOINT_PATH" ]; then
    echo "Error: Checkpoint not found at $CHECKPOINT_PATH"
    exit 1
fi

# Check if data directory exists
if [ ! -d "$DATA_ROOT" ]; then
    echo "Error: Data directory not found at $DATA_ROOT"
    exit 1
fi

echo "Checkpoint: $CHECKPOINT_PATH"
echo "Data Root: $DATA_ROOT"
echo "Output Directory: $OUTPUT_DIR"
echo "Split: $SPLIT"
echo "Number of samples: $NUM_SAMPLES"
echo ""

# Run inference with comprehensive options
python inference_annotated.py \
    --checkpoint_path "$CHECKPOINT_PATH" \
    --data_root "$DATA_ROOT" \
    --output_dir "$OUTPUT_DIR" \
    --split "$SPLIT" \
    --num_samples $NUM_SAMPLES \
    --device cuda \
    --save_metrics \
    --create_summary

# Check if inference was successful
if [ $? -eq 0 ]; then
    echo ""
    echo "Inference completed successfully!"
    echo "Results saved to: $OUTPUT_DIR"
    echo ""
    echo "Generated files:"
    echo "- Annotated visualizations: sample_*.png"
    echo "- Summary report: inference_summary.txt"
    echo "- Detailed metrics: detailed_metrics.json"
    echo "- JSON summary: inference_summary.json"
    echo ""
    echo "You can now review the results and present them to your professor."
else
    echo "Error: Inference failed!"
    exit 1
fi 