#!/bin/bash

# Set environment variables
export PYTHONPATH=$PYTHONPATH:/home/pokle/Trans-VG/visual_grounding

# Run the test script with a small batch size
python scripts/train_dino_vit.py --batch_size 2 --partial_freeze

echo "Test complete" 