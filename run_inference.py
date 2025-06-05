#!/usr/bin/env python3
"""
Simple wrapper script to run inference on the trained TransVG model
with the specific checkpoint path provided by the user.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run inference with the specified checkpoint"""
    
    # Checkpoint path provided by user
    checkpoint_path = "/home/pokle/Trans-VG/visual_grounding/checkpoints/all_data_dino_improved_epoch54.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please check the path and try again.")
        return 1
    
    # Default parameters
    data_root = "/home/pokle/Trans-VG/visual_grounding/dior-rsvg"
    output_dir = "inference_results_epoch54"
    split = "test"  # Can be changed to "val"
    num_samples = 50  # Set to 0 for all samples
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Build command
    cmd = [
        sys.executable, "inference_annotated.py",
        "--checkpoint_path", checkpoint_path,
        "--data_root", data_root,
        "--output_dir", output_dir,
        "--split", split,
        "--num_samples", str(num_samples),
        "--save_metrics",
        "--create_summary", 
        "--create_gallery"
    ]
    
    print("Running inference with the following parameters:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data root: {data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"  Split: {split}")
    print(f"  Num samples: {num_samples}")
    print(f"  Command: {' '.join(cmd)}")
    print()
    
    # Run the inference script
    try:
        result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))
        return result.returncode
    except KeyboardInterrupt:
        print("\nInference interrupted by user")
        return 1
    except Exception as e:
        print(f"Error running inference: {e}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 