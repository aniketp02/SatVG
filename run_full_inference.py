#!/usr/bin/env python3
"""
Script to run comprehensive inference on the trained TransVG model
with more samples for thorough academic evaluation.
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run comprehensive inference with more samples"""
    
    # Checkpoint path provided by user
    checkpoint_path = "/home/pokle/Trans-VG/visual_grounding/checkpoints/all_data_dino_improved_epoch54.pth"
    
    # Check if checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"Error: Checkpoint not found at {checkpoint_path}")
        print("Please check the path and try again.")
        return 1
    
    # Parameters for comprehensive evaluation
    data_root = "/home/pokle/Trans-VG/visual_grounding/dior-rsvg"
    output_dir = "comprehensive_inference_results_epoch54"
    split = "test"  # Can be changed to "val"
    num_samples = 200  # More samples for better statistics
    
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
    
    print("Running comprehensive inference with the following parameters:")
    print(f"  Checkpoint: {checkpoint_path}")
    print(f"  Data root: {data_root}")
    print(f"  Output dir: {output_dir}")
    print(f"  Split: {split}")
    print(f"  Num samples: {num_samples}")
    print(f"  Command: {' '.join(cmd)}")
    print()
    print("This will take longer but provide more comprehensive results...")
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