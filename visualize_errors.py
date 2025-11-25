#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import matplotlib.pyplot as plt
import tifffile


# Path to your predictions file
predictions_folder = "outputs/OrganoidsINRIA/swinunetr/01/testing/errors_logs/"
predictions_file = predictions_folder + "testing_errors.txt"  # Change this to your file path

CLASSES = {
    0: "Chouxfleurs",
    1: "Compact",
    2: "Cystiques",
}

def parse_predictions_file(filepath):
    """
    Parse file with format:
    Epoch 0:
    /path/to/file.tif	Pred: 2	Target: 0
    ...
    
    Returns list of dicts: [{'path': ..., 'pred': ..., 'target': ...}, ...]
    """
    samples = []
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            
            # Skip empty lines or epoch headers
            if not line or line.startswith('Epoch') or line.startswith('*'):
                continue
            
            # Parse: path\tPred: X\tTarget: Y
            parts = line.split('\t')
            if len(parts) != 3:
                print(f"[WARN] Skipping malformed line: {line}", file=sys.stderr)
                continue
            
            path = parts[0].strip()
            pred = parts[1].split(':')[1].strip()
            pred = int(pred)
            pred = CLASSES.get(pred, "unknown")
            target = parts[2].split(':')[1].strip()
            target = int(target)
            target = CLASSES.get(target, "unknown")
            
            samples.append({
                'path': path,
                'pred': pred,
                'target': target
            })
    
    return samples

def visualize_sample(sample_info):
    """
    Load 3D volume and display XY and YZ slices with Pred/Target in title.
    """
    path = sample_info['path']
    pred = sample_info['pred']
    target = sample_info['target']
    
    if not os.path.exists(path):
        print(f"[ERROR] File not found: {path}", file=sys.stderr)
        return
    
    try:
        # Load 3D volume (expected shape: D, H, W or C, D, H, W)
        vol = tifffile.imread(path)
        
        # Handle different shapes
        if vol.ndim == 3:
            # Shape: (D, H, W)
            mid_depth = vol.shape[0] // 2
            mid_height = vol.shape[2] // 2
            img_xy = vol[mid_depth, :, :]
            img_yz = vol[:, mid_height, :]
        elif vol.ndim == 4:
            # Shape: (C, D, H, W) - take first channel
            mid_depth = vol.shape[1] // 2
            mid_height = vol.shape[2] // 2
            img_xy = vol[0, mid_depth, :, :]
            img_yz = vol[0, :, mid_height, :]
        else:
            print(f"[ERROR] Unexpected volume shape: {vol.shape}", file=sys.stderr)
            return
        
        # Create figure with two subplots
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # XY slice
        axes[0].imshow(img_xy, cmap='gray')
        axes[0].set_title(f'XY - X_size: {img_xy.shape[1]}, Y_size: {img_xy.shape[0]}')
        axes[0].axis('on')
        
        # YZ slice
        axes[1].imshow(img_yz, cmap='gray')
        axes[1].set_title(f'YZ - Y_size: {img_yz.shape[1]}, Z_size: {img_yz.shape[0]}')
        axes[1].axis('on')
        
        # Add filename at the top
        filename = os.path.basename(path)
        fig.suptitle(filename, fontsize=10)
        fig.suptitle(f'Pred: {pred}, Target: {target}', fontsize=12)
        
        plt.tight_layout()
        plt.show()
        folder = predictions_folder + "img/"
        if not os.path.exists(folder):
            os.makedirs(folder)
        plt.savefig(f"{folder}{filename}.png")
        # Close figure to free memory
        plt.close(fig)
        
    except Exception as e:
        print(f"[ERROR] Failed to load/display {path}: {e}", file=sys.stderr)

def main():
    
    if not os.path.exists(predictions_file):
        print(f"[ERROR] Predictions file not found: {predictions_file}", file=sys.stderr)
        sys.exit(1)
    
    # Parse all samples
    samples = parse_predictions_file(predictions_file)
    print(f"[INFO] Loaded {len(samples)} samples from {predictions_file}")
    
    if not samples:
        print("[WARN] No samples found in file.")
        return
    
    # Visualize each sample one at a time
    for i, sample in enumerate(samples):
        print(f"\n[{i+1}/{len(samples)}] Processing: {sample['path']}")
        print(f"  Pred: {sample['pred']}, Target: {sample['target']}")
        
        visualize_sample(sample)
        
        # Optional: wait for user input to continue
        # if i < len(samples) - 1:  # Don't prompt after last sample
        #     choice = input("Press Enter to continue, 'q' to quit: ").strip().lower()
        #     if choice == 'q':
        #         print("[INFO] Visualization stopped by user.")
        #         break
    
    print(f"\n[INFO] Finished visualizing {i+1} samples.")

if __name__ == "__main__":
    main()
