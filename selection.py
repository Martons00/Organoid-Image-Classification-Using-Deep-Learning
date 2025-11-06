#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataset import OrganoidsINRIA3D

import os
import sys
import signal
import matplotlib.pyplot as plt


PROBLEMATIC_FILE = "problematic_samples.txt"   # file for problematic samples
GOOD_FILE        = "good_samples.txt"          # file for already reviewed good samples


def load_id_set(path: str) -> set:
    """
    Load IDs/paths from a text file (one per line).
    Return an empty set if the file does not exist.
    """
    if not os.path.exists(path):
        return set()
    try:
        with open(path, "r", encoding="utf-8") as f:
            return set(line.strip() for line in f if line.strip())
    except Exception as e:
        print(f"[WARN] Unable to read {path}: {e}", file=sys.stderr)
        return set()


def append_id(path: str, sample_id: str) -> None:
    """
    Append a new ID to the given file and force a disk flush for immediate persistence.
    """
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.exists(parent):
        os.makedirs(parent, exist_ok=True)

    try:
        with open(path, "a", encoding="utf-8") as f:
            # ensure each entry is on its own line
            f.write(sample_id + "\n")
            f.flush()              # flush Python buffers
            os.fsync(f.fileno())   # sync to disk
    except Exception as e:
        print(f"[ERROR] Writing to {path} failed: {e}", file=sys.stderr)


def get_sample_id(sample: dict, index: int) -> str:
    """
    Return a stable identifier, preferring 'path' if available.
    """
    return sample.get("path", f"index_{index}")


def safe_to_numpy(t):
    """
    Convert torch tensors or compatible arrays to numpy.
    """
    try:
        return t.detach().cpu().numpy()
    except Exception:
        try:
            import numpy as np
            return np.asarray(t)
        except Exception:
            raise


def main():
    # TRAINING dataset with augmentation
    train_ds = OrganoidsINRIA3D(
        root="'/Volumes/LaCie/Organoids/Organoids_Dataset'",
        exact_class_dir=False,
    )

    # Load persistent state
    problematic_set = load_id_set(PROBLEMATIC_FILE)
    good_set = load_id_set(GOOD_FILE)
    print(f"[INFO] Loaded {len(problematic_set)} problematic and {len(good_set)} good IDs.")    

    # Handle CTRL+C cleanly
    interrupted = {"flag": False}
    def _sigint_handler(sig, frame):
        interrupted["flag"] = True
        print("\n[INFO] Interrupted by user (SIGINT).")
    signal.signal(signal.SIGINT, _sigint_handler)

    newly_problematic = []
    newly_good = []

    try:
        for i in range(len(train_ds)):
            if interrupted["flag"]:
                break

            # Load the sample; if your dataset exposes an index->id method, you could fetch ID first
            sample = train_ds[i]
            sample_id = get_sample_id(sample, i)

            # Skip if already processed (either problematic or good)
            if sample_id in problematic_set:
                print(f"[SKIP] Already marked problematic: {sample_id}")
                continue
            if sample_id in good_set:
                print(f"[SKIP] Already marked good: {sample_id}")
                continue

            vol = sample["vol"]  # expected shape: [C, D, H, W]

            # slice indices
            mid_depth = vol.shape[1] // 2
            mid_height = vol.shape[2] // 2

            # convert to numpy
            img_xy = safe_to_numpy(vol[0, mid_depth, :, :])
            img_yz = safe_to_numpy(vol[0, :, mid_height, :])

            # Show projections
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            axes[0].imshow(img_xy, cmap="gray")
            axes[0].set_title(f"Sample {i} - XY (depth={mid_depth})")
            axes[0].axis("off")

            axes[1].imshow(img_yz, cmap="gray")
            axes[1].set_title(f"Sample {i} - YZ (height={mid_height})")
            axes[1].axis("off")

            plt.tight_layout()
            plt.show()

            # User input
            choice = input("Good sample? [Y/N] (q to quit): ").strip().lower()

            # Close the figure to free resources
            try:
                plt.close(fig)
            except Exception:
                pass

            if choice == "y":
                if sample_id not in good_set:
                    append_id(GOOD_FILE, sample_id)
                    good_set.add(sample_id)
                    newly_good.append(sample_id)
                    print(f"[MARKED] Added to good: {sample_id}")
                else:
                    print(f"[INFO] Already present in good: {sample_id}")
            elif choice == "n":
                if sample_id not in problematic_set:
                    append_id(PROBLEMATIC_FILE, sample_id)
                    problematic_set.add(sample_id)
                    newly_problematic.append(sample_id)
                    print(f"[MARKED] Added to problematic: {sample_id}")
                else:
                    print(f"[INFO] Already present in problematic: {sample_id}")
            elif choice in ("q", ""):
                break
            else:
                print("[INFO] Unrecognized input, stopping.")
                break

    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user (KeyboardInterrupt).")

    # Final report
    if newly_good:
        print("\nNew good samples:")
        for p in newly_good:
            print(p)
    else:
        print("\nNo new good samples.")

    if newly_problematic:
        print("\nNew problematic samples:")
        for p in newly_problematic:
            print(p)
    else:
        print("\nNo new problematic samples.")

    print(f"\nTotals — good: {len(good_set)}, problematic: {len(problematic_set)}.")
    print(f"State files — good: {GOOD_FILE}, problematic: {PROBLEMATIC_FILE}")


if __name__ == "__main__":
    main()
