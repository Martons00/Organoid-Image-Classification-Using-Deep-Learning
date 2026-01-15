#!/bin/bash

source models/SwinUNETR/BRATS21/swin_unetr_env/bin/activate


# Test con 5 campioni, selezionando 32 slice
python preprocessing/sliceSelector.py \
    --input_dir /home/mraffael/martone_project/Organoids_Dataset/test_set/Cystiques \
    --n_samples 5 \
    --n_slices 32 \
    --method feature_variance \
    --save_path results_32slices.png


