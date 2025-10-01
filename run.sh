bash#!/bin/bash

#OAR -q production 
#OAR -l host=1/gpu=1,walltime=3:00:00
#OAR -p gpu-16GB AND gpu_compute_capability_major>=5
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err 
hostname
nvidia-smi

# make use of a python torch environment
module load conda
conda activate pytorch_env
python3 tools/train.py --cfg configs/SwinUNETR+ML_Decoder/train.yaml GPUS "[0]";
python3 tools/train.py --cfg configs/SwinUNETR+NOAH/train.yaml GPUS "[0]";