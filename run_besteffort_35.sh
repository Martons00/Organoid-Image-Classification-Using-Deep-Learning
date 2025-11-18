#!/bin/bash

#OAR -q besteffort
#OAR -l gpu=1,walltime=30:00:00
#OAR -p esterel35
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

set -euo pipefail
lscpu
nvidia-smi
pwd
# Attiva l'ambiente Python
source models/SwinUNETR/BRATS21/swin_unetr_env/bin/activate

# Avvia il training
python train.py --cfg config/training/SwinUNETR+NOAH/training_lr_6e-4_aug_warmup_reduced.yaml --oar_id $OAR_JOB_ID
exit
