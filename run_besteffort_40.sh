#!/bin/bash

#OAR -q besteffort
#OAR -l gpu=1,walltime=24:00:00
#OAR -p esterel40
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err

set -euo pipefail
lscpu
nvidia-smi
pwd
# Attiva l'ambiente Python
source models/SwinUNETR/BRATS21/swin_unetr_env/bin/activate

# Avvia il training
python train.py --cfg config/training/DenseNet/training_lr_5e-2_warmup_focal.yaml --oar_id $OAR_JOB_ID

exit
