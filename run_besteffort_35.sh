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
python testing.py --cfg config/training/DenseNet/training_lr_1e-2_128.yaml --oar_id $OAR_JOB_ID

exit
