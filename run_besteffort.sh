#!/bin/bash

#OAR -q besteffort
#OAR -l gpu=1,walltime=4:00:00
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
#python train.py --cfg config/OrganoidsINRIA_config_debug.yaml --oar_id $OAR_JOB_ID
#python train.py --cfg config/SwinUNETR+ML_Decoder/training.yaml --oar_id $OAR_JOB_ID

#python train.py --cfg config/SwinUNETR/training.yaml --oar_id $OAR_JOB_ID
#python train.py --cfg config/SwinUNETR/training_lr_5e-5.yaml --oar_id $OAR_JOB_ID
python train.py --cfg config/SwinUNETR/training_lr_1e-5.yaml --oar_id $OAR_JOB_ID
#python train.py --cfg config/SwinUNETR/training_lr_5e-5.yaml --oar_id 2068528
#python train.py --cfg config/SwinUNETR/training_lr_5e-6.yaml --oar_id $OAR_JOB_ID
