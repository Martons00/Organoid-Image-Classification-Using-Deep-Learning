#!/bin/bash

#OAR -l gpu=1,walltime=30:00:00
#OAR -p esterel39
#OAR -O OAR_%jobid%.out
#OAR -E OAR_%jobid%.err


set -euo pipefail
lscpu
nvidia-smi
pwd
# Attiva l'ambiente Python
source models/SwinUNETR/BRATS21/swin_unetr_env/bin/activate

# Avvia il training
python testing.py --cfg config/testing/DenseNet/training_lr_5e-2_warmup_focal_reduced.yaml --oar_id $OAR_JOB_ID
python testing.py --cfg config/testing/DenseNet/training_lr_5e-2_warmup_focal.yaml --oar_id $OAR_JOB_ID
python testing.py --cfg config/testing/Resnet18/training_lr_8e-2_aug_warmup_focal_reduced.yaml --oar_id $OAR_JOB_ID
python testing.py --cfg config/testing/ResNet/training_lr_1e-3_reduced.yaml --oar_id $OAR_JOB_ID
python testing.py --cfg config/testing/SwinUNETR+NOAH/training_lr_6e-4_aug_warmup_reduced.yaml --oar_id $OAR_JOB_ID

exit
