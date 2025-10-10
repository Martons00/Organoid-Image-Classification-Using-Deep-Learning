#!/bin/bash

#OAR -l gpu=1,walltime=3:00:00
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
python train.py --cfg config/OrganoidsINRIA_config_debug.yaml
