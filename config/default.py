import argparse

from yacs.config import CfgNode as CN

# Configurazione di default

_C = CN()

# System settings

_C.SYSTEM = CN()

_C.SYSTEM.seed = 304

_C.SYSTEM.distributed = False

_C.SYSTEM.world_size = 1

_C.SYSTEM.rank = 0

_C.SYSTEM.dist_url = "tcp://127.0.0.1:23456"

_C.SYSTEM.dist_backend = "nccl"

_C.SYSTEM.workers = 8

_C.SYSTEM.logs_dir = "./logs"

_C.SYSTEM.output_dir = "./outputs"

_C.SYSTEM.telegram_log = False

_C.SYSTEM.telegram_token = "./token/tlg_token.txt"

_C.SYSTEM.oar_id = 0

# Model settings

_C.MODEL = CN()

_C.MODEL.pretrained_model_name = "model_swinvit.pt"

_C.MODEL.name = "swinunetr"

_C.MODEL.pretrained_dir = "./pretrained_models/fold1_f48_ep300_4gpu_dice0_9059/"

_C.MODEL.checkpoint = ""

_C.MODEL.resume_ckpt = False

_C.MODEL.feature_size = 48

_C.MODEL.in_channels = 4

_C.MODEL.out_channels = 3

_C.MODEL.spatial_dims = 3

_C.MODEL.norm_name = "instance"

_C.MODEL.dropout_rate = 0.0

_C.MODEL.dropout_path_rate = 0.0

_C.MODEL.use_checkpoint = False

# Dataset settings

_C.DATASET = CN()

_C.DATASET.name = "OrganoidsINRIA"

_C.DATASET.data_dir = "/home/mraffael/martone_project/Organoids_Dataset/"

_C.DATASET.exact_class = False

_C.DATASET.ignore_label = 3

_C.DATASET.fold = 0

_C.DATASET.cache_dataset = False

_C.DATASET.roi_x = 128

_C.DATASET.roi_y = 128

_C.DATASET.roi_z = 128

# Augmentation settings

_C.AUGMENTATION = CN()

_C.AUGMENTATION.RandFlipd_prob = 0.2

_C.AUGMENTATION.RandRotate90d_prob = 0.2

_C.AUGMENTATION.RandScaleIntensityd_prob = 0.1

_C.AUGMENTATION.RandShiftIntensityd_prob = 0.1

# Training settings

_C.TRAINING = CN()

_C.TRAINING.max_epochs = 300

_C.TRAINING.batch_size = 1

_C.TRAINING.sw_batch_size = 1

_C.TRAINING.val_every = 100

_C.TRAINING.early_stopping = True

_C.TRAINING.patience_val = 3

_C.TRAINING.min_delta_val = 0.001

_C.TRAINING.patience_loss = 20

_C.TRAINING.min_delta_loss = 0.001

_C.TRAINING.folds = False

_C.TRAINING.k_folds = 5

_C.TRAINING.debug = False

_C.TRAINING.debug_train_samples = 40

_C.TRAINING.debug_val_samples = 10

_C.TRAINING.split_method = "stratified"  # "random" or "stratified"

_C.TRAINING.save_checkpoint = False

_C.TRAINING.noamp = False

_C.TRAINING.optim_lr = 1e-4

_C.TRAINING.optim_name = "adamw"

_C.TRAINING.reg_weight = 1e-5

_C.TRAINING.momentum = 0.99

_C.TRAINING.lrschedule = ""

_C.TRAINING.warmup_epochs = 50

# Loss settings

_C.LOSS = CN()
_C.LOSS.weight = 0.5

# Inference settings

_C.INFERENCE = CN()

_C.INFERENCE.infer_overlap = 0.5

# Logging settings

_C.LOGGING = CN()

_C.LOGGING.logdir = "test"

# VARIABILE GLOBALE CHE MANTIENE LA CONFIG AGGIORNATA
config = _C.clone()

def update_config(cfg, args):
    """Update config with args and yaml file."""
    cfg.defrost()
    
    
    if args.cfg:
        cfg.merge_from_file(args.cfg)
    
    if args.opts:
        cfg.merge_from_list(args.opts)
    
    cfg.set_new_allowed(False)
    cfg.freeze()

def get_config():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()

def parse_args():
    """Parse command line arguments and update global config."""
    global config  # Usa la variabile globale
    
    parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for OrganoidsINRIA Challenge')
    
    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="./config/OrganoidsINRIA_config_debug.yaml",
                        type=str)
    
    parser.add_argument('--seed', type=int, default=304)

    parser.add_argument('--oar_id', type=int, default=0)
    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    
    args = parser.parse_args()

    oar_id = args.oar_id
    
    # Aggiorna la config globale
    config = get_config()
    config_to_args(config)
    update_config(config, args)
    args = config_to_args(config)  # Ora usa la config AGGIORNATA dal YAML
    args.oar_id = oar_id  # Sovrascrivi con il valore passato da linea di comando
    
    return args

# Funzione di utilità per accedere ai parametri in modo compatibile
def config_to_args(cfg):
    """Convert config to argparse-like namespace for backward compatibility."""
    args = argparse.Namespace()
    
    # System
    args.seed = cfg.SYSTEM.seed
    args.distributed = cfg.SYSTEM.distributed
    args.world_size = cfg.SYSTEM.world_size
    args.rank = cfg.SYSTEM.rank
    args.dist_url = cfg.SYSTEM.dist_url
    args.dist_backend = cfg.SYSTEM.dist_backend
    args.workers = cfg.SYSTEM.workers
    args.output_dir = cfg.SYSTEM.output_dir
    args.logs_dir = cfg.SYSTEM.logs_dir
    args.telegram_log = cfg.SYSTEM.telegram_log
    args.token = None
    if cfg.SYSTEM.telegram_log:
        args.token = cfg.SYSTEM.telegram_token
    args.oar_id = cfg.SYSTEM.oar_id 

    # Model
    args.model_name = cfg.MODEL.name
    args.pretrained_model_name = cfg.MODEL.pretrained_model_name
    args.pretrained_dir = cfg.MODEL.pretrained_dir
    args.checkpoint = cfg.MODEL.checkpoint if cfg.MODEL.checkpoint else None
    args.resume_ckpt = cfg.MODEL.resume_ckpt
    args.feature_size = cfg.MODEL.feature_size
    args.in_channels = cfg.MODEL.in_channels
    args.out_channels = cfg.MODEL.out_channels
    args.spatial_dims = cfg.MODEL.spatial_dims
    args.norm_name = cfg.MODEL.norm_name
    args.dropout_rate = cfg.MODEL.dropout_rate
    args.dropout_path_rate = cfg.MODEL.dropout_path_rate
    args.use_checkpoint = cfg.MODEL.use_checkpoint
    
    # Dataset
    args.dataset_name = cfg.DATASET.name
    args.data_dir = cfg.DATASET.data_dir
    args.exact_class = cfg.DATASET.exact_class
    args.ignore_label = cfg.DATASET.ignore_label
    args.fold = cfg.DATASET.fold
    args.cache_dataset = cfg.DATASET.cache_dataset
    args.roi_x = cfg.DATASET.roi_x
    args.roi_y = cfg.DATASET.roi_y
    args.roi_z = cfg.DATASET.roi_z
    
    # Augmentation
    args.RandFlipd_prob = cfg.AUGMENTATION.RandFlipd_prob
    args.RandRotate90d_prob = cfg.AUGMENTATION.RandRotate90d_prob
    args.RandScaleIntensityd_prob = cfg.AUGMENTATION.RandScaleIntensityd_prob
    args.RandShiftIntensityd_prob = cfg.AUGMENTATION.RandShiftIntensityd_prob
    
    # Training
    args.max_epochs = cfg.TRAINING.max_epochs
    args.batch_size = cfg.TRAINING.batch_size
    args.sw_batch_size = cfg.TRAINING.sw_batch_size
    args.val_every = cfg.TRAINING.val_every
    args.early_stopping = cfg.TRAINING.early_stopping
    args.patience_val = cfg.TRAINING.patience_val
    args.min_delta_val = cfg.TRAINING.min_delta_val
    args.patience_loss = cfg.TRAINING.patience_loss
    args.min_delta_loss = cfg.TRAINING.min_delta_loss
    args.folds = cfg.TRAINING.folds 
    args.k_folds = cfg.TRAINING.k_folds
    args.debug = cfg.TRAINING.debug
    args.debug_train_samples = cfg.TRAINING.debug_train_samples
    args.debug_val_samples = cfg.TRAINING.debug_val_samples
    args.split_method = cfg.TRAINING.split_method
    args.save_checkpoint = cfg.TRAINING.save_checkpoint
    args.noamp = cfg.TRAINING.noamp
    args.optim_lr = cfg.TRAINING.optim_lr
    args.optim_name = cfg.TRAINING.optim_name
    args.reg_weight = cfg.TRAINING.reg_weight
    args.momentum = cfg.TRAINING.momentum
    args.lrschedule = cfg.TRAINING.lrschedule
    args.warmup_epochs = cfg.TRAINING.warmup_epochs
    
    # Loss
    args.loss_weight = cfg.LOSS.weight
    # Inference
    args.infer_overlap = cfg.INFERENCE.infer_overlap
    
    # Logging
    args.logdir = cfg.LOGGING.logdir
    
    # AMP (dedotta da noamp)
    args.amp = not cfg.TRAINING.noamp
    
    return args

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)