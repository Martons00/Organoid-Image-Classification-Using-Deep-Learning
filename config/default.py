import argparse
import yaml
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
_C.SYSTEM.output_dir = "./outputs"

# Model settings
_C.MODEL = CN()
_C.MODEL.pretrained_model_name = "model.pt"
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
_C.DATASET.data_dir = "/dataset/OrganoidsINRIA/"
_C.DATASET.json_list = "./jsons/OrganoidsINRIA_folds.json"
_C.DATASET.fold = 0
_C.DATASET.cache_dataset = False
_C.DATASET.a_min = -175.0
_C.DATASET.a_max = 250.0
_C.DATASET.b_min = 0.0
_C.DATASET.b_max = 1.0
_C.DATASET.space_x = 1.5
_C.DATASET.space_y = 1.5
_C.DATASET.space_z = 2.0
_C.DATASET.roi_x = 96
_C.DATASET.roi_y = 96
_C.DATASET.roi_z = 96

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
_C.TRAINING.sw_batch_size = 4
_C.TRAINING.val_every = 100
_C.TRAINING.save_checkpoint = False
_C.TRAINING.noamp = False
_C.TRAINING.optim_lr = 1e-4
_C.TRAINING.optim_name = "adamw"
_C.TRAINING.reg_weight = 1e-5
_C.TRAINING.momentum = 0.99
_C.TRAINING.lrschedule = "warmup_cosine"
_C.TRAINING.warmup_epochs = 50

# Loss settings
_C.LOSS = CN()
_C.LOSS.smooth_dr = 1e-6
_C.LOSS.smooth_nr = 0.0
_C.LOSS.squared_dice = False

# Inference settings
_C.INFERENCE = CN()
_C.INFERENCE.infer_overlap = 0.5

# Logging settings
_C.LOGGING = CN()
_C.LOGGING.logdir = "test"

def update_config(cfg, args):
    """Update config with args and yaml file."""
    cfg.defrost()

    if args.cfg:
        cfg.merge_from_file(args.cfg)

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

def get_config():
    """Get a yacs CfgNode object with default values."""
    return _C.clone()

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Swin UNETR segmentation pipeline for OrganoidsINRIA Challenge')

    parser.add_argument('--cfg',
                        help='experiment configure file name',
                        default="configs/OrganoidsINRIA_config.yaml",
                        type=str)
    parser.add_argument('--seed', type=int, default=304)    
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)

    args = parser.parse_args()

    # Get config
    config = get_config()
    update_config(config, args)

    return args, config

# Funzione di utilità per accedere ai parametri in modo compatibile
def config_to_args(config):
    """Convert config to argparse-like namespace for backward compatibility."""
    args = argparse.Namespace()

    # System
    args.seed = config.SYSTEM.seed
    args.distributed = config.SYSTEM.distributed
    args.world_size = config.SYSTEM.world_size
    args.rank = config.SYSTEM.rank
    args.dist_url = config.SYSTEM.dist_url
    args.dist_backend = config.SYSTEM.dist_backend
    args.workers = config.SYSTEM.workers
    args.output_dir = config.SYSTEM.output_dir
    args.logs_dir = config.SYSTEM.logs_dir 

    # Model
    args.model_name = config.MODEL.name
    args.pretrained_model_name = config.MODEL.pretrained_model_name
    args.pretrained_dir = config.MODEL.pretrained_dir
    args.checkpoint = config.MODEL.checkpoint if config.MODEL.checkpoint else None
    args.resume_ckpt = config.MODEL.resume_ckpt
    args.feature_size = config.MODEL.feature_size
    args.in_channels = config.MODEL.in_channels
    args.out_channels = config.MODEL.out_channels
    args.spatial_dims = config.MODEL.spatial_dims
    args.norm_name = config.MODEL.norm_name
    args.dropout_rate = config.MODEL.dropout_rate
    args.dropout_path_rate = config.MODEL.dropout_path_rate
    args.use_checkpoint = config.MODEL.use_checkpoint

    # Dataset
    args.dataset_name = config.DATASET.name
    args.data_dir = config.DATASET.data_dir
    args.json_list = config.DATASET.json_list
    args.fold = config.DATASET.fold
    args.cache_dataset = config.DATASET.cache_dataset
    args.a_min = config.DATASET.a_min
    args.a_max = config.DATASET.a_max
    args.b_min = config.DATASET.b_min
    args.b_max = config.DATASET.b_max
    args.space_x = config.DATASET.space_x
    args.space_y = config.DATASET.space_y
    args.space_z = config.DATASET.space_z
    args.roi_x = config.DATASET.roi_x
    args.roi_y = config.DATASET.roi_y
    args.roi_z = config.DATASET.roi_z

    # Augmentation
    args.RandFlipd_prob = config.AUGMENTATION.RandFlipd_prob
    args.RandRotate90d_prob = config.AUGMENTATION.RandRotate90d_prob
    args.RandScaleIntensityd_prob = config.AUGMENTATION.RandScaleIntensityd_prob
    args.RandShiftIntensityd_prob = config.AUGMENTATION.RandShiftIntensityd_prob

    # Training
    args.max_epochs = config.TRAINING.max_epochs
    args.batch_size = config.TRAINING.batch_size
    args.sw_batch_size = config.TRAINING.sw_batch_size
    args.val_every = config.TRAINING.val_every
    args.save_checkpoint = config.TRAINING.save_checkpoint
    args.noamp = config.TRAINING.noamp
    args.optim_lr = config.TRAINING.optim_lr
    args.optim_name = config.TRAINING.optim_name
    args.reg_weight = config.TRAINING.reg_weight
    args.momentum = config.TRAINING.momentum
    args.lrschedule = config.TRAINING.lrschedule
    args.warmup_epochs = config.TRAINING.warmup_epochs

    # Loss
    args.smooth_dr = config.LOSS.smooth_dr
    args.smooth_nr = config.LOSS.smooth_nr
    args.squared_dice = config.LOSS.squared_dice

    # Inference
    args.infer_overlap = config.INFERENCE.infer_overlap

    # Logging
    args.logdir = config.LOGGING.logdir

    return args

if __name__ == '__main__':
    import sys
    with open(sys.argv[1], 'w') as f:
        print(_C, file=f)

