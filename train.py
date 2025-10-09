# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

import argparse
import os
import pprint

import logging
import timeit
#from utils.trainer import run_training 

import numpy as np

from torch.utils.data import DataLoader, Subset

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from tensorboardX import SummaryWriter
from torchmetrics.classification import MulticlassAccuracy

from config import config
from config import update_config
from utils.criterion import CrossEntropy, DiceLoss, OhemCrossEntropy, BondaryLoss, FocalLoss
from utils.function import train, validate
from utils.utils_old import create_logger, FullModel,suppress_stdout
#from datasets.base_dataset import AugmentedDataset
from config import parse_args, config_to_args

import argparse
import os
from functools import partial

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel
import torch.utils.data.distributed
#from models import LinearWarmupCosineAnnealingLR
from utils.trainer import run_training
from utils.data_utils import get_loader

from monai.inferers import sliding_window_inference
from monai.losses import DiceLoss
from monai.metrics import DiceMetric
from monai.networks.nets import SwinUNETR
from monai.transforms import Activations, AsDiscrete, Compose
from monai.utils.enums import MetricReduction
from test import  SwinUNETREncoder
from dataset import OrganoidsINRIA3D

from typing import Tuple, Union
import torch
from torch.utils.data import Dataset, random_split
from sklearn.utils.class_weight import compute_class_weight

def split_dataset_random(
    dataset: Dataset,
    val_size: Union[int, float] = 0.2,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Divide un Dataset PyTorch in train/val in modo casuale.
    val_size: frazione (0<val<=1) oppure numero intero di campioni.
    Restituisce (train_subset, val_subset).
    """
    n = len(dataset)
    if isinstance(val_size, float):
        val_len = int(round(val_size * n))
    else:
        val_len = int(val_size)
    val_len = max(1, min(n - 1, val_len))
    train_len = n - val_len

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=g)
    return train_subset, val_subset

from typing import Tuple, Union, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

def split_dataset_stratified(
    dataset: Dataset,
    val_size: Union[int, float] = 0.2,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Divide un Dataset PyTorch in train/val mantenendo la distribuzione delle classi.
    val_size: frazione (0<val<=1) oppure numero intero di campioni.
    Restituisce (train_subset, val_subset).
    """
    n = len(dataset)
    labels = dataset.labels  # Assume che il dataset abbia un attributo .labels (np.array)
    
    if isinstance(val_size, float):
        val_len = int(round(val_size * n))
    else:
        val_len = int(val_size)
    val_len = max(1, min(n - 1, val_len))
    
    # Split stratificato usando sklearn
    indices = np.arange(n)
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_len, 
        stratify=labels, 
        random_state=seed
    )
    
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    
    return train_subset, val_subset

def create_stratified_debug_subset(
    dataset_subset: Subset, 
    original_labels: np.ndarray,
    n_samples: int, 
    seed: int = 42
) -> Subset:
    """
    Crea un subset stratificato per il debug da un Subset esistente.
    """
    if len(dataset_subset) <= n_samples:
        return dataset_subset
    
    # Ottieni le label corrispondenti agli indici del subset
    subset_labels = original_labels[dataset_subset.indices]
    
    # Verifica che ci siano abbastanza campioni per classe
    unique_classes, class_counts = np.unique(subset_labels, return_counts=True)
    min_samples_per_class = max(1, n_samples // len(unique_classes))
    
    # Se qualche classe ha troppo pochi campioni, usa un approccio diverso
    if np.any(class_counts < min_samples_per_class):
        # Prendi almeno 1 campione per classe, poi riempi casualmente
        debug_indices = []
        remaining_slots = n_samples
        
        for cls in unique_classes:
            cls_indices = np.where(subset_labels == cls)[0]
            n_take = min(len(cls_indices), max(1, remaining_slots // len(unique_classes)))
            np.random.seed(seed + cls)  # Seed diverso per ogni classe per riproducibilità
            selected = np.random.choice(cls_indices, size=n_take, replace=False)
            debug_indices.extend(selected)
            remaining_slots -= n_take
        
        # Riempi gli slot rimanenti casualmente
        if remaining_slots > 0:
            all_indices = np.arange(len(dataset_subset))
            available = np.setdiff1d(all_indices, debug_indices)
            if len(available) > 0:
                np.random.seed(seed)
                extra = np.random.choice(available, size=min(remaining_slots, len(available)), replace=False)
                debug_indices.extend(extra)
        
        debug_indices = np.array(debug_indices)
    else:
        # Split stratificato normale
        subset_indices = np.arange(len(dataset_subset))
        debug_indices, _ = train_test_split(
            subset_indices,
            train_size=n_samples,
            stratify=subset_labels,
            random_state=seed
        )
    
    # Mappa gli indici del subset agli indici originali del dataset
    original_indices = [dataset_subset.indices[i] for i in debug_indices]
    return Subset(dataset_subset.dataset, original_indices)


def main():
    args = parse_args()  # Aggiorna automaticamente la variabile globale config
    args.amp = not args.noamp


    
    args.logdir = "./runs/" + args.logdir
    if args.distributed:
        args.ngpus_per_node = torch.cuda.device_count()
        print("Found total gpus", args.ngpus_per_node)
        args.world_size = args.ngpus_per_node * args.world_size
        mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
    else:
        main_worker(gpu=0, args=args)

def main_worker(gpu, args):
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
    np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
    args.gpu = gpu
    if args.distributed:
        args.rank = args.rank * args.ngpus_per_node + gpu
        dist.init_process_group(
            backend=args.dist_backend, init_method=args.dist_url, world_size=args.world_size, rank=args.rank
        )
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False


    logger, final_output_dir, tb_log_dir = create_logger(
        args, args.logdir, 'train')
    
    
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    logger.info(pprint.pformat(args))
    logger.info(config)

    # Here we prepare the data loader
    dataset = OrganoidsINRIA3D(args.data_dir, exact_class_dir=args.exact_class)
    print("Dataset length is:", len(dataset))
    logger.info(f"Dataset length is: {len(dataset)}")
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
        drop_last=False,
    )


    num_classes = 3  # 0,1,2 + "other"=3
    labels = dataset.labels  # np.ndarray

    dataset_counts = np.bincount(labels, minlength=num_classes)

    print("Class distribution in the entire dataset:")
    for c, n in enumerate(dataset_counts):
        print(f"Class {c}: {n}")
        logger.info(f"Class {c}: {n}")

    if args.split_method == "random":
        train_set, val_set = split_dataset_random(dataset, val_size=0.2, seed=args.seed)
        print("Training set length:", len(train_set))
        logger.info(f"Training set length: {len(train_set)}")
        print("Validation set length:", len(val_set))
        logger.info(f"Validation set length: {len(val_set)}")
    elif args.split_method == "stratified":
        train_set, val_set = split_dataset_stratified(dataset, val_size=0.2, seed=args.seed)
        print("Training set length:", len(train_set))
        logger.info(f"Training set length: {len(train_set)}")
        print("Validation set length:", len(val_set))
        logger.info(f"Validation set length: {len(val_set)}")
    else:
        raise ValueError(f"Unsupported split method: {args.split_method}")
        
    
    # Eventuale sottoinsieme per il debug con split stratificato
    if args.debug:
        # Impostazioni debug
        DEBUG_TRAIN_SAMPLES = args.debug_train_samples if args.debug_train_samples > 0 else 20
        DEBUG_VAL_SAMPLES = args.debug_val_samples if args.debug_val_samples > 0 else 10
        train_set = create_stratified_debug_subset(
            train_set, labels, DEBUG_TRAIN_SAMPLES, seed=args.seed
        )
        val_set = create_stratified_debug_subset(
            val_set, labels, DEBUG_VAL_SAMPLES, seed=args.seed + 1
        )
        
        print(f"DEBUG: using {len(train_set)} samples for training (stratified)")
        print(f"DEBUG: using {len(val_set)} samples for validation (stratified)")
    
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        num_workers=args.workers,
        shuffle=True,
        pin_memory=True,
        drop_last=True,
    )
    validation_loader = DataLoader(
        val_set,
        batch_size=1,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    print(f"Training loader length is {len(train_loader)} batches")
    print(f"Validation loader length is {len(validation_loader)} batches")
    
    # Calcola le distribuzioni finali
    train_indices = train_set.indices if hasattr(train_set, 'indices') else list(range(len(train_set)))
    val_indices = val_set.indices if hasattr(val_set, 'indices') else list(range(len(val_set)))
    
    train_counts = np.bincount(labels[train_indices], minlength=num_classes)
    val_counts = np.bincount(labels[val_indices], minlength=num_classes)
    
    print("Class distribution in the training set:")
    for c, n in enumerate(train_counts):
        percentage = (n / len(train_set)) * 100 if len(train_set) > 0 else 0
        print(f"Train class {c}: {n} ({percentage:.1f}%)")
        logger.info(f"Train class {c}: {n} ({percentage:.1f}%)")
        
    print("Class distribution in the validation set:")
    for c, n in enumerate(val_counts):
        percentage = (n / len(val_set)) * 100 if len(val_set) > 0 else 0
        print(f"Val class {c}: {n} ({percentage:.1f}%)")
        logger.info(f"Val class {c}: {n} ({percentage:.1f}%)")



    print(args.rank, " gpu", args.gpu)
    if args.rank == 0:
        print("Batch size is:", args.batch_size, "epochs", args.max_epochs)
        logger.info("Batch size is: %d, epochs: %d", args.batch_size, args.max_epochs)

    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z), 
        in_channels=1, 
        out_channels=1, 
        feature_size=48,    
        use_checkpoint=True
    )

    if args.resume_ckpt:
        model_dict = torch.load(pretrained_pth)["state_dict"]
        model.load_state_dict(model_dict)
        print("Using pretrained weights")

    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(labels),y=labels)
    weights = torch.tensor(class_weights, dtype=torch.float)
    print("Class weights:", weights.numpy())
    logger.info("Class weights: " + str(weights.numpy()))
    loss_func = nn.CrossEntropyLoss(weight=weights.cuda(args.gpu))

    #loss_func = nn.CrossEntropyLoss()  # per classificazione
    acc_metric = MulticlassAccuracy(num_classes=args.out_channels, average='macro').cuda(args.gpu)



    if args.squared_dice:
        dice_loss = DiceLoss(
            to_onehot_y=False, sigmoid=True, squared_pred=True, smooth_nr=args.smooth_nr, smooth_dr=args.smooth_dr
        )
    else:
        dice_loss = DiceLoss(to_onehot_y=False, sigmoid=True)

    post_sigmoid = Activations(sigmoid=True)
    post_pred = AsDiscrete(argmax=False, logit_thresh=0.5)
    dice_acc = DiceMetric(include_background=True, reduction=MetricReduction.MEAN_BATCH, get_not_nans=True)
    model_inferer = partial(
        sliding_window_inference,
        roi_size=inf_size,
        sw_batch_size=args.sw_batch_size,
        predictor=model,
        overlap=args.infer_overlap,
    )
    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    logger.info("Total parameters count: %d", pytorch_total_params)

    best_acc = 0
    start_epoch = 0

    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")
        from collections import OrderedDict

        new_state_dict = OrderedDict()
        for k, v in checkpoint["state_dict"].items():
            new_state_dict[k.replace("backbone.", "")] = v
        model.load_state_dict(new_state_dict, strict=False)
        if "epoch" in checkpoint:
            start_epoch = checkpoint["epoch"]
        if "best_acc" in checkpoint:
            best_acc = checkpoint["best_acc"]
        print("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))
        logger.info("=> loaded checkpoint '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc))


    # Here we have to extract the encoder part from the pretrained model and load it
    # into our model

    model = SwinUNETREncoder(
        model, 
        num_classes=3, 
        num_features=768
    )

    # Here we add the classification head

    model.cuda(args.gpu)

    if args.distributed:
        torch.cuda.set_device(args.gpu)
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model.cuda(args.gpu)
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu], output_device=args.gpu)
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.optim_lr, weight_decay=args.reg_weight)
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(), lr=args.optim_lr, momentum=args.momentum, nesterov=True, weight_decay=args.reg_weight
        )
    else:
        logger.error("Unsupported Optimization Procedure: " + str(args.optim_name))
        raise ValueError("Unsupported Optimization Procedure: " + str(args.optim_name))

    if args.lrschedule == "warmup_cosine":
        scheduler = None
        #scheduler = LinearWarmupCosineAnnealingLR(
        #    optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        #)
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
        if args.checkpoint is not None:
            scheduler.step(epoch=start_epoch)
    else:
        scheduler = None

    semantic_classes = ["Dice_Val_TC", "Dice_Val_WT", "Dice_Val_ET"]


    #epoch_iters = int(loader[0].__len__() + loader[1].__len__() / config.batch_size / 1)
    epoch_iters = 100


    start = timeit.default_timer()
    end_epoch = args.max_epochs
    num_iters = args.max_epochs * epoch_iters


    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc_metric,
        args=args,
        model_inferer=model_inferer,
        scheduler=scheduler,
        start_epoch=start_epoch,
        post_sigmoid=post_sigmoid,
        post_pred=post_pred,
        semantic_classes=semantic_classes,
        writer_dict=writer_dict,

    )

    end = timeit.default_timer()

    return accuracy


if __name__ == "__main__":
    main()
