# ==============================================================================
# K-Fold Cross-Validation Training Script
# Basato su train.py con Support per K-Fold stratificato
# ==============================================================================

import os
import time
import timeit
import pprint
import numpy as np
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

from torchmetrics.classification import MulticlassAccuracy
from tensorboardX import SummaryWriter
from sklearn.utils.class_weight import compute_class_weight

# Imports dal progetto
from config import config, parse_args
from utils.utils import create_logger
from utils.trainer import run_training, run_testing
from utils.data_utils import (
    get_loader,
    create_kfold_splits_stratified,
    create_fold_dataloaders,
    verify_balance,
    send_alert,
    build_training_message,
)
from tools.loss import FocalLoss, LabelSmoothingLoss, DiversityLoss, CombinedLoss, CenterLoss
from dataset import OrganoidsINRIA3D
from models import SwinUNETREncoder, resnet, ResNet_3D, DenseNet_3D, SwinVit_3D
from models.ML_Decoder_main.src_files.ml_decoder.ml_decoder import MLDecoder
from models.NOAH_main.modules.noah import NOAH

import asyncio

# ==============================================================================
# MAIN ENTRY POINT PER K-FOLD
# ==============================================================================

def main():
    """Main entry point per k-fold cross-validation."""
    try:
        args, cfgs = parse_args()
        args.amp = not args.noamp
        args.start_epoch = 0
        
        # Numero di fold da usare
        n_splits = getattr(args, 'n_splits', 5)
        args.n_splits = n_splits
        
        if args.distributed:
            # args.ngpus_per_node = torch.cuda.device_count()
            print(f"Found total gpus: {args.ngpus_per_node}")
            # args.world_size = args.ngpus_per_node * args.world_size
            # mp.spawn(main_worker_kfold, nprocs=args.ngpus_per_node, args=(args,))
        else:
            main_worker_kfold(gpu=0, args=args, configs=cfgs)
            
    except Exception as e:
        print("An exception occurred during k-fold training:")
        print(str(e))
        if args.telegram_log:
            message = f"🚨 *ERROR*\nK-Fold Training Failed:\n{str(e)}"
            asyncio.run(send_alert(args.oar_id, message, token_file=args.token))
        raise e


def main_worker_kfold(gpu, args, configs):
    """Main worker per k-fold cross-validation."""
    
    # Setup base
    _setup_distributed(gpu, args)
    _setup_logging_and_device(args)
    
    # Crea directories principali
    logger, final_output_dir, tb_log_dir = create_logger(args, args.logdir, args.model_name)
    
    def log(msg, level="info"):
        """Helper per logging unificato."""
        print(msg)
        if logger:
            getattr(logger, level)(msg)
    
    log(pprint.pformat(vars(args)))
    log("")
    
    # Salva config
    try:
        config_path = os.path.join(final_output_dir, "config.txt")
        with open(config_path, "w") as cf:
            cf.write(configs.dump())
        log(f"Saved configuration to: {config_path}")
    except Exception as e:
        log(f"Failed to write config file: {e}", level="warning")
    
    # Telegram notification
    if args.telegram_log:
        message = build_training_message(args)
        _send_telegram_safe(args, message)
    
    log(f"Using GPU: {args.gpu}")
    log(f"Starting K-Fold Cross-Validation with {args.n_splits} folds")
    log("*" * 50)
    
    # ====================
    # CARICA IL DATASET
    # ====================
    
    full_dataset = OrganoidsINRIA3D(args.data_dir + "/train_set", exact_class_dir=args.exact_class)
    labels = np.array(full_dataset.labels)
    num_classes = 3
    
    log(f"Full dataset length: {len(full_dataset)}")
    dataset_counts = np.bincount(labels, minlength=num_classes)
    log("Class distribution in full dataset:")
    for c, n in enumerate(dataset_counts):
        log(f"  Class {c}: {n}")
    
    # ====================
    # CREA K-FOLD SPLITS
    # ====================
    
    log(f"\nCreating {args.n_splits}-Fold Stratified Splits...")
    fold_splits = create_kfold_splits_stratified(
        full_dataset, 
        args.n_splits, 
        args.seed
    )
    
    log(f"Created {len(fold_splits)} folds")
    log("*" * 50)
    
    # ====================
    # TRAINING LOOP K-FOLD
    # ====================
    
    fold_results = {
        'fold_train_acc': [],
        'fold_val_acc': [],
        'fold_test_acc': [],
        'fold_train_loss': [],
        'fold_metrics': [],
    }
    
    start_total = timeit.default_timer()
    
    # ✅ CORRETTO - train_set e val_set sono già tuple di indici da StratifiedKFold
    for fold_idx, (train_indices, val_indices) in enumerate(fold_splits):
        
        log(f"\n{'='*50}")
        log(f"FOLD {fold_idx + 1}/{args.n_splits}")
        log(f"{'='*50}")

        # Crea directory per questo fold 
        fold_output_dir = os.path.join(final_output_dir, f"fold_{fold_idx}") 
        os.makedirs(fold_output_dir, exist_ok=True)  
        fold_tb_dir = os.path.join(fold_output_dir, "tensorboard") 
        os.makedirs(fold_tb_dir, exist_ok=True)  
        
        writer_dict = { 'writer': SummaryWriter(fold_tb_dir), 'train_global_steps': 0, 'valid_global_steps': 0, }
        
        # Verifica balance nei fold - USA DIRETTAMENTE gli indici!
        log(f"\nFold {fold_idx} - Train set ({len(train_indices)} samples):")
        train_labels_fold = labels[train_indices]  # ✅ Accesso DIRETTO agli indici
        train_counts = np.bincount(train_labels_fold, minlength=num_classes)
        for c, n in enumerate(train_counts):
            pct = (n / len(train_indices)) * 100
            log(f"  Class {c}: {n} ({pct:.1f}%)")
        
        log(f"\nFold {fold_idx} - Val set ({len(val_indices)} samples):")
        val_labels_fold = labels[val_indices]  # ✅ Accesso DIRETTO agli indici
        val_counts = np.bincount(val_labels_fold, minlength=num_classes)
        for c, n in enumerate(val_counts):
            pct = (n / len(val_indices)) * 100
            log(f"  Class {c}: {n} ({pct:.1f}%)")
        
        # Crea dataloaders per questo fold
        log(f"\nCreating dataloaders for fold {fold_idx}...")
        train_loader, val_loader = create_fold_dataloaders(
            dataset=full_dataset,
            train_idx=train_indices,  # ✅ Passa direttamente gli indici
            val_idx=val_indices,      # ✅ Passa direttamente gli indici
            batch_size=args.batch_size,
            num_workers=args.workers,
        )
        
        # ... resto del training ...

        
        log(f"Train loader: {len(train_loader)} batches")
        log(f"Val loader: {len(val_loader)} batches")
        
        # Setup model per questo fold
        log(f"\nSetting up model for fold {fold_idx}...")
        model = _setup_model(args, logger, log)
        
        # Setup optimizer e scheduler
        optimizer = _setup_optimizer(model, args, logger, log)
        scheduler = _setup_scheduler(optimizer, args)
        
        # Setup loss function
        class_weights = compute_class_weight(
            class_weight='balanced',
            classes=np.unique(train_labels_fold),
            y=train_labels_fold
        )
        weights_tensor = torch.tensor(class_weights, dtype=torch.float).cuda(args.gpu) * 10
        loss_func = _setup_loss(args, weights_tensor, log)
        
        acc_metric = MulticlassAccuracy(num_classes=3, average='macro').cuda(args.gpu)
        
        # Training per questo fold
        log(f"\nStarting training for fold {fold_idx}...")
        start_fold = timeit.default_timer()
        
        train_loss, train_acc, val_acc_max, best_metrics_training = run_training(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            loss_func=loss_func,
            acc_func=acc_metric,
            args=args,
            scheduler=scheduler,
            start_epoch=0,
            writer_dict=writer_dict,
            final_output_dir=fold_output_dir,
            logger=logger,
        )
        
        end_fold = timeit.default_timer()
        fold_time = end_fold - start_fold
        
        # Salva risultati fold
        fold_results['fold_train_loss'].append(train_loss)
        fold_results['fold_train_acc'].append(train_acc)
        fold_results['fold_val_acc'].append(val_acc_max)
        fold_results['fold_metrics'].append(best_metrics_training)
        
        log(f"\nFold {fold_idx} Training Summary:")
        log(f"  Train Loss: {train_loss:.4f}")
        log(f"  Train Acc:  {train_acc:.4f}")
        log(f"  Val Acc:    {val_acc_max:.4f}")
        log(f"  Time:       {time.strftime('%H:%M:%S', time.gmtime(fold_time))}")
        
        writer_dict['writer'].close()
        torch.cuda.empty_cache()
        
        log(f"{'='*50}\n")
    
    # ====================
    # RISULTATI FINALI
    # ====================
    
    end_total = timeit.default_timer()
    total_time = end_total - start_total
    
    log(f"\n{'='*50}")
    log("K-FOLD CROSS-VALIDATION RESULTS")
    log(f"{'='*50}")
    
    train_accs = np.array(fold_results['fold_train_acc'])
    val_accs = np.array(fold_results['fold_val_acc'])
    train_losses = np.array(fold_results['fold_train_loss'])
    
    log(f"\nTrain Accuracy:")
    log(f"  Mean:  {train_accs.mean():.4f} ± {train_accs.std():.4f}")
    log(f"  Min:   {train_accs.min():.4f}")
    log(f"  Max:   {train_accs.max():.4f}")
    
    log(f"\nValidation Accuracy:")
    log(f"  Mean:  {val_accs.mean():.4f} ± {val_accs.std():.4f}")
    log(f"  Min:   {val_accs.min():.4f}")
    log(f"  Max:   {val_accs.max():.4f}")
    
    log(f"\nTrain Loss:")
    log(f"  Mean:  {train_losses.mean():.4f} ± {train_losses.std():.4f}")
    log(f"  Min:   {train_losses.min():.4f}")
    log(f"  Max:   {train_losses.max():.4f}")
    
    log(f"\nFold-by-Fold Results:")
    log("| Fold | Train Acc | Val Acc |")
    log("|------|-----------|---------|")
    for fold_idx, (ta, va) in enumerate(zip(train_accs, val_accs)):
        log(f"| {fold_idx+1:4d} | {ta:.4f}    | {va:.4f}  |")
    
    time_str = time.strftime("%H hours %M minutes %S seconds", time.gmtime(total_time))
    log(f"\nTotal K-Fold Time: {time_str}")
    
    log(f"{'='*50}\n")
    
    return fold_results


# ==============================================================================
# HELPER FUNCTIONS (basate su train.py)
# ==============================================================================

def _setup_distributed(gpu, args):
    """Setup per training distribuito."""
    if args.distributed:
        torch.multiprocessing.set_start_method("fork", force=True)
        np.set_printoptions(formatter={"float": "{: 0.3f}".format}, suppress=True)
        args.gpu = gpu
        if args.distributed:
            args.rank = args.rank * args.ngpus_per_node + gpu
            dist.init_process_group(
                backend=args.dist_backend,
                init_method=args.dist_url,
                world_size=args.world_size,
                rank=args.rank
            )


def _setup_logging_and_device(args):
    """Setup device e cudnn."""
    torch.cuda.set_device(args.gpu)
    torch.backends.cudnn.benchmark = True
    args.test_mode = False


def _setup_model(args, logger, log):
    """Setup e caricamento del modello (importato da train.py)."""
    from train import _setup_model as original_setup_model
    return original_setup_model(args, logger, log)


def _setup_optimizer(model, args, logger, log):
    """Setup optimizer (importato da train.py)."""
    from train import _setup_optimizer as original_setup_optimizer
    return original_setup_optimizer(model, args, logger, log)


def _setup_scheduler(optimizer, args):
    """Setup scheduler (importato da train.py)."""
    from train import _setup_scheduler as original_setup_scheduler
    return original_setup_scheduler(optimizer, args)


def _setup_loss(args, class_weights, log):
    """Setup loss function (importato da train.py)."""
    from train import _setup_loss as original_setup_loss
    return original_setup_loss(args, class_weights, log)


def _get_indices(dataset, default_length):
    """Helper per ottenere indices da dataset."""
    if hasattr(dataset, 'indices'):
        return dataset.indices
    return list(range(default_length))


def _send_telegram_safe(args, message):
    """Helper per telegram con error handling."""
    try:
        asyncio.run(send_alert(args.oar_id, message, token_file=args.token))
    except Exception as e:
        print(f"[Warning] Telegram notification failed: {e}")


# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()
