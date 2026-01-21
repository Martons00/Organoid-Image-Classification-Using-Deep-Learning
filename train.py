# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

# ========== Standard Library ==========
import os
import time
import timeit
import pprint
from pathlib import Path
import random

# ========== Third-party Libraries ==========
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorboardX import SummaryWriter

# ========== PyTorch Core ==========
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
from torch.utils.data import Sampler
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# ========== Metrics ==========
from torchmetrics.classification import MulticlassAccuracy

# ========== MONAI ==========
from monai.networks.nets import SwinUNETR,DenseNet201

# ========== Project-Specific ==========
from config import config, parse_args
from utils.utils import create_logger
from utils.trainer import run_training, run_testing,run_mc_droput_testing
from utils.data_utils import (
    split_dataset_balanced,
    split_dataset_percentage,
    split_dataset_random,
    split_dataset_stratified,
    create_stratified_debug_subset,
    create_balanced_debug_subset,
    verify_balance,
)

from utils.funcs_telegram import (
    send_alert,
    build_training_message,
)
from tools.loss import (
FocalLoss, LabelSmoothingLoss, DiversityLoss, CombinedLoss, CenterLoss
)

import asyncio
from models.ML_Decoder_main.src_files.ml_decoder.ml_decoder import MLDecoder
from models.NOAH_main.modules.noah import NOAH
from dataset import OrganoidsINRIA3D
from models import SwinUNETREncoder,resnet,ResNet_3D,DenseNet_3D,SwinVit_3D

# from datasets.base_dataset import AugmentedDataset
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR  # Uncomment if used
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    LinearLR,
    SequentialLR,
)

def main():
    try:
        args,cfgs = parse_args()  # Aggiorna automaticamente la variabile globale config
        args.amp = not args.noamp
        args.start_epoch = 0
        
        if args.distributed:
            args.ngpus_per_node = torch.cuda.device_count()
            print("Found total gpus", args.ngpus_per_node)
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
        else:
            main_worker(gpu=0, args=args, configs=cfgs)
    except Exception as e:
        print("An exception occurred during training:")
        print(str(e))
        if args.telegram_log:
            message = f"🚨 *ERROR*\nAn exception occurred during training:\n{str(e)}"
            asyncio.run(send_alert(args.oar_id, message, token_file=args.token))
        raise e  # Re-raise the exception for further handling if needed

def main_worker(gpu, args, configs):
    """Main training worker con setup distribuito."""
    # Setup base
    _setup_distributed(gpu, args)
    _setup_logging_and_device(args)
    
    # Crea logger e directories
    logger, final_output_dir, tb_log_dir = create_logger(args, args.logdir, args.model_name)
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }
    
    # Log helper
    def log(msg, level="info"):
        """Helper per logging unificato."""
        print(msg)
        if logger:
            getattr(logger, level)(msg)
    
    log(pprint.pformat(vars(args)))
    log("")
    # Save args/config to a file named "config" in the final output directory
    try:
        config_path = os.path.join(final_output_dir, "config.txt")
        with open(config_path, "w") as cf:
            cf.write(configs.dump())
        log(f"Saved configuration to: {config_path}")
    except Exception as e:
        log(f"Failed to write config file: {e}", level="warning")
    
    # Telegram notification iniziale
    if args.telegram_log:
        message = build_training_message(args)
        _send_telegram_safe(args, message)
    
    log(f"Using GPU: {args.gpu}")

    # Setup model
    model = _setup_model(args, logger, log)
    
    # Setup optimizer e scheduler
    optimizer = _setup_optimizer(model, args, logger, log)
    scheduler = _setup_scheduler(optimizer, args)
    
    # Setup dataset e dataloaders
    train_loader, val_loader, class_weights = _setup_data(args, logger, log)

    test_loader = _setup_data_test(args, logger, log)
    
    # Setup loss function
    loss_func = _setup_loss(args, class_weights, log)
    acc_metric = MulticlassAccuracy(num_classes=3, average='macro').cuda(args.gpu)
    
    # Training
    log("*" * 50)
    log("Starting training...")
    
    start = timeit.default_timer()
    train_loss,train_acc,val_acc_max, best_metrics_training = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc_metric,
        args=args,
        scheduler=scheduler,
        start_epoch=args.start_epoch if args.start_epoch else 0,
        writer_dict=writer_dict,
        final_output_dir=final_output_dir,
        logger=logger,
    )

    checkpoint = final_output_dir + "/training/best_model.pt"
    _maybe_load_checkpoint(model, checkpoint, log=log, strict=False, args=args)

    test_acc,test_metrics = run_testing(
        model=model,
        test_loader=test_loader,
        acc_func=acc_metric,
        loss_func=loss_func,
        args=args,
        writer_dict=writer_dict,
        final_output_dir=final_output_dir,
        logger=logger,
    )

    if args.dropout_rate > 0.0:
        test_loss_mc, test_acc_mc, test_acc_per_class, test_mc_metrics, uncertainties = run_mc_droput_testing(
            model=model,
            test_loader=test_loader,
            acc_func=acc_metric,
            loss_func=loss_func,
            args=args,
            writer_dict=writer_dict,
            final_output_dir=final_output_dir,
            logger=logger,
        )

    
    end = timeit.default_timer()
    time_end = end - start
    time_str = time.strftime("%H hours %M minutes %S seconds", time.gmtime(time_end))

    log(f"Total time spent: {time_str}")
    log("")
    log("Final Results Summary:")
    wa = best_metrics_training.get("weighted_avg", {})
    accuracy = best_metrics_training.get("accuracy", 0.0)
    log("| TrainLoss | TrainAcc | ValAccMax | W_TrainF1 | W_TrainPrecision | W_TrainRecall | W_TrainSpecificity | TestAcc | W_TestF1 | W_TestPrecision | W_TestRecall | W_TestSpecificity |")
    t_wa = test_metrics.get("weighted_avg", {})
    t_accuracy = test_metrics.get("accuracy", 0.0)
    log("| {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} | {:.4f} |".format(
        train_loss,
        train_acc,
        accuracy,
        wa.get("f1", 0.0),
        wa.get("precision", 0.0),
        wa.get("recall", 0.0),
        wa.get("specificity", 0.0),
        t_accuracy,
        t_wa.get("f1", 0.0),
        t_wa.get("precision", 0.0),
        t_wa.get("recall", 0.0),
        t_wa.get("specificity", 0.0)
    ))
    log("")
    log("| Detailed Test MC Metrics |")
    mc_acc = test_mc_metrics.get("accuracy", 0.0)
    mc_per_class = test_mc_metrics.get("per_class", {})
    mc_macro = test_mc_metrics.get("macro_avg", {})
    mc_weighted = test_mc_metrics.get("weighted_avg", {})
    mc_micro = test_mc_metrics.get("micro_avg", {})

    if args.dropout_rate > 0.0:
        log(f"MC Accuracy: {mc_acc:.4f}")
        log("Per-class metrics (MC):")
        log("class | precision | recall | specificity | f1 | support")
        for c in sorted(mc_per_class.keys()):
            m = mc_per_class[c]
            log(
                f"{c:5d} | "
                f"{m['precision']:.4f} | "
                f"{m['recall']:.4f} | "
                f"{m['specificity']:.4f} | "
                f"{m['f1']:.4f} | "
                f"{m['support']:7d}"
            )

        log("Macro avg (MC): "
            f"precision={mc_macro.get('precision', 0.0):.4f}, "
            f"recall={mc_macro.get('recall', 0.0):.4f}, "
            f"specificity={mc_macro.get('specificity', 0.0):.4f}, "
            f"f1={mc_macro.get('f1', 0.0):.4f}")

        log("Weighted avg (MC): "
            f"precision={mc_weighted.get('precision', 0.0):.4f}, "
            f"recall={mc_weighted.get('recall', 0.0):.4f}, "
            f"specificity={mc_weighted.get('specificity', 0.0):.4f}, "
            f"f1={mc_weighted.get('f1', 0.0):.4f}")

        log("Micro avg (MC): "
            f"precision={mc_micro.get('precision', 0.0):.4f}, "
            f"recall={mc_micro.get('recall', 0.0):.4f}, "
            f"specificity={mc_micro.get('specificity', 0.0):.4f}, "
            f"f1={mc_micro.get('f1', 0.0):.4f}")




    writer_dict['writer'].close()
    torch.cuda.empty_cache()
    
    return 0


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

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


def _load_checkpoint_dict(path):
    # Carica un checkpoint flessibilmente (state_dict o modello diretto)
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    sd = ckpt.get("state_dict", ckpt)
    if "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]
    return sd, ckpt

def _remap_keys_for_swinunetr(sd, log, skip_patterns=None):
    # Rinomina chiavi per compatibilità e salta quelle indesiderate
    skip_patterns = skip_patterns or ['out.conv.conv.weight', 'out.conv.conv.bias']
    new_sd = {}
    for k, v in sd.items():
        if any(pat in k for pat in skip_patterns):
            log(f"Skipping layer {k} due to skip pattern")
            continue
        if k.startswith("module."):
            new_key = "swinViT." + k[len("module."):]
            new_key = new_key.replace("fc", "linear")
        else:
            new_key = k
        new_sd[new_key] = v
    return new_sd

def _remap_keys_strip_module(sd):
    # Rimuovi prefisso 'module.' per modelli DataParallel
    new_sd = {}
    for k, v in sd.items():
        new_sd[k[7:]] = v if k.startswith("module.") else v
    return new_sd

def _maybe_load_pretrained(model, path, remap_fn=None, log=lambda *a, **k: None, strict=False):
    if not path or not os.path.exists(path):
        return [], []
    log(f"=> loading pretrained model '{path}'")
    sd, _ = _load_checkpoint_dict(path)
    if remap_fn:
        sd = remap_fn(sd)
    missing, unexpected = model.load_state_dict(sd, strict=strict)
    if missing:
        log(f"Missing keys when loading pretrained: {len(missing)}")
    if unexpected:
        log(f"Unexpected keys when loading pretrained: {len(unexpected)}")
    return missing, unexpected

def _maybe_load_checkpoint(model, path, log=lambda *a, **k: None, strict=False, args=None):
    if not path or not os.path.exists(path):
        return
    log(f"=> loading checkpoint '{path}'")
    sd, ckpt = _load_checkpoint_dict(path)
    incompatible = model.load_state_dict(sd, strict=strict)
    if args is not None:
        if "epoch" in ckpt:
            args.start_epoch = ckpt["epoch"]
        if "best_acc" in ckpt:
            args.best_acc = ckpt["best_acc"]
    loaded_keys = len(model.state_dict().keys()) - len(getattr(incompatible, "missing_keys", []))
    log(f"Loaded {loaded_keys} keys from checkpoint")

def _build_swinunetr_base(args):
    # Costruzione base del backbone SwinUNETR (MONAI)
    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z),
        in_channels=args.in_channels,
        out_channels=args.out_channels,
        feature_size=48,
        use_checkpoint=False,
    )
    return model

def _attach_classification_head_vit(model, args, log):
    # Imposta encoder + classification head secondo il nome modello
    model = SwinVit_3D(model, num_classes=3, num_features=768,dropout_p=args.dropout_rate)
    name = args.model_name.lower()
    if name == "swinvit+ml_decoder":
        try:
            head = MLDecoder(
                num_classes=3,
                initial_num_features=1024,
                num_of_groups=1,
                decoder_embedding=768,
                zsl=0
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using SwinViT with ML-Decoder Classification Head")
        except NameError:
            log("Warning: MLDecoder not imported, using default head", level="warning")
    elif name == "swinvit+noah":
        try:
            head = NOAH(
                inplanes=384,
                outplanes=3,
                dropout=0.1,
                head_num=1,
                head_split=True,
                kv_split=False
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using SwinViT with NOAH Classification Head")
        except NameError:
            log("Warning: NOAH not imported, using default head", level="warning")
    else:
        log("Using SwinViT with Single Linear Classification Head")
        
    return model

def _attach_classification_head(model, args, log):
    # Imposta encoder + classification head secondo il nome modello
    model = SwinUNETREncoder(model, num_classes=3, num_features=768,dropout_p=args.dropout_rate)
    name = args.model_name.lower()
    if name == "swinunetr+ml_decoder":
        try:
            head = MLDecoder(
                num_classes=3,
                initial_num_features=1024,
                num_of_groups=1,
                decoder_embedding=768,
                zsl=0
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using SwinUNETR with ML-Decoder Classification Head")
        except NameError:
            log("Warning: MLDecoder not imported, using default head", level="warning")
    elif name == "swinunetr+noah":
        try:
            head = NOAH(
                inplanes=768,
                outplanes=3,
                dropout=0.1,
                head_num=1,
                head_split=True,
                kv_split=False
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using SwinUNETR with NOAH Classification Head")
        except NameError:
            log("Warning: NOAH not imported, using default head", level="warning")
    else:
        log("Using SwinUNETR with Single Linear Classification Head")
    return model


def _attach_resnet_head(model, args, log):
    name = args.model_name.lower()
    if name.endswith("+ml_decoder"):
        try:
            head = MLDecoder(
                num_classes=3,
                initial_num_features=1024,
                num_of_groups=1,
                decoder_embedding=768,
                zsl=0
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using ResNet with ML-Decoder Classification Head")
        except NameError:
            log("Warning: MLDecoder not imported, using default head", level="warning")
    elif name.endswith("+noah"):
        try:
            head = NOAH(
                inplanes=768,
                outplanes=3,
                dropout=0.1,
                head_num=1,
                head_split=True,
                kv_split=False
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using ResNet with NOAH Classification Head")
        except NameError:
            log("Warning: NOAH not imported, using default head", level="warning")
    else:
        log("Using ResNet with Single Linear Classification Head")
    return model

def _attach_densenet_head(model, args, log):
    name = args.model_name.lower()
    if name.endswith("+ml_decoder"):
        try:
            head = MLDecoder(
                num_classes=3,
                initial_num_features=1024,
                num_of_groups=1,
                decoder_embedding=768,
                zsl=0
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using DenseNet with ML-Decoder Classification Head")
        except NameError:
            log("Warning: MLDecoder not imported, using default head", level="warning")
    elif name.endswith("+noah"):
        try:
            head = NOAH(
                inplanes=768,
                outplanes=3,
                dropout=0.1,
                head_num=1,
                head_split=True,
                kv_split=False
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            log("Using DenseNet with NOAH Classification Head")
        except NameError:
            log("Warning: NOAH not imported, using default head", level="warning")
    else:
        log("Using DenseNet with Single Linear Classification Head")
    return model

def _setup_model(args, logger, log):
    """Setup e caricamento del modello."""
    log("")
    log("Model INFO:")
    log(f"Model architecture: {args.model_name}")

    name = args.model_name.lower()
    pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name) if getattr(args, "pretrained_dir", None) else None

    if "swinunetr" in name:
        # Backbone
        model = _build_swinunetr_base(args)
        # Pretrained (se presente e se non si è passato un checkpoint completo)
        if pretrained_pth and os.path.exists(pretrained_pth) and args.checkpoint_path is None:
            log("Using pretrained weights")
            _maybe_load_pretrained(
                model, pretrained_pth,
                remap_fn=lambda sd: _remap_keys_for_swinunetr(sd, log),
                log=log, strict=False
            )
        else:
            if args.checkpoint_path is None:
                log(f"Warning: pretrained model not found at '{pretrained_pth}'", level="warning")
            else:
                log("Skipping loading pretrained weights since checkpoint path is provided")

        # Encoder dedicato (se fornito) prima della testa
        if getattr(args, "encoder10_pth", None) and os.path.exists(args.encoder10_pth) and args.checkpoint_path is None:
            log(f"=> loading encoder '{args.encoder10_pth}'")
            sd, _ = _load_checkpoint_dict(args.encoder10_pth)
            enc_sd = {k: v for k, v in sd.items() if k.startswith("encoder10.")}
            incompatible = model.load_state_dict(enc_sd, strict=False)
            loaded = len(model.state_dict().keys()) - len(getattr(incompatible, "missing_keys", []))
            log(f"Loaded {loaded} keys from encoder10 checkpoint")

        # Classification head
        model = _attach_classification_head(model, args, log)

        # Checkpoint di fine-tuning (se specificato)
        if getattr(args, "checkpoint_path", None) and os.path.exists(args.checkpoint_path):
            _maybe_load_checkpoint(model, args.checkpoint_path, log=log, strict=False, args=args)
    
    elif "swinvit" in name:
        # Backbone
        model = _build_swinunetr_base(args)
        print("swinvit")
        # Pretrained (se presente e se non si è passato un checkpoint completo)
        if pretrained_pth and os.path.exists(pretrained_pth) and args.checkpoint_path is None:
            log("Using pretrained weights")
            _maybe_load_pretrained(
                model, pretrained_pth,
                remap_fn=lambda sd: _remap_keys_for_swinunetr(sd, log),
                log=log, strict=False
            )
        else:
            if args.checkpoint_path is None:
                log(f"Warning: pretrained model not found at '{pretrained_pth}'", level="warning")
            else:
                log("Skipping loading pretrained weights since checkpoint path is provided")

        # Classification head
        model = _attach_classification_head_vit(model, args, log)

        # Checkpoint di fine-tuning (se specificato)
        if getattr(args, "checkpoint_path", None) and os.path.exists(args.checkpoint_path):
            _maybe_load_checkpoint(model, args.checkpoint_path, log=log, strict=False, args=args)

    elif "densenet" in name:
        net = DenseNet201(spatial_dims=3, in_channels=1, out_channels=1)   
        model = DenseNet_3D(original_model=net, num_classes=3,dropout_p=args.dropout_rate)

    elif "resnet50" in name or "resnet18" in name:
        depth = 50 if "resnet50" in name else 18
        
        # Carica backbone 3D
        backbone = resnet.resnet50 if depth == 50 else resnet.resnet18
        net = backbone(
            sample_input_W=args.roi_x,
            sample_input_H=args.roi_y,
            sample_input_D=args.roi_z,
            num_seg_classes=1
        )

        # Carica pesi pre-addestrati se disponibili
        if pretrained_pth and os.path.exists(pretrained_pth) and args.checkpoint_path is None:
            log("Using pretrained weights")
            log(f"=> loading pretrained model '{pretrained_pth}'")
            sd, _ = _load_checkpoint_dict(pretrained_pth)
            sd = _remap_keys_strip_module(sd)
            missing, unexpected = net.load_state_dict(sd, strict=False)
            if missing:
                log(f"Missing keys: {len(missing)}")
            if unexpected:
                log(f"Unexpected keys: {len(unexpected)}")
        elif args.checkpoint_path is None:
            log(f"Warning: pretrained model not found at '{pretrained_pth}'", level="warning")

        # Wrap con ResNet_3D e head di classificazione
        model = ResNet_3D(net, num_classes=3, dropout_p=args.dropout_rate)
        model = _attach_resnet_head(model, args, log)

        # Carica checkpoint di fine-tuning se specificato
        if getattr(args, "checkpoint_path", None) and os.path.exists(args.checkpoint_path):
            _maybe_load_checkpoint(model, args.checkpoint_path, log=log, strict=False, args=args)

    else:
        raise ValueError(f"Unsupported model architecture: {args.model_name}")

    # Conta parametri
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log(f"Total trainable parameters: {total_params:,}")

    # Sposta su GPU
    model = model.cuda(args.gpu)

    # Distributed: converti BN -> SyncBN prima del DDP, poi avvolgi in DDP
    if getattr(args, "distributed", False):
        if getattr(args, "norm_name", "") == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model, device_ids=[args.gpu], output_device=args.gpu
        )

    return model


def _setup_optimizer(model, args, logger, log):
    """Setup optimizer."""
    log("")
    log("Optimizer INFO:")

    # Istanzia l’optimizer PRIMA, in ogni caso!
    if args.optim_name == "adam":
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.optim_lr,
            weight_decay=args.reg_weight
        )
    elif args.optim_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=args.optim_lr,
            weight_decay=args.reg_weight
        )
    elif args.optim_name == "sgd":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=args.optim_lr,
            momentum=args.momentum,
            nesterov=True,
            weight_decay=args.reg_weight
        )
    else:
        raise ValueError(f"Unsupported optimizer: {args.optim_name}")

    log(f"Using optimizer: {args.optim_name} with lr={args.optim_lr}, weight_decay={args.reg_weight}")

    # Solo DOPO istanziazione puoi caricare lo stato, se serve
    if args.checkpoint_path is not None:
        log(f"Resuming optimizer state from checkpoint: {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if 'optimizer' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])

    return optimizer



def _setup_scheduler(optimizer, args):
    """Setup learning rate scheduler."""

    # 1. Istanzia lo scheduler con i parametri giusti
    if args.lrschedule == "warmup_cosine":
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer,
            warmup_epochs=args.warmup_epochs,
            max_epochs=args.max_epochs,
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=args.max_epochs,
            eta_min=getattr(args, "eta_min", 0.0),
        )
    elif args.lrschedule == "cosine_restarts":
        scheduler = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_T0,
            T_mult=getattr(args, "restart_Tmult", 2),
            eta_min=getattr(args, "eta_min", 0.0),
        )
    elif args.lrschedule == "warmup_cosine_restarts":
        warmup = LinearLR(
            optimizer,
            start_factor=getattr(args, "warmup_start_factor", 0.01),
            end_factor=1.0,
            total_iters=args.warmup_epochs,
        )
        cosine_wr = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=args.restart_T0,
            T_mult=getattr(args, "restart_Tmult", 2),
            eta_min=getattr(args, "eta_min", 0.0),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup, cosine_wr],
            milestones=[args.warmup_epochs],
        )
    else:
        # Nessuna scheduler selezionata
        scheduler = None

    # 2. Se c'è il checkpoint, carica lo stato DOPO la creazione dello scheduler
    if args.checkpoint_path is not None:
        checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
        if 'scheduler' in checkpoint and scheduler is not None:
            scheduler.load_state_dict(checkpoint['scheduler'])

    return scheduler



class BalancedBatchSampler(Sampler):
    """Crea batch con esattamente lo stesso numero di campioni per classe."""
    
    def __init__(self, labels, batch_size, num_classes=3):
        self.labels = np.array(labels)
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.samples_per_class = batch_size // num_classes
        
        # Raggruppa indici per classe
        self.class_indices = [
            np.where(self.labels == c)[0].tolist()
            for c in range(num_classes)
        ]
        
        # Numero di batch = minimo tra le classi
        min_samples = min(len(idx) for idx in self.class_indices)
        self.num_batches = min_samples // self.samples_per_class
    
    def __iter__(self):
        # Shuffle ogni classe
        shuffled = [random.sample(idx, len(idx)) for idx in self.class_indices]
        
        # Crea batch bilanciati
        for i in range(self.num_batches):
            batch = []
            for c in range(self.num_classes):
                start = i * self.samples_per_class
                end = start + self.samples_per_class
                batch.extend(shuffled[c][start:end])
            
            random.shuffle(batch)  # Shuffle interno al batch
            yield batch
    
    def __len__(self):
        return self.num_batches

def _setup_data_test(args, logger, log):
    """Setup dataset di test e dataloader."""
    log("")
    log("Test Dataset INFO:")
    
    # Carica dataset di test
    test_dataset = OrganoidsINRIA3D(args.data_dir + "/test_set", exact_class_dir=args.exact_class, slice_selection=args.slice_selection, n_slices=args.n_slices)
    test_labels = test_dataset.labels
    num_classes = 3
    
    log(f"Test Dataset length: {len(test_dataset)}")
    
    # Class distribution totale nel test set
    test_counts = np.bincount(test_labels, minlength=num_classes)
    log("Class distribution in test dataset:")
    for c, n in enumerate(test_counts):
        log(f"  Class {c}: {n}")
    
    # Crea dataloader di test
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
        num_workers=args.workers,
    )

    if args.debug:
        log("\nDEBUG MODE ACTIVE - Test Set")
        test_samples = 3
        test_loader = DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            pin_memory=True,
            drop_last=False,
            num_workers=args.workers,
            sampler=torch.utils.data.SubsetRandomSampler(
                list(range(min(test_samples, len(test_dataset))))
            )
        )
        log(f"DEBUG: using {min(test_samples, len(test_dataset))} test samples")
    
    log(f"Test loader: {len(test_loader)} batches")
    log("*" * 50)
    
    return test_loader

def _setup_data(args, logger, log):
    """Setup dataset, dataloaders e calcola class weights."""
    log("")
    log("Dataset INFO:")
    
    # Carica dataset
    dataset = OrganoidsINRIA3D(args.data_dir + "/train_set", exact_class_dir=args.exact_class, slice_selection=args.slice_selection, n_slices=args.n_slices)
    labels = dataset.labels
    num_classes = 3
    
    log(f"Dataset length: {len(dataset)}")
    
    # Class distribution totale
    dataset_counts = np.bincount(labels, minlength=num_classes)
    log("Class distribution in entire dataset:")
    for c, n in enumerate(dataset_counts):
        log(f"  Class {c}: {n}")
    
    # Split dataset
    log(f"\nUsing split method: {args.split_method}")
    
    if args.split_method == "random":
        train_set, val_set = split_dataset_random(dataset, val_size=0.2, seed=args.seed)
    elif args.split_method == "stratified":
        train_set, val_set = split_dataset_stratified(dataset, val_size=0.2, seed=args.seed)
    elif args.split_method == "balanced":
        train_set, val_set = split_dataset_balanced(dataset, val_size=0.2, seed=args.seed)
    elif args.split_method == "percentage":
        train_set, val_set = split_dataset_percentage(dataset, val_size=0.2, seed=args.seed)
    else:
        raise ValueError(f"Unsupported split method: {args.split_method}")
    
    log("\nTrain set balance:")
    verify_balance(train_set, labels)
    log("\nVal set balance:")
    verify_balance(val_set, labels)
    # Debug mode
    if args.debug:
        log("\nDEBUG MODE ACTIVE")
        train_samples = args.debug_train_samples if args.debug_train_samples > 0 else 20
        val_samples = args.debug_val_samples if args.debug_val_samples > 0 else 10
        
        if args.split_method == "balanced":
            samples_per_class_train = max(1, train_samples // num_classes)
            samples_per_class_val = max(1, val_samples // num_classes)
            
            train_set = create_balanced_debug_subset(
                train_set, labels, samples_per_class=samples_per_class_train, seed=args.seed
            )
            val_set = create_balanced_debug_subset(
                val_set, labels, samples_per_class=samples_per_class_val, seed=args.seed + 1
            )
            
            log("\nDebug train balance:")
            verify_balance(train_set, labels)
            log("\nDebug val balance:")
            verify_balance(val_set, labels)
        else:
            train_set = create_stratified_debug_subset(train_set, labels, train_samples, seed=args.seed)
            val_set = create_stratified_debug_subset(val_set, labels, val_samples, seed=args.seed + 1)
        
        log(f"DEBUG: using {len(train_set)} train samples, {len(val_set)} val samples")
    
    log(f"Training set length: {len(train_set)}")
    log(f"Validation set length: {len(val_set)}")
    
    # Ottieni indici per calcolare weights
    train_indices = _get_indices(train_set, len(train_set))
    val_indices = _get_indices(val_set, len(val_set))
    train_labels = labels[train_indices]
    
    # Create dataloaders
    # Se similarity_loss è attivo, usa BalancedBatchSampler
    if args.similarity_loss is not None and args.similarity_loss != "":
        # Verifica che batch_size sia multiplo di num_classes
        if args.batch_size % num_classes != 0:
            log(f"\nWARNING: batch_size ({args.batch_size}) non è multiplo di {num_classes}")
            log(f"Con similarity loss, ogni batch avrà {args.batch_size // num_classes} campioni per classe")
        
        log(f"\nUsing BalancedBatchSampler: {args.batch_size // num_classes} samples per class per batch")
        
        batch_sampler = BalancedBatchSampler(
            labels=train_labels,
            batch_size=args.batch_size,
            num_classes=num_classes
        )
        
        train_loader = DataLoader(
            train_set,
            num_workers=args.workers,
            batch_sampler=batch_sampler,  # batch_sampler sostituisce batch_size e shuffle
            pin_memory=True,
        )
        
        log(f"Balanced batches created: {len(batch_sampler)} batches")
    else:
        log("\nUsing standard shuffle for training")
        train_loader = DataLoader(
            train_set,
            batch_size=args.batch_size,
            num_workers=args.workers,
            shuffle=True,
            pin_memory=True,
            drop_last=True,
        )
    
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=True,
        drop_last=False,
    )
    
    log(f"Training loader: {len(train_loader)} batches")
    log(f"Validation loader: {len(val_loader)} batches")
    
    # Calcola distribuzioni finali
    train_counts = np.bincount(train_labels, minlength=num_classes)
    val_counts = np.bincount(labels[val_indices], minlength=num_classes)
    
    log("\nClass distribution in training set:")
    for c, n in enumerate(train_counts):
        pct = (n / len(train_set)) * 100 if len(train_set) > 0 else 0
        log(f"  Class {c}: {n} ({pct:.1f}%)")
    
    log("Class distribution in validation set:")
    for c, n in enumerate(val_counts):
        pct = (n / len(val_set)) * 100 if len(val_set) > 0 else 0
        log(f"  Class {c}: {n} ({pct:.1f}%)")
    
    log("*" * 50)
    
    # Calcola class weights
    class_weights = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=train_labels
    )
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).cuda(args.gpu) * 10
    log(f"Class weights: {weights_tensor.cpu().numpy()}")
    
    return train_loader, val_loader, weights_tensor



def _setup_loss(args, class_weights, log):
    """Setup loss function."""
    log("")
    log(f"Using loss function: {args.loss_name}")
    
    if args.similarity_loss:
        log(f"Using similarity loss: {args.similarity_loss} (weight: {args.similarity_loss_weight})")
    
    if args.loss_name == "FocalLoss":
        return FocalLoss(alpha=class_weights, gamma=2.0)
    elif args.loss_name == "LabelSmoothingLoss":
        return LabelSmoothingLoss(classes=3, smoothing=0.1, weight=class_weights)
    elif args.loss_name == "DiversityLoss":
        base = nn.CrossEntropyLoss(weight=class_weights)
        return DiversityLoss(base, diversity_weight=0.15)
    elif args.loss_name == "CombinedLoss":
        return CombinedLoss(alpha=class_weights, gamma=2.0, diversity_weight=0.15)
    elif args.loss_name == "CenterLoss":
        return CenterLoss(num_classes=3, feat_dim=128)
    elif args.loss_name == "CE":
        return nn.CrossEntropyLoss(weight=class_weights)
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_name}")


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


if __name__ == "__main__":
    main()
