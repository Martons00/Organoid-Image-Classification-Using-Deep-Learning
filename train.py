# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

# ========== Standard Library ==========
import os
import time
import timeit
import pprint

# ========== Third-party Libraries ==========
import numpy as np
from sklearn.utils.class_weight import compute_class_weight
from tensorboardX import SummaryWriter

# ========== PyTorch Core ==========
import torch
import torch.nn as nn
import torch.optim
import torch.backends.cudnn as cudnn
import torch.nn.parallel
import torch.utils.data.distributed
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import DataLoader

# ========== Metrics ==========
from torchmetrics.classification import MulticlassAccuracy

# ========== MONAI ==========
from monai.networks.nets import SwinUNETR

# ========== Project-Specific ==========
from config import config, parse_args
from utils.utils_old import create_logger
from utils.trainer import run_training
from utils.data_utils import (
    get_loader,
    split_dataset_balanced,
    split_dataset_random,
    split_dataset_stratified,
    create_stratified_debug_subset,
    create_balanced_debug_subset,
    train_test_split,
    verify_balance,
    send_alert,
    build_training_message,
    create_kfold_splits_stratified,
    create_kfold_splits_balanced,
    create_fold_dataloaders,
)
from tools.loss import (
FocalLoss, LabelSmoothingLoss, DiversityLoss, CombinedLoss, CenterLoss
)

import asyncio
from models.ML_Decoder_main.src_files.ml_decoder.ml_decoder import MLDecoder
from models.NOAH_main.modules.noah import NOAH
from dataset import OrganoidsINRIA3D
from test import SwinUNETREncoder
# from datasets.base_dataset import AugmentedDataset
from optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR  # Uncomment if used

def main():
    try:
        args = parse_args()  # Aggiorna automaticamente la variabile globale config
        args.amp = not args.noamp
        
        if args.distributed:
            args.ngpus_per_node = torch.cuda.device_count()
            print("Found total gpus", args.ngpus_per_node)
            args.world_size = args.ngpus_per_node * args.world_size
            mp.spawn(main_worker, nprocs=args.ngpus_per_node, args=(args,))
        else:
            main_worker(gpu=0, args=args)
    except Exception as e:
        print("An exception occurred during training:")
        print(str(e))
        if args.telegram_log:
            message = f"🚨 *ERROR*\nAn exception occurred during training:\n{str(e)}"
            asyncio.run(send_alert(args.oar_id, message, token_file=args.token))
        raise e  # Re-raise the exception for further handling if needed

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
        args, args.logdir, args.model_name)
    
    
    writer_dict = {
        'writer': SummaryWriter(tb_log_dir),
        'train_global_steps': 0,
        'valid_global_steps': 0,
    }

    logger.info(pprint.pformat(vars(args)))
    logger.info("")


    if args.telegram_log:
        message = build_training_message(args)
        asyncio.run(send_alert(args.oar_id, message, token_file=args.token))


    print("Using GPU:", args.gpu)
    logger.info("Using GPU: %d" % (args.gpu))

    inf_size = [args.roi_x, args.roi_y, args.roi_z]

    pretrained_dir = args.pretrained_dir
    model_name = args.pretrained_model_name
    pretrained_pth = os.path.join(pretrained_dir, model_name)

    model = SwinUNETR(
        img_size=(args.roi_x, args.roi_y, args.roi_z), 
        in_channels=args.in_channels, 
        out_channels=args.out_channels, 
        feature_size=48,    
        use_checkpoint=False
    )

    state_dict_model = model.state_dict()

    
    # Caricamento del checkpoint
    checkpoint = torch.load(pretrained_pth, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)



    # Rinomino le chiavi rimuovendo 'module.' e aggiungendo 'swinViT.'
    new_state_dict = {}
    for k, v in state_dict.items():
        # rimuove 'module.' all'inizio e aggiunge 'swinViT.'
        if k.startswith('module.'):
            new_key = 'swinViT.' + k[len('module.'):]
            new_key = new_key.replace('fc', 'linear')
        else:
            new_key = k
        new_state_dict[new_key] = v
    
    # Caricamento flessibile dei pesi
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("")
    logger.info("")
    print("Model INFO:")
    logger.info("Model INFO:")
    print(f"Model architecture: {args.model_name}")
    logger.info(f"Model architecture: {args.model_name}")
    print("Using pretrained weights")
    logger.info("Using pretrained weights")
    print(f"=> loaded pretrained model '{pretrained_pth}'")
    logger.info(f"=> loaded pretrained model '{pretrained_pth}'")
    if missing:
        print(f"Number of missing keys when loading pretrained weights: {len(missing)}")
        logger.info("Number of missing keys when loading pretrained weights: %d", len(missing))
    if unexpected:
        print(f"Number of unexpected keys when loading pretrained weights: {len(unexpected)}")
        logger.info("Number of unexpected keys when loading pretrained weights: %d", len(unexpected))



    best_acc = 0
    start_epoch = 0


    if args.checkpoint is not None:
        checkpoint = torch.load(args.checkpoint, map_location="cpu")  # può essere un dict con "state_dict" [parametri] [web:27]
        state_dict = checkpoint.get("state_dict", checkpoint)  # fallback se il checkpoint è già uno state_dict [web:27]

        new_state_dict = {}
        for k, v in state_dict.items():
            # Caso 1: i pesi sono sotto "backbone.encode10.*" -> rimuovi solo "backbone."
            if k.startswith("encoder10."):
                new_state_dict[k] = v
            # Altri prefissi vengono ignorati

        # Caricamento parziale: ignora mismatch e preserva solo le chiavi compatibili
        incompatible = model.load_state_dict(new_state_dict, strict=False)  # utile per caricare subset di pesi [web:29]
        # Opzionale: logga chiavi mancanti/inattese per debug
        if getattr(incompatible, "missing_keys", None):
            print(f"Caricati: {len(model.state_dict().keys())-len(incompatible.missing_keys)}")  # utile per capire cosa non è stato caricato [web:27]
            logger.info(f"Caricati: {len(model.state_dict().keys())-len(incompatible.missing_keys)}")

        if "epoch" in checkpoint:
            start_epoch = 0 #checkpoint["epoch"]  ripristina lo stato di training se presente [web:27]
        if "best_acc" in checkpoint:
            best_acc = 0 #checkpoint["best_acc"]  ripristina metrica migliore se presente [web:27]

        msg = "=> loaded checkpoint for encoder10 '{}' (epoch {}) (bestacc {})".format(args.checkpoint, start_epoch, best_acc)  # messaggio riepilogo [web:27]
        print(msg)  # stampa su stdout [web:27]
        logger.info(msg)  # log su logger [web:27]


    # Here we have to extract the encoder part from the pretrained model and load it
    # into our model

    model = SwinUNETREncoder(
        model, 
        num_classes=3, 
        num_features=768
    )

    if args.model_name == "swinunetr+ml_decoder":
        # Here we add the classification head
        if MLDecoder:
            head = MLDecoder(
                num_classes=3,
                initial_num_features=1024, 
                num_of_groups=1, 
                decoder_embedding=768, 
                zsl=0
            )
            model.global_pool = torch.nn.Identity()
            model.fc = head
            print("ML-Decoder applicato con successo")
        print("Using SwinUNETR with Multi-Layer Classification Head")
        logger.info("Using SwinUNETR with Multi-Layer Classification Head")
    elif args.model_name == "swinunetr+noah":
        # Here we add the classification head
        if NOAH:
            head = NOAH(inplanes=768, outplanes=3, dropout=0.0, head_num=1, head_split=True, kv_split=False)
            model.global_pool = torch.nn.Identity()
            model.fc = head
            print("NOAH applicato con successo")
            print("Using SwinUNETR with NOAH Classification Head")
            logger.info("Using SwinUNETR with NOAH Classification Head")
    else:
        print("Using SwinUNETR with Single Linear Classification Head")
        logger.info("Using SwinUNETR with Single Linear Classification Head")
        # The classification head is already added in the SwinUNETREncoder class


    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Total parameters count", pytorch_total_params)
    logger.info("Total parameters count: %d", pytorch_total_params)

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
        scheduler = LinearWarmupCosineAnnealingLR(
            optimizer, warmup_epochs=args.warmup_epochs, max_epochs=args.max_epochs
        )
    elif args.lrschedule == "cosine_anneal":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.max_epochs)
    else:
        scheduler = None

    
    # Here we prepare the data loader
    dataset = OrganoidsINRIA3D(args.data_dir, exact_class_dir=args.exact_class)
    print("")
    logger.info("")
    print("Dataset INFO:")
    logger.info("Dataset INFO:")
    print("Dataset length is:", len(dataset))
    logger.info(f"Dataset length is: {len(dataset)}")
    dataset_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=args.batch_size,
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


    # Eventuale sottoinsieme per il debug con split stratificato
    print(f"\nUsing split method: {args.split_method}")
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
    elif args.split_method == "balanced":
        # Split bilanciato - stesso numero di campioni per classe
        train_set, val_set = split_dataset_balanced(dataset, val_size=0.2, seed=42)
        print("\nTrain set balance:")
        verify_balance(train_set, dataset.labels)
        print("\nVal set balance:")
        verify_balance(val_set, dataset.labels)
    else:
        raise ValueError(f"Unsupported split method: {args.split_method}")
    
    if args.debug:
        # Impostazioni debug
        print("\nDEBUG MODE ACTIVE")
        DEBUG_TRAIN_SAMPLES = args.debug_train_samples if args.debug_train_samples > 0 else 20
        DEBUG_VAL_SAMPLES = args.debug_val_samples if args.debug_val_samples > 0 else 10
        if args.split_method == "balanced":
            # Debug subset bilanciato - es. 10 campioni per classe per train, 3 per val
            train_set = create_balanced_debug_subset(train_set, dataset.labels, samples_per_class=int(DEBUG_TRAIN_SAMPLES/3), seed=42)
            val_set = create_balanced_debug_subset(val_set, dataset.labels, samples_per_class=int(DEBUG_VAL_SAMPLES/3), seed=43)
            
            print("\nDebug train balance:")
            verify_balance(train_set, dataset.labels)
            print("\nDebug val balance:")
            verify_balance(val_set, dataset.labels)
        else:   
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
    print("")

    
    
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
    logger.info("" + "*" * 50)
    print("*" * 50)
    logger.info("")


    print("\nTraining setting summary:")
    logger.info("Training setting summary:")
    print(f"Using optimizer: {args.optim_name} with lr={args.optim_lr}, weight decay={args.reg_weight}")
    logger.info(f"Using optimizer: {args.optim_name} with lr={args.optim_lr}, weight decay={args.reg_weight}")
    if scheduler is not None:
        print(f"Using LR scheduler: {args.lrschedule}")
        logger.info(f"Using LR scheduler: {args.lrschedule}")
    class_weights = compute_class_weight(class_weight='balanced',classes=np.unique(labels),y=labels[train_indices])
    weights = torch.tensor(class_weights, dtype=torch.float) * 10
    print("Class weights:", weights.numpy())
    logger.info("Class weights: " + str(weights.numpy()))

    print(f"Using loss function: {args.loss_name}")
    logger.info(f"Using loss function: {args.loss_name}")
    if args.loss_name == "FocalLoss":
        loss_func = FocalLoss(alpha=weights.cuda(args.gpu), gamma=2.0)
    elif args.loss_name == "LabelSmoothingLoss":
        loss_func = LabelSmoothingLoss(classes=3, smoothing=0.1, weight=weights.cuda(args.gpu))
    elif args.loss_name == "DiversityLoss":
        base = nn.CrossEntropyLoss(weight=weights.cuda(args.gpu))
        loss_func = DiversityLoss(base, diversity_weight=0.15)
    elif args.loss_name == "CombinedLoss":
        loss_func = CombinedLoss(
            alpha=weights.cuda(args.gpu), 
            gamma=2.0, 
            diversity_weight=0.15
        )
    elif args.loss_name == "CenterLoss":
        loss_func = CenterLoss(num_classes=3, feat_dim=128)
    elif args.loss_name == "CE":
        loss_func = nn.CrossEntropyLoss(weight=weights.cuda(args.gpu))
    else:
        raise ValueError(f"Unsupported loss function: {args.loss_name}")

    acc_metric = MulticlassAccuracy(num_classes=3, average='macro').cuda(args.gpu)

    start = timeit.default_timer()
    print("")
    logger.info("")
    print("*" * 50)
    logger.info("*" * 50)
    print("Starting training...")
    logger.info("Starting training...")




    accuracy = run_training(
        model=model,
        train_loader=train_loader,
        val_loader=validation_loader,
        optimizer=optimizer,
        loss_func=loss_func,
        acc_func=acc_metric,
        args=args,
        scheduler=scheduler,
        start_epoch=start_epoch,
        writer_dict=writer_dict,
        final_output_dir = final_output_dir,
        logger=logger,
    )

    end = timeit.default_timer()
    print("Total time spent:", end - start)
    logger.info("Total time spent: %d", end - start)
    writer_dict['writer'].close()
    torch.cuda.empty_cache()

    return 0


if __name__ == "__main__":
    main()
