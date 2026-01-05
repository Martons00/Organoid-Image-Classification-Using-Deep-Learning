# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import logging
import time
from pathlib import Path

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from configs import config

import sys
import os
from contextlib import contextmanager


import matplotlib.pyplot as plt
import numpy as np
from models.MedicalNet.models import resnet
from models.ResNet50_3D import ResNet_3D
from torchvision.utils import make_grid

class FullModel(nn.Module):

  def __init__(self, model, sem_loss, bd_loss):
    super(FullModel, self).__init__()
    self.model = model
    self.sem_loss = sem_loss
    self.bd_loss = bd_loss

  def pixel_acc(self, pred, label):
    _, preds = torch.max(pred, dim=1)
    valid = (label >= 0).long()
    acc_sum = torch.sum(valid * (preds == label).long())
    pixel_sum = torch.sum(valid)
    acc = acc_sum.float() / (pixel_sum.float() + 1e-10)
    return acc

  def forward(self, inputs, labels, bd_gt, *args, **kwargs):
    
    outputs = self.model(inputs, *args, **kwargs)
    inputs.cuda()
    labels.cuda()
    bd_gt.cuda()

    h, w = labels.size(1), labels.size(2)
    ph, pw = outputs[0].size(2), outputs[0].size(3)
    if ph != h or pw != w:
        for i in range(len(outputs)):
            outputs[i] = F.interpolate(outputs[i], size=(
                h, w), mode='bilinear', align_corners=config.MODEL.ALIGN_CORNERS)

    acc  = self.pixel_acc(outputs[-2], labels)
    loss_s = self.sem_loss(outputs[:-1], labels)
    loss_b = self.bd_loss(outputs[-1], bd_gt)

    filler = torch.ones_like(labels) * config.TRAIN.IGNORE_LABEL
    try:
        bd_label = torch.where(torch.sigmoid(outputs[-1][:,0,:,:]) > 0.7, labels, filler)
        loss_sb = self.sem_loss([outputs[-2]], bd_label)
    except:
        loss_sb = self.sem_loss([outputs[-2]], labels)
    loss = loss_s + loss_b + loss_sb

    return torch.unsqueeze(loss,0), outputs[:-1], acc, [loss_s, loss_b]

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.initialized = False
        self.val = None
        self.avg = None
        self.sum = None
        self.count = None

    def initialize(self, val, weight):
        self.val = val
        self.avg = val
        self.sum = val * weight
        self.count = weight
        self.initialized = True

    def update(self, val, weight=1):
        if not self.initialized:
            self.initialize(val, weight)
        else:
            self.add(val, weight)

    def add(self, val, weight):
        self.val = val
        self.sum += val * weight
        self.count += weight
        self.avg = self.sum / self.count

    def value(self):
        return self.val

    def average(self):
        return self.avg

def create_logger(cfg, cfg_name, phase="training"):
    # Base: outputs/OrganoidsINRIA/training/<YYYY-MM-DD-HH-MM>
    root_output_dir = Path(cfg.output_dir)  # es. "outputs"
    dataset = cfg.dataset_name
    phase_dir = phase
    time_str = time.strftime("%Y-%m-%d-%H-%M")

    final_output_dir = root_output_dir / dataset / phase_dir / time_str
    final_output_dir.mkdir(parents=True, exist_ok=True)

    # training.log nel final_output_dir
    log_path = final_output_dir / "training.log"

    # Logger dedicato e pulito da handler duplicati
    logger = logging.getLogger(f"{dataset}.{phase_dir}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
    fh = logging.FileHandler(str(log_path))
    fh.setFormatter(fmt)
    #ch = logging.StreamHandler()
    #ch.setFormatter(fmt)

    logger.addHandler(fh)
    #logger.addHandler(ch)

    # TensorBoard: opzionale, coerente con struttura
    tensorboard_log_dir = final_output_dir / "tensorboard"
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Log directory: {final_output_dir}")
    return logger, str(final_output_dir), str(tensorboard_log_dir)

def get_confusion_matrix(label, pred, size, num_class, ignore=-1):
    """
    Calcute the confusion matrix by given label and pred
    """
    output = pred.cpu().numpy().transpose(0, 2, 3, 1)
    seg_pred = np.asarray(np.argmax(output, axis=3), dtype=np.uint8)
    seg_gt = np.asarray(
    label.cpu().numpy()[:, :size[-2], :size[-1]], dtype=int)

    ignore_index = seg_gt != ignore
    seg_gt = seg_gt[ignore_index]
    seg_pred = seg_pred[ignore_index]

    index = (seg_gt * num_class + seg_pred).astype('int32')
    label_count = np.bincount(index)
    confusion_matrix = np.zeros((num_class, num_class))

    for i_label in range(num_class):
        for i_pred in range(num_class):
            cur_index = i_label * num_class + i_pred
            if cur_index < len(label_count):
                confusion_matrix[i_label,
                                 i_pred] = label_count[cur_index]
    return confusion_matrix

def adjust_learning_rate(optimizer, base_lr, max_iters, 
        cur_iters, power=0.9, nbb_mult=10):
    lr = base_lr*((1-float(cur_iters)/max_iters)**(power))
    optimizer.param_groups[0]['lr'] = lr
    if len(optimizer.param_groups) == 2:
        optimizer.param_groups[1]['lr'] = lr * nbb_mult
    return lr

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as fnull:
        old_stdout = sys.stdout
        sys.stdout = fnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            
def denormalize(tensor, mean, std):
  for i in range(len(mean)):
    tensor[i] = tensor[i]*std[i] + mean[i]
  return tensor
            
def visualize_images(image_tensor):
    # Ensure tensor is on CPU and denormalize
    image_tensor = image_tensor.cpu()
    image_tensor = denormalize(image_tensor, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

    # Debug: Print the shape of the tensor
    print(f"Image tensor shape before permute: {image_tensor.shape}")

    # Handle different tensor shapes
    if image_tensor.dim() == 4:  # Batch of images (B, C, H, W)
        # Permute to (B, H, W, C)
        batch_images = image_tensor.permute(0, 2, 3, 1).numpy()  # Reorder dimensions
        for img in batch_images:
            plt.imshow(img)
            plt.show()
    elif image_tensor.dim() == 3:  # Single image (C, H, W)
        # Permute to (H, W, C)
        image = image_tensor.permute(1, 2, 0).numpy()  # Reorder dimensions
        plt.imshow(image)
        plt.show()
    else:
        raise ValueError(f"Unexpected tensor dimensions: {image_tensor.dim()}")

def visualize_segmentation(segmentation_tensor):
    # Sposta il tensor sulla CPU e converti in numpy array
    seg_map = segmentation_tensor.cpu().numpy()
    
    # Definisci una mappa colori per 8 classi (0-7)
    # Ogni colore è in formato RGB
    color_map = {
        0: [0, 0, 0],        # Nero Tutto il resto
        1: [255, 0, 0],      # Rosso Background
        2: [0, 255, 0],      # Verde Building
        3: [0, 0, 255],      # Blu Road 
        4: [255, 255, 0],    # Giallo Water
        5: [255, 0, 255],    # Magenta Barren
        6: [0, 255, 255],    # Ciano Forest
        7: [128, 128, 128]   # Grigio Agricolture
    }
    
    # Crea un'immagine RGB vuota
    height, width = seg_map.shape
    colored_seg = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Assegna i colori in base al valore della classe
    for class_idx in range(8):
        mask = (seg_map == class_idx)
        colored_seg[mask] = color_map[class_idx]
    
    # Visualizza l'immagine
    plt.figure(figsize=(10, 10))
    plt.imshow(colored_seg)
    plt.axis('off')
    plt.show()

def _setup_model_old(args, logger, log):
    """Setup e caricamento del modello."""
    log("")
    log("Model INFO:")
    log(f"Model architecture: {args.model_name}")
    if "swinunetr" in args.model_name.lower():
        # Crea modello base
        model = SwinUNETR(
            img_size=(args.roi_x, args.roi_y, args.roi_z),
            in_channels=args.in_channels,
            out_channels=args.out_channels,
            feature_size=48,
            use_checkpoint=False
        )
        
        # Carica pretrained weights
        pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name)


        if os.path.exists(pretrained_pth) and args.checkpoint_path is None:
            log("Using pretrained weights")
            log(f"=> loading pretrained model '{pretrained_pth}'")
            
            checkpoint = torch.load(pretrained_pth, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            
            # Rinomina chiavi per compatibilità
            new_state_dict = {}
            for k, v in state_dict.items():
                # Prima controlla se il layer deve essere saltato
                if any(skip_layer in k for skip_layer in ['out.conv.conv.weight', 'out.conv.conv.bias']):
                    log(f"Skipping layer {k} due to size mismatch")
                    continue  # Salta questo layer completamente
                
                # Poi processa il nome della chiave
                if k.startswith('module.'):
                    new_key = 'swinViT.' + k[len('module.'):]
                    new_key = new_key.replace('fc', 'linear')
                else:
                    new_key = k
                
                # Aggiungi al nuovo state_dict solo se non è stato saltato
                new_state_dict[new_key] = v

            
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            
            if missing:
                log(f"Missing keys when loading pretrained: {len(missing)}")
            if unexpected:
                log(f"Unexpected keys when loading pretrained: {len(unexpected)}")
        else:
            if args.checkpoint_path is None:
                log(f"Warning: pretrained model not found at '{pretrained_pth}'", level="warning")
            else:
                log("Skipping loading pretrained weights since checkpoint path is provided")
        
        # Carica checkpoint per fine-tuning (se specificato)
        if args.encoder10_pth is not None and os.path.exists(args.encoder10_pth) and args.checkpoint_path is None:
            log(f"=> loading encoder '{args.encoder10_pth}'")
            checkpoint = torch.load(args.encoder10_pth, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            # Filtra solo encoder10 keys
            new_state_dict = {k: v for k, v in state_dict.items() if k.startswith("encoder10.")}
            
            incompatible = model.load_state_dict(new_state_dict, strict=False)
            loaded_keys = len(model.state_dict().keys()) - len(getattr(incompatible, "missing_keys", []))
            log(f"Loaded {loaded_keys} keys from encoder10 checkpoint")
        
        # Converti a encoder + classification head
        model = SwinUNETREncoder(model, num_classes=3, num_features=768,dropout_p=args.dropout_rate)
        
        # Aggiungi classification head custom
        if args.model_name == "swinunetr+ml_decoder":
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
        
        elif args.model_name == "swinunetr+noah":
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

            # Carica checkpoint per fine-tuning (se specificato)
        if args.checkpoint_path is not None and os.path.exists(args.checkpoint_path):
            log(f"=> loading checkpoint '{args.checkpoint_path}'")
            checkpoint = torch.load(args.checkpoint_path, map_location="cpu")
            state_dict = checkpoint.get("state_dict", checkpoint)
            
            incompatible = model.load_state_dict(state_dict, strict=False)

            if "epoch" in checkpoint:
                args.start_epoch = checkpoint["epoch"]  
            if "best_acc" in checkpoint:
                args.best_acc = checkpoint["best_acc"]  

            
            loaded_keys = len(model.state_dict().keys()) - len(getattr(incompatible, "missing_keys", []))
            log(f"Loaded {loaded_keys} keys from checkpoint")
        
        # Count parameters
        total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        log(f"Total trainable parameters: {total_params:,}")
    elif "resnet50" in args.model_name.lower():
        model = resnet.resnet50(
                sample_input_W=args.roi_x,
                sample_input_H=args.roi_y,
                sample_input_D=args.roi_z,
                num_seg_classes=1)
        
        
        # Carica pretrained weights
        pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name)


        if os.path.exists(pretrained_pth) and args.checkpoint_path is None:
            log("Using pretrained weights")
            log(f"=> loading pretrained model '{pretrained_pth}'")
            
            checkpoint = torch.load(pretrained_pth, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            
            # Rinomina chiavi per compatibilità
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith('module.'):
                    new_key = k[7:]
                new_state_dict[new_key] = v

            
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(missing)
            
            if missing:
                log(f"Missing keys when loading pretrained: {len(missing)}")
            if unexpected:
                log(f"Unexpected keys when loading pretrained: {len(unexpected)}")
        else:
            if args.checkpoint_path is None:
                log(f"Warning: pretrained model not found at '{pretrained_pth}'", level="warning")
            else:
                log("Skipping loading pretrained weights since checkpoint path is provided")

        model = ResNet_3D(model, num_classes=3, dropout_p=args.dropout_rate)

                # Aggiungi classification head custom
        if args.model_name == "resnet50+ml_decoder":
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

        elif args.model_name == "resnet50+noah":
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
    elif "resnet18" in args.model_name.lower():
        model = resnet.resnet18(
                sample_input_W=args.roi_x,
                sample_input_H=args.roi_y,
                sample_input_D=args.roi_z,
                num_seg_classes=1)
        
        
        # Carica pretrained weights
        pretrained_pth = os.path.join(args.pretrained_dir, args.pretrained_model_name)


        if os.path.exists(pretrained_pth) and args.checkpoint_path is None:
            log("Using pretrained weights")
            log(f"=> loading pretrained model '{pretrained_pth}'")
            
            checkpoint = torch.load(pretrained_pth, map_location="cpu", weights_only=False)
            state_dict = checkpoint.get("state_dict", checkpoint)
            state_dict = state_dict['model'] if 'model' in state_dict else state_dict
            
            # Rinomina chiavi per compatibilità
            new_state_dict = {}
            for k, v in state_dict.items():
                new_key = k
                if k.startswith('module.'):
                    new_key = k[7:]
                new_state_dict[new_key] = v

            
            missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
            print(missing)
            
            if missing:
                log(f"Missing keys when loading pretrained: {len(missing)}")
            if unexpected:
                log(f"Unexpected keys when loading pretrained: {len(unexpected)}")
        else:
            if args.checkpoint_path is None:
                log(f"Warning: pretrained model not found at '{pretrained_pth}'", level="warning")
            else:
                log("Skipping loading pretrained weights since checkpoint path is provided")

        model = ResNet_3D(model, num_classes=3,dropout_p=args.dropout_rate)

                # Aggiungi classification head custom
        if args.model_name == "resnet18+ml_decoder":
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

        elif args.model_name == "resnet18+noah":
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
    else:
        raise ValueError(f"Unsupported model architecture: {args.model_name}")
    
    # Move to GPU
    model = model.cuda(args.gpu)
    
    # Setup distributed
    if args.distributed:
        if args.norm_name == "batch":
            model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.gpu],
            output_device=args.gpu
        )
    
    return model

def _build_resnet_base(args, depth):
    if depth == 50:
        backbone = resnet.resnet50(
            sample_input_W=args.roi_x, sample_input_H=args.roi_y,
            sample_input_D=args.roi_z, num_seg_classes=1
        )
        wrapped = ResNet_3D(backbone, num_classes=3,dropout_p=args.dropout_rate)
    elif depth == 18:
        backbsone = resnet.resnet18(
            sample_input_W=args.roi_x, sample_input_H=args.roi_y,
            sample_input_D=args.roi_z, num_seg_classes=1
        )
        # Usa wrapper specifico se presente, altrimenti fallback
        if "ResNet18_3D" in globals():
            wrapped = ResNet_3D(backbone, num_classes=3,dropout_p=args.dropout_rate)
        else:
            wrapped = ResNet_3D(backbone, num_classes=3,dropout_p=args.dropout_rate)
    else:
        raise ValueError(f"Unsupported ResNet depth: {depth}")
    return wrapped