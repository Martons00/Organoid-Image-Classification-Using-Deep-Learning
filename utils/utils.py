# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import torch

# ------------------------------------------------------------------------------
# Modified based on https://github.com/HRNet/HRNet-Semantic-Segmentation
# ------------------------------------------------------------------------------


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

from typing import Sequence, Dict, List, Union
import matplotlib.pyplot as plt
import numpy as np

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


def dice(x, y):
    intersect = np.sum(np.sum(np.sum(x * y)))
    y_sum = np.sum(np.sum(np.sum(y)))
    if y_sum == 0:
        return 0.0
    x_sum = np.sum(np.sum(np.sum(x)))
    return 2 * intersect / (x_sum + y_sum)


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = np.where(self.count > 0, self.sum / self.count, self.sum)


def distributed_all_gather(
    tensor_list, valid_batch_size=None, out_numpy=False, world_size=None, no_barrier=False, is_valid=None
):
    if world_size is None:
        world_size = torch.distributed.get_world_size()
    if valid_batch_size is not None:
        valid_batch_size = min(valid_batch_size, world_size)
    elif is_valid is not None:
        is_valid = torch.tensor(bool(is_valid), dtype=torch.bool, device=tensor_list[0].device)
    if not no_barrier:
        torch.distributed.barrier()
    tensor_list_out = []
    with torch.no_grad():
        if is_valid is not None:
            is_valid_list = [torch.zeros_like(is_valid) for _ in range(world_size)]
            torch.distributed.all_gather(is_valid_list, is_valid)
            is_valid = [x.item() for x in is_valid_list]
        for tensor in tensor_list:
            gather_list = [torch.zeros_like(tensor) for _ in range(world_size)]
            torch.distributed.all_gather(gather_list, tensor)
            if valid_batch_size is not None:
                gather_list = gather_list[:valid_batch_size]
            elif is_valid is not None:
                gather_list = [g for g, v in zip(gather_list, is_valid_list) if v]
            if out_numpy:
                gather_list = [t.cpu().numpy() for t in gather_list]
            tensor_list_out.append(gather_list)
    return tensor_list_out


# feats: [N, 768, fD, fH, fW] in ordine (z major -> y -> x) come nell’estrazione
def tile_feature_patches(feats: torch.Tensor, coords) -> torch.Tensor:
    # feats: [N, C, fD, fH, fW] dove N = nZ*nY*nX patch in ordine z->y->x
        # Calcola la griglia dalle coordinate
    unique_coords = {}
    for i, (b, z, y, x) in enumerate(coords):
        if b not in unique_coords:
            unique_coords[b] = {'z': set(), 'y': set(), 'x': set()}
        unique_coords[b]['z'].add(z)
        unique_coords[b]['y'].add(y)  
        unique_coords[b]['x'].add(x)
    
    # Assumendo batch=0
    nZ = len(unique_coords[0]['z'])
    nY = len(unique_coords[0]['y']) 
    nX = len(unique_coords[0]['x'])
    print(f"Griglia ricomposta: nZ={nZ}, nY={nY}, nX={nX}")

    N, C, fD, fH, fW = feats.shape
    assert C == 768, f"Attesi 768 canali, trovato {C}"
    assert N == nZ * nY * nX, f"N non combacia con griglia: N={N} vs {nZ*nY*nX}"
    
    # [N, C, fD, fH, fW] -> [nZ, nY, nX, C, fD, fH, fW]
    g = feats.reshape(nZ, nY, nX, C, fD, fH, fW)
    
    # Riordina: [nZ, nY, nX, C, fD, fH, fW] -> [C, nZ, fD, nY, fH, nX, fW]  
    g = g.permute(3, 0, 4, 1, 5, 2, 6).contiguous()
    
    # Ricomponi le dimensioni spaziali: [C, nZ*fD, nY*fH, nX*fW]
    vol = g.reshape(C, nZ*fD, nY*fH, nX*fW)
    
    # Aggiungi dimensione batch: [1, C, nZ*fD, nY*fH, nX*fW]
    return vol.unsqueeze(0)


def ensure_single_channel(x, mode="first"):
    # x: [B,C,D,H,W] oppure [B,D,H,W]
    if x.dim() == 4:
        x = x.unsqueeze(1)  # -> [B,1,D,H,W]
    if x.shape[1] != 1:
        if mode == "first":
            x = x[:, :1]  # usa il primo canale/slice
        elif mode == "mean":
            x = x.mean(dim=1, keepdim=True)  # media sui canali
        else:
            raise ValueError("mode deve essere 'first' o 'mean'")
    return x

def _starts(size, patch, step):
    if size <= patch:
        return [0]
    s = list(range(0, size - patch + 1, step))
    if s[-1] != size - patch:
        s.append(size - patch)
    return s

def extract_patches_5d_torch(x, patch_size=(128,256,256), step=(128,256,256), pad_value=0):
    # x: [B,1,D,H,W], ritorna patches: [N,1,pd,ph,pw] e coords: [(b,z,y,x0), ...]
    B, C, D, H, W = x.shape
    pd, ph, pw = patch_size
    sd, sh, sw = step
    zs, ys, xs = _starts(D, pd, sd), _starts(H, ph, sh), _starts(W, pw, sw)

    patches = []
    coords  = []
    for b in range(B):
        for z in zs:
            for y in ys:
                for x0 in xs:
                    patch = x[b:b+1, :, z:z+pd, y:y+ph, x0:x0+pw]  # [1,1,d',h',w']
                    dd, hh, ww = patch.shape[-3:]
                    if (dd, hh, ww) != (pd, ph, pw):
                        # pad solo a destra su D,H,W: (wL,wR,hL,hR,dL,dR)
                        pad_d = pd - dd
                        pad_h = ph - hh
                        pad_w = pw - ww
                        patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), value=pad_value)
                    patches.append(patch)   # [1,1,pd,ph,pw]
                    coords.append((b, z, y, x0))
    if not patches:
        return torch.empty(0, 1, *patch_size), []
    patches = torch.cat(patches, dim=0)  # [N,1,pd,ph,pw]
    return patches, coords


# feats: [N, 768, fD, fH, fW] in ordine (z major -> y -> x) come nell’estrazione
def tile_feature_patches(feats: torch.Tensor, coords) -> torch.Tensor:
    # feats: [N, C, fD, fH, fW] dove N = nZ*nY*nX patch in ordine z->y->x
        # Calcola la griglia dalle coordinate
    unique_coords = {}
    for i, (b, z, y, x) in enumerate(coords):
        if b not in unique_coords:
            unique_coords[b] = {'z': set(), 'y': set(), 'x': set()}
        unique_coords[b]['z'].add(z)
        unique_coords[b]['y'].add(y)  
        unique_coords[b]['x'].add(x)
    
    # Assumendo batch=0
    nZ = len(unique_coords[0]['z'])
    nY = len(unique_coords[0]['y']) 
    nX = len(unique_coords[0]['x'])

    N, C, fD, fH, fW = feats.shape
    assert C == 768, f"Attesi 768 canali, trovato {C}"
    assert N == nZ * nY * nX, f"N non combacia con griglia: N={N} vs {nZ*nY*nX}"
    
    # [N, C, fD, fH, fW] -> [nZ, nY, nX, C, fD, fH, fW]
    g = feats.reshape(nZ, nY, nX, C, fD, fH, fW)
    
    # Riordina: [nZ, nY, nX, C, fD, fH, fW] -> [C, nZ, fD, nY, fH, nX, fW]  
    g = g.permute(3, 0, 4, 1, 5, 2, 6).contiguous()
    
    # Ricomponi le dimensioni spaziali: [C, nZ*fD, nY*fH, nX*fW]
    vol = g.reshape(C, nZ*fD, nY*fH, nX*fW)
    
    # Aggiungi dimensione batch: [1, C, nZ*fD, nY*fH, nX*fW]
    return vol.unsqueeze(0)


def ensure_single_channel(x, mode="first"):
    # x: [B,C,D,H,W] oppure [B,D,H,W]
    if x.dim() == 4:
        x = x.unsqueeze(1)  # -> [B,1,D,H,W]
    if x.shape[1] != 1:
        if mode == "first":
            x = x[:, :1]  # usa il primo canale/slice
        elif mode == "mean":
            x = x.mean(dim=1, keepdim=True)  # media sui canali
        else:
            raise ValueError("mode deve essere 'first' o 'mean'")
    return x

def _starts(size, patch, step):
    if size <= patch:
        return [0]
    s = list(range(0, size - patch + 1, step))
    if s[-1] != size - patch:
        s.append(size - patch)
    return s

import math
import torch
import torch.nn.functional as F

def extract_patches_5d_torch(
    x,
    patch_size=(128, 256, 256),
    step=(128, 256, 256),
    pad_value=0,
    max_D=128,
    D_mode="uniform"  # "uniform" | "center" | "first" | "interpolate"
):
    """
    x: [B, C, D, H, W] -> patches: [N, C, pd, ph, pw], coords: [(b, z, y, x0), ...]
    """
    assert x.dim() == 5, "atteso tensor 5D [B,C,D,H,W]"
    B, C, D, H, W = x.shape
    pd, ph, pw = patch_size
    sd, sh, sw = step

    # 2) Generazione degli indici di start (si assume esista _starts)
    zs = _starts(D, pd, sd)
    ys = _starts(H, ph, sh)
    xs = _starts(W, pw, sw)

    patches = []
    coords = []
    for b in range(B):
        for z in zs:
            for y in ys:
                for x0 in xs:
                    patch = x[b:b+1, :, z:z+pd, y:y+ph, x0:x0+pw]  # [1,C,d',h',w']
                    dd, hh, ww = patch.shape[-3:]
                    if (dd, hh, ww) != (pd, ph, pw):
                        # pad solo a destra su D,H,W: (wL,wR, hL,hR, dL,dR)
                        pad_d = pd - dd
                        pad_h = ph - hh
                        pad_w = pw - ww
                        patch = F.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), value=pad_value)
                    patches.append(patch)  # [1,C,pd,ph,pw]
                    coords.append((b, z, y, x0))

    if not patches:
        return torch.empty(0, C, *patch_size, device=x.device, dtype=x.dtype), []
    patches = torch.cat(patches, dim=0)  # [N,C,pd,ph,pw]
    return patches, coords

def tile_with_gaussian_blending(feats, coords, patch_size, step):
    """
    Merge delle feature patches con blending gaussiano.
    Opera direttamente nello spazio delle features senza rescaling.
    
    Args:
        feats: [N, C, fD, fH, fW] features delle N patch
        coords: lista di coordinate [(b, z, y, x), ...] in input space
        patch_size: (pD, pH, pW) dimensioni della patch in input space
        step: (sD, sH, sW) step tra patch in input space
    
    Returns:
        torch.Tensor: [1, C, out_fD, out_fH, out_fW] volume features merged
    """
    from monai.inferers.utils import compute_importance_map
    
    N, C, fD, fH, fW = feats.shape
    pD, pH, pW = patch_size
    sD, sH, sW = step
    
    # Calcola la griglia delle patch dalle coordinate
    unique_z = sorted(set(c[1] for c in coords))
    unique_y = sorted(set(c[2] for c in coords))
    unique_x = sorted(set(c[3] for c in coords))
    
    # Calcola le dimensioni output in feature space
    # Numero di patch in ogni dimensione
    nZ = len(unique_z)
    nY = len(unique_y)
    nX = len(unique_x)
    
    # Dimensione totale considerando overlap
    # Ultima patch: posizione + dimensione patch
    max_z = unique_z[-1] + pD
    max_y = unique_y[-1] + pH
    max_x = unique_x[-1] + pW
    
    # Scale factor da input space a feature space (downsampling 32x)
    scale_d = fD / pD  # es. 4 / 128 = 0.03125
    scale_h = fH / pH
    scale_w = fW / pW
    
    # Dimensioni output in feature space
    out_D = int(max_z * scale_d)
    out_H = int(max_y * scale_h)
    out_W = int(max_x * scale_w)
    
    # Step size in feature space
    step_d_feat = int(sD * scale_d)
    step_h_feat = int(sH * scale_h)
    step_w_feat = int(sW * scale_w)
    
    # Alloca output e count map
    output = torch.zeros(1, C, out_D, out_H, out_W, 
                        device=feats.device, dtype=feats.dtype)
    count = torch.zeros(1, 1, out_D, out_H, out_W, 
                       device=feats.device, dtype=feats.dtype)
    
    # Crea importance map gaussiana
    try:
        importance = compute_importance_map(
            (fD, fH, fW),
            mode="gaussian",
            sigma_scale=0.125,
            device=feats.device,
            dtype=feats.dtype
        )
        # Reshape per broadcasting: [1, 1, fD, fH, fW]
        if importance.ndim == 3:
            importance = importance.unsqueeze(0).unsqueeze(0)
    except:
        # Fallback a constant se gaussian fallisce
        importance = torch.ones(1, 1, fD, fH, fW, 
                               device=feats.device, dtype=feats.dtype)
    
    # Riempi output con weighted accumulation
    for i, (b, z, y, x) in enumerate(coords):
        # Converti coordinate da input space a feature space
        z_f = int(z * scale_d)
        y_f = int(y * scale_h)
        x_f = int(x * scale_w)
        
        # Calcola end coordinates (clamp ai bordi)
        z_end = min(z_f + fD, out_D)
        y_end = min(y_f + fH, out_H)
        x_end = min(x_f + fW, out_W)
        
        # Dimensioni effettive da copiare (gestisce bordi)
        actual_d = z_end - z_f
        actual_h = y_end - y_f
        actual_w = x_end - x_f
        
        # Slice della feature e importance map
        feat_slice = feats[i:i+1, :, :actual_d, :actual_h, :actual_w]
        imp_slice = importance[:, :, :actual_d, :actual_h, :actual_w]
        
        # Accumula con peso
        output[:, :, z_f:z_end, y_f:y_end, x_f:x_end] += feat_slice * imp_slice
        count[:, :, z_f:z_end, y_f:y_end, x_f:x_end] += imp_slice
    
    # Normalizza dividendo per count (evita divisione per zero)
    output = output / count.clamp(min=1e-8)
    
    return output

