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

def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.output_dir)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.dataset_name
    model = cfg.model_name
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    final_output_dir = root_output_dir / dataset / cfg_name

    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)

    time_str = time.strftime('%Y-%m-%d-%H-%M')
    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.logs_dir) / dataset / model / \
            (cfg_name + '_' + time_str)
    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

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



def plot_training_curve(
    values: Sequence[float],
    metric_name: str = "Loss",
    epochs: Sequence[int] = None,
    title: str = None,
    figsize: tuple = (8, 5),
    save_path: str = None,
    max_xticks: int = 20
) -> None:
    """
    Plotta la curva di training di loss o accuracy in funzione delle epoche.

    Args:
        values: lista o array di valori (loss o accuracy) per ogni epoca.
        metric_name: nome del metrica mostrata sull'asse y ("Loss" o "Accuracy").
        epochs: lista o array di numeri di epoca; se None, usa range(len(values)).
        title: titolo del grafico; se None, usa f"{metric_name} vs Epoch".
        figsize: dimensione della figura (width, height).
        save_path: percorso file per salvare il grafico; se None, mostra a schermo.
        max_xticks: numero massimo di tick sull'asse x (default: 10).
    """
    if epochs is None:
        epochs = list(range(1, len(values) + 1))
    if title is None:
        title = f"{metric_name} vs Epoche"

    plt.figure(figsize=figsize)
    plt.plot(epochs, values, marker='o', linestyle='-', color='C0')
    plt.xlabel("Epoca")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.grid(True)
    
    # Gestione intelligente degli xticks
    if len(epochs) <= max_xticks:
        plt.xticks(epochs)
    else:
        # Calcola step per avere circa max_xticks tick
        step = max(1, len(epochs) // max_xticks)
        tick_positions = epochs[::step]
        # Assicurati di includere sempre l'ultima epoca
        if epochs[-1] not in tick_positions:
            tick_positions.append(epochs[-1])
        plt.xticks(tick_positions)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()



def plot_multi_class_training_curve(
    avg_acc: Sequence[float],
    per_class_acc: Sequence[Dict[str, float]],
    epochs: Sequence[int] = None,
    title: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    max_xticks: int = 20
) -> None:
    """
    Plotta le curve di training per accuracy media e per-class in funzione delle epoche.
    
    Args:
        avg_acc: lista o array di valori di accuracy media per ogni epoca.
        per_class_acc: lista di dizionari contenenti le accuracies per classe per ogni epoca.
                      Formato: [{'c0': 0.0, 'c1': 0.0, 'c2': 1.0}, ...]
        epochs: lista o array di numeri di epoca; se None, usa range(len(avg_acc)).
        title: titolo del grafico; se None, usa "Training Accuracy vs Epoche".
        figsize: dimensione della figura (width, height).
        save_path: percorso file per salvare il grafico; se None, mostra a schermo.
        max_xticks: numero massimo di tick sull'asse x (default: 20).
    """
    if epochs is None:
        epochs = list(range(1, len(avg_acc) + 1))
    if title is None:
        title = "Training Accuracy vs Epoche"
    
    # Estrai i nomi delle classi dal primo dizionario
    class_names = list(per_class_acc[0].keys())
    
    # Crea arrays per ogni classe
    class_accuracies = {}
    for class_name in class_names:
        class_accuracies[class_name] = [epoch_acc[class_name] for epoch_acc in per_class_acc]
    
    plt.figure(figsize=figsize)
    
    # Plotta accuracy media
    plt.plot(epochs, avg_acc, marker='o', linestyle='-', linewidth=2, 
             label='Average Accuracy', color='black')
    
    # Plotta accuracies per classe
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']  # Colori diversi per ogni classe
    for i, (class_name, acc_values) in enumerate(class_accuracies.items()):
        color = colors[i % len(colors)]
        plt.plot(epochs, acc_values, marker='s', linestyle='--', 
                 label=f'Class {class_name}', color=color)
    
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gestione intelligente degli xticks
    if len(epochs) <= max_xticks:
        plt.xticks(epochs)
    else:
        # Calcola step per avere circa max_xticks tick
        step = max(1, len(epochs) // max_xticks)
        tick_positions = epochs[::step]
        # Assicurati di includere sempre l'ultima epoca
        if epochs[-1] not in tick_positions:
            tick_positions.append(epochs[-1])
        plt.xticks(tick_positions)
    
    # Imposta i limiti dell'asse y da 0 a 1 per le accuracies
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional

def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def plot_loss_lr(
    loss: Sequence[float],
    lr: Sequence[float],
    epochs: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
    max_xticks: int = 20,
    kind: str = "all",           # "twin", "loss_vs_lr", "delta_vs_lr", "all"
    smooth_alpha: Optional[float] = None,  # 0<alpha<1 per smoothing EMA della loss
) -> None:
    """
    Crea plot informativi a partire da loss per epoca e learning rate per epoca.

    kind:
      - "twin": loss (asse sinistro) + lr (asse destro) vs epoche
      - "loss_vs_lr": loss in funzione del lr (asse x in log)
      - "delta_vs_lr": -Δloss per epoca vs lr
      - "all": pannello 1x3 con tutte le viste
    """
    loss = np.asarray(loss, dtype=float)
    lr = np.asarray(lr, dtype=float)
    assert loss.ndim == 1 and lr.ndim == 1, "loss e lr devono essere 1-D"
    assert len(loss) == len(lr), "loss e lr devono avere la stessa lunghezza"
    n = len(loss)

    if epochs is None:
        epochs = np.arange(1, n + 1)
    else:
        epochs = np.asarray(epochs)
        assert len(epochs) == n, "epochs deve avere la stessa lunghezza di loss/lr"

    # Smoothing opzionale sulla loss (EMA)
    loss_plot = loss.copy()
    if smooth_alpha is not None:
        if not (0 < smooth_alpha < 1):
            raise ValueError("smooth_alpha deve essere in (0,1)")
        loss_plot = _ema(loss_plot, smooth_alpha)

    # Delta loss per epoca
    delta_loss = np.diff(loss, prepend=loss[0])

    def _set_xticks(ax):
        if len(epochs) <= max_xticks:
            ax.set_xticks(epochs)
        else:
            step = max(1, len(epochs) // max_xticks)
            ticks = epochs[::step]
            if epochs[-1] not in ticks:
                ticks = np.append(ticks, epochs[-1])
            ax.set_xticks(ticks)

    if kind == "twin":
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        ax1.plot(epochs, loss_plot, color='C0', marker='o', label='Loss')
        ax2.plot(epochs, lr,        color='C1', marker='s', label='LR')
        ax1.set_xlabel('Epoca'); ax1.set_ylabel('Loss', color='C0')
        ax2.set_ylabel('LR', color='C1')
        _set_xticks(ax1)
        ax1.grid(True, alpha=0.3)
        if title is None: title = 'Loss e LR per epoca'
        ax1.set_title(title)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        fig.tight_layout()

    elif kind == "loss_vs_lr":
        plt.figure(figsize=figsize)
        plt.semilogx(lr, loss_plot, marker='o', color='C0')
        plt.xlabel('Learning Rate (log)')
        plt.ylabel('Loss')
        plt.grid(True, which='both', alpha=0.3)
        if title is None: title = 'Loss vs Learning Rate'
        plt.title(title)
        plt.tight_layout()

    elif kind == "delta_vs_lr":
        plt.figure(figsize=figsize)
        plt.plot(lr, -delta_loss, marker='s', color='C2')
        plt.xlabel('Learning Rate')
        plt.ylabel('-ΔLoss per epoca')
        plt.grid(True, alpha=0.3)
        if title is None: title = '-ΔLoss vs Learning Rate'
        plt.title(title)
        plt.tight_layout()

    elif kind == "all":
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*1.6, figsize[1]))
        # 1) Twin axes
        ax1 = axes[0]; ax1b = ax1.twinx()
        ax1.plot(epochs, loss_plot, color='C0', marker='o', label='Loss')
        ax1b.plot(epochs, lr,        color='C1', marker='s', label='LR')
        ax1.set_xlabel('Epoca'); ax1.set_ylabel('Loss', color='C0')
        ax1b.set_ylabel('LR', color='C1')
        _set_xticks(ax1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Loss & LR (assi gemelli)')
        lines = ax1.get_lines() + ax1b.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=9)
        # 2) Loss vs LR (log x)
        ax2 = axes[1]
        ax2.semilogx(lr, loss_plot, marker='o', color='C0')
        ax2.set_xlabel('Learning Rate (log)'); ax2.set_ylabel('Loss')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_title('Loss vs LR (log x)')
        # 3) -ΔLoss vs LR
        ax3 = axes[2]
        ax3.plot(lr, -delta_loss, marker='s', color='C2')
        ax3.set_xlabel('Learning Rate'); ax3.set_ylabel('-ΔLoss per epoca')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('-ΔLoss vs LR')
        if title is None: title = 'Diagnostica LR e dinamica della Loss'
        fig.suptitle(title, y=1.02)
        fig.tight_layout()

    else:
        raise ValueError("kind deve essere uno tra {'twin','loss_vs_lr','delta_vs_lr','all'}")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
