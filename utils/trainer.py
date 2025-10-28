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

import asyncio
import os
import shutil
import time
from tracemalloc import start
import torch.nn.functional as F

from sklearn.metrics import confusion_matrix

from torch.utils.data import DataLoader
import numpy as np
import torch
import torch.nn as nn
from .utils import AverageMeter, distributed_all_gather
from .utils import extract_patches_5d_torch, ensure_single_channel, tile_feature_patches,tile_with_gaussian_blending
from tools.plots import plot_training_curve,plot_multi_class_training_curve,plot_loss_lr
from tools.confusion_matrix import plot_confusion_matrix,metrics_from_confusion_matrix, format_print_metrics,plot_metrics_table
from .data_utils import send_alert
from optimizers.early_stop import EarlyStopping  # Uncomment if used
from tools.similarity import compute_similarity_matrix, plot_similarity_heatmap, plot_similarity_heatmap_new
from tools.loss import similarity_margin_loss, supervised_contrastive_from_similarity
from dataset import get_train_transforms,selective_augmentation

def freeze_backbone_and_select_head_fixed_plus(model):
    """Freezing corretto - chiama SOLO UNA VOLTA all'inizio del training"""
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'global_pool' in name or 'fc' in name or 'head' in name or "encoder10" in name or "swinViT.layers4.0.blocks.1." in name or "swinViT.layers4.0.downsample." in name  :
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"✓ Unfrozen: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
            #print(f"✗ Frozen: {name} ({param.numel()} params)")
            frozen_params += param.numel()
    
    print(f"Total frozen: {frozen_params}, trainable: {trainable_params}")
    return model

def freeze_backbone_and_select_head_fixed(model):
    """Freezing corretto - chiama SOLO UNA VOLTA all'inizio del training"""
    frozen_params = 0
    trainable_params = 0
    
    for name, param in model.named_parameters():
        if 'global_pool' in name or 'fc' in name or 'head' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            print(f"✓ Unfrozen: {name} ({param.numel()} params)")
        else:
            param.requires_grad = False
            #print(f"✗ Frozen : {name} ({param.numel()} params)")
            frozen_params += param.numel()
    
    print(f"Total frozen: {frozen_params}, trainable: {trainable_params}")
    return model

def train_epoch_new(model, loader, optimizer, epoch, loss_func, acc_func, args):
    """
    Training con pipeline di inferenza a patch usando forward_features.
    Fine-tuning di global_pool e fc con backbone congelato.
    """
    model.train()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()

    if args.augmentation:
        train_transform = get_train_transforms()
    else:
        train_transform = None
    
    # Contatori per classe
    num_classes = None
    per_class_correct = None
    per_class_total = None
    
    # Liste per confusion matrix
    all_preds = []
    all_targets = []
    
    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    
    for idx, batch_data in enumerate(loader):
        # Estrai data e target
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]
        
                # Calcola similarity matrices solo per epoche selezionate
        should_compute_sim = (
            (epoch == 0 or epoch == (args.max_epochs - 1) or epoch == int(args.max_epochs * 0.5)) 
            and idx == 0 
            and args.rank == 0
        )
        
        # ============================================
        # AUGMENTATION qui, on-the-fly
        # ============================================
        if train_transform is not None:
            # Augmenta solo il 50% dei samples nel batch
            data = selective_augmentation(
                data, 
                train_transform,
                augmentation_ratio=0.5  # ← 50% originali, 50% augmentati
            )

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Assicura single channel per tutto il batch
        data = ensure_single_channel(data, mode="first")  # [B,1,D,H,W]
        B = data.shape[0]
        
        # ============================================
        # STEP 1: Estrai TUTTE le patch di TUTTO il batch
        # ============================================
        all_patches = []
        all_coords = []
        batch_indices = []  # Tiene traccia di quale sample del batch appartiene ogni patch
        patches_per_sample = []  # Numero di patch per ogni sample
        
        for b in range(B):
            vol = data[b:b+1]  # [1,1,D,H,W]
            
            # Estrai patch con overlap per smooth blending
            patches, coords = extract_patches_5d_torch(
                vol,
                patch_size=(args.roi_z, args.roi_y, args.roi_x),
                step=args.step,
                pad_value=0
            )
            
            all_patches.append(patches)
            all_coords.extend(coords)
            batch_indices.extend([b] * patches.shape[0])
            patches_per_sample.append(patches.shape[0])
        
        # Concatena tutte le patch in un mega-batch
        all_patches = torch.cat(all_patches, dim=0).to(torch.float32)  # [Total_N, 1, D, H, W]
        total_patches = all_patches.shape[0]
        
        # ============================================
        # STEP 2: Batch processing delle patch
        # ============================================
        sw_batch_size = args.sw_batch_size if hasattr(args, 'sw_batch_size') else 4
        
        all_feats = []
        all_hidden = []
        
        # Processa le patch in batch invece che una alla volta
        for i in range(0, total_patches, sw_batch_size):
            end_idx = min(i + sw_batch_size, total_patches)
            batch_patches = all_patches[i:end_idx]  # [sw_batch_size, 1, D, H, W]
            
            # Forward pass su batch di patch
            feats, hidden = model.forward_features(batch_patches)
            all_feats.append(feats)
            all_hidden.append(hidden)
        
        # Concatena tutti i risultati
        all_feats = torch.cat(all_feats, dim=0)  # [Total_N, Cf, fD, fH, fW]
        all_hidden = torch.cat(all_hidden, dim=0)  # [Total_N, Ch, hD, hH, hW]
        
        # ============================================
        # STEP 3: Ricostruisci per ogni sample del batch con blending
        # ============================================
        batch_logits = []
        feat_list_all = []
        hidden_list_all = []
        
        start_idx = 0
        for b in range(B):
            num_patches = patches_per_sample[b]
            end_idx = start_idx + num_patches
            
            # Estrai features di questo sample
            b_feats = all_feats[start_idx:end_idx]  # [num_patches, Cf, fD, fH, fW]
            b_hidden = all_hidden[start_idx:end_idx]  # [num_patches, Ch, hD, hH, hW]
            b_coords = all_coords[start_idx:end_idx]
            
            # Ricostruisci con blending (invece di tiling rigido)
            feats_tiled = tile_with_gaussian_blending(
                b_feats,
                b_coords,
                patch_size=(args.roi_z, args.roi_y, args.roi_x),
                step=args.step
            )
            
            
            feat_list_all.append(feats_tiled)

            if should_compute_sim:
                hidden_tiled = tile_with_gaussian_blending(
                    b_hidden,
                    b_coords,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=args.step
                    )

                hidden_list_all.append(hidden_tiled)
            
            # Classificazione: global_pool → flatten → fc
            pooled = model.global_pool(feats_tiled)
            
            if args.model_name == "swinunetr+ml_decoder":
                pooled = pooled.flatten(2)
            elif args.model_name == "swinunetr":
                pooled = pooled.flatten(1)
            
            logits_b = model.fc(pooled)  # [1, num_classes]
            batch_logits.append(logits_b)
            
            start_idx = end_idx

        

        
        sim = None
        if should_compute_sim or args.similarity_loss in ["contrastive", "margin"]:
            feat_concat = torch.cat(feat_list_all, dim=0)  # [B,Cf,D,H,W]
            feat_flat = feat_concat.view(feat_concat.shape[0], -1)  # [B,Cf*D*H*W]
            sim = compute_similarity_matrix(feat_flat)
            
            if should_compute_sim:
                sim_np = sim.detach().float().cpu().numpy()
                plot_similarity_heatmap_new(
                    sim_np, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_epoch{epoch}_iter{idx}.png")
                )
                

                
                hidden_concat = torch.cat(hidden_list_all, dim=0)
                hidden_flat = hidden_concat.view(hidden_concat.shape[0], -1)
                sim_hidden = compute_similarity_matrix(hidden_flat).cpu().detach().numpy()
                plot_similarity_heatmap_new(
                    sim_hidden, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_hidden_epoch{epoch}_iter{idx}.png")
                )
        
        # Calcola loss
        logits = torch.cat(batch_logits, dim=0)  # [B,num_classes]
        loss = loss_func(logits, target)
        
        # Inizializza contatori per classe alla prima iterazione
        if num_classes is None:
            num_classes = logits.shape[1]
            per_class_correct = np.zeros(num_classes, dtype=np.int64)
            per_class_total = np.zeros(num_classes, dtype=np.int64)
        
        # Calcola metriche di accuracy
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # [B]
            target_eval = target.view(-1) if target.ndim > 1 else target
            
            all_preds.append(preds.cpu())
            all_targets.append(target_eval.cpu())
            
            # Accuracy batch
            correct = (preds == target_eval).sum().item()
            not_nans = target_eval.numel()
            
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)
            
            # Per-class accuracy
            t_cpu = target_eval.cpu().numpy()
            p_cpu = preds.cpu().numpy()
            
            # Calcola correttezza per ogni sample
            mask = (p_cpu == t_cpu)
            
            # Conta totali e corretti per classe
            batch_total = np.bincount(t_cpu, minlength=num_classes)
            batch_correct = np.bincount(t_cpu[mask], minlength=num_classes)
            
            per_class_correct += batch_correct
            per_class_total += batch_total
        
        sim_loss_value = 0.0
        if args.similarity_loss == "contrastive" and sim is not None:
            loss_sim = supervised_contrastive_from_similarity(sim, target, temperature=0.07)
            loss = loss + args.similarity_loss_weight * loss_sim
            sim_loss_value = loss_sim.item()
        elif args.similarity_loss == "margin" and sim is not None:
            loss_sim = similarity_margin_loss(sim, target, pos_margin=0.5, neg_margin=0.0)
            loss = loss + args.similarity_loss_weight * loss_sim
            sim_loss_value = loss_sim.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Aggiorna metriche
        if is_distributed:
            loss_list = distributed_all_gather(
                [loss], 
                out_numpy=True, 
                is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size
            )
            
            # Aggregazione accuracy globale
            acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
            n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
            
            acc_list, not_nans_list = distributed_all_gather(
                [acc_tensor, n_tensor],
                out_numpy=True,
                is_valid=idx < loader.sampler.valid_length
            )
            
            for al, nl in zip(acc_list, not_nans_list):
                run_acc.update(float(al), n=int(nl))
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_acc.update(acc, n=not_nans)
        
        # Logging
        if is_main_process:
            if sim is not None:
                print(
                    f"Epoch: {epoch}/{args.max_epochs} Iter: {idx}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"sim_loss: {sim_loss_value:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
            else:
                print(
                    f"Epoch: {epoch}/{args.max_epochs} Iter: {idx}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
        start_time = time.time()
    
    # Calcola accuracy per classe
    per_class_acc = {
        int(c): float(per_class_correct[c]) / max(1, int(per_class_total[c]))
        for c in range(num_classes)
    }
    
    # Confusion matrix
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    # Stampa riassunto
    if is_main_process:
        summary = ", ".join([f"c{c}: {per_class_acc[c]:.3f}" for c in range(num_classes)])
        print(f"[Train epoch {epoch}] avg_loss={run_loss.avg:.4f} avg_acc={run_acc.avg:.4f} | per-class [{summary}]")
    
    return run_loss.avg, float(run_acc.avg), per_class_acc, cm

def train_epoch(model, loader, optimizer, epoch, loss_func, acc_func, args):
    """
    Training con pipeline di inferenza a patch usando forward_features.
    Fine-tuning di global_pool e fc con backbone congelato.
    """
    model.train()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_loss = AverageMeter()
    run_acc = AverageMeter()
    
    # Contatori per classe
    num_classes = None
    per_class_correct = None
    per_class_total = None

    
    if args.augmentation:
        train_transform = get_train_transforms()
    else:
        train_transform = None
    
    # Liste per confusion matrix
    all_preds = []
    all_targets = []
    
    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    
    for idx, batch_data in enumerate(loader):
        # Estrai data e target
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]
        
        # ============================================
        # AUGMENTATION qui, on-the-fly
        # ============================================
        if train_transform is not None:
            # Augmenta solo il 50% dei samples nel batch
            data = selective_augmentation(
                data, 
                train_transform,
                augmentation_ratio=0.5  # ← 50% originali, 50% augmentati
            )

        data = data.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        
        # Costruisci logits per l'intero batch
        batch_logits = []
        feat_list_all = []
        hidden_list_all = []
        
        B = data.shape[0]
        for b in range(B):
            vol = data[b:b+1]  # [1,C,D,H,W]
            vol = ensure_single_channel(vol, mode="first")  # [1,1,D,H,W]
            
            # Estrai patch dal volume
            patches, coords = extract_patches_5d_torch(
                vol, 
                patch_size=(args.roi_z, args.roi_y, args.roi_x), 
                step=(args.roi_z, args.roi_y, args.roi_x), 
                pad_value=0
            )
            
            # Inferenza per ogni patch
            patches = patches.to(device).to(torch.float32)  # Converti una volta sola
            
            sw_batch_size = args.sw_batch_size if hasattr(args, 'sw_batch_size') else 4
            feat_list = []
            hidden_list = []

            for i in range(0, patches.shape[0], sw_batch_size):
                end_idx = min(i + sw_batch_size, patches.shape[0])
                batch_patches = patches[i:end_idx]  # [sw_batch_size, 1, 128, 128, 128]
                
                feats, hidden = model.forward_features(batch_patches)  # Forward su batch
                
                feat_list.append(feats)    # [sw_batch_size, Cf, fD, fH, fW]
                hidden_list.append(hidden) # [sw_batch_size, Ch, hD, hH, hW]

            # Concatena tutti i batch
            feats_cat = torch.cat(feat_list, dim=0)   # [N, Cf, fD, fH, fW]
            hidden_cat = torch.cat(hidden_list, dim=0) # [N, Ch, hD, hH, hW]
            
            feats_tiled = tile_feature_patches(feats_cat, coords=coords)
            hidden_tiled = tile_feature_patches(hidden_cat, coords=coords)
            
            feat_list_all.append(feats_tiled)
            hidden_list_all.append(hidden_tiled)
            
            # Classificazione: global_pool → flatten → fc
            pooled = model.global_pool(feats_tiled)
            
            if args.model_name == "swinunetr+ml_decoder":
                pooled = pooled.flatten(2)
            elif args.model_name == "swinunetr":
                pooled = pooled.flatten(1)
            
            logits_b = model.fc(pooled)  # [1,num_classes]
            batch_logits.append(logits_b)
        
        # Calcola similarity matrices solo per epoche selezionate
        should_compute_sim = (
            (epoch == 0 or epoch == (args.max_epochs - 1) or epoch == int(args.max_epochs * 0.5)) 
            and idx == 0 
            and args.rank == 0
        )
        
        sim = None
        if should_compute_sim or args.similarity_loss in ["contrastive", "margin"]:
            feat_concat = torch.cat(feat_list_all, dim=0)  # [B,Cf,D,H,W]
            feat_flat = feat_concat.view(feat_concat.shape[0], -1)  # [B,Cf*D*H*W]
            sim = compute_similarity_matrix(feat_flat)
            
            if should_compute_sim:
                sim_np = sim.detach().float().cpu().numpy()
                plot_similarity_heatmap_new(
                    sim_np, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_epoch{epoch}_iter{idx}.png")
                )
                
                hidden_concat = torch.cat(hidden_list_all, dim=0)
                hidden_flat = hidden_concat.view(hidden_concat.shape[0], -1)
                sim_hidden = compute_similarity_matrix(hidden_flat).cpu().detach().numpy()
                plot_similarity_heatmap_new(
                    sim_hidden, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_hidden_epoch{epoch}_iter{idx}.png")
                )
        
        # Calcola loss
        logits = torch.cat(batch_logits, dim=0)  # [B,num_classes]
        loss = loss_func(logits, target)
        
        # Inizializza contatori per classe alla prima iterazione
        if num_classes is None:
            num_classes = logits.shape[1]
            per_class_correct = np.zeros(num_classes, dtype=np.int64)
            per_class_total = np.zeros(num_classes, dtype=np.int64)
        
        # Calcola metriche di accuracy
        with torch.no_grad():
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # [B]
            target_eval = target.view(-1) if target.ndim > 1 else target
            
            all_preds.append(preds.cpu())
            all_targets.append(target_eval.cpu())
            
            # Accuracy batch
            correct = (preds == target_eval).sum().item()
            not_nans = target_eval.numel()
            
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)
            
            # Per-class accuracy
            t_cpu = target_eval.cpu().numpy()
            p_cpu = preds.cpu().numpy()
            
            # Calcola correttezza per ogni sample
            mask = (p_cpu == t_cpu)
            
            # Conta totali e corretti per classe
            batch_total = np.bincount(t_cpu, minlength=num_classes)
            batch_correct = np.bincount(t_cpu[mask], minlength=num_classes)
            
            per_class_correct += batch_correct
            per_class_total += batch_total
        
        sim_loss_value = 0.0
        if args.similarity_loss == "contrastive" and sim is not None:
            loss_sim = supervised_contrastive_from_similarity(sim, target, temperature=0.07)
            loss = loss + args.similarity_loss_weight * loss_sim
            sim_loss_value = loss_sim.item()
        elif args.similarity_loss == "margin" and sim is not None:
            loss_sim = similarity_margin_loss(sim, target, pos_margin=0.5, neg_margin=0.0)
            loss = loss + args.similarity_loss_weight * loss_sim
            sim_loss_value = loss_sim.item()
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Aggiorna metriche
        if is_distributed:
            loss_list = distributed_all_gather(
                [loss], 
                out_numpy=True, 
                is_valid=idx < loader.sampler.valid_length
            )
            run_loss.update(
                np.mean(np.mean(np.stack(loss_list, axis=0), axis=0), axis=0),
                n=args.batch_size * args.world_size
            )
            
            # Aggregazione accuracy globale
            acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
            n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
            
            acc_list, not_nans_list = distributed_all_gather(
                [acc_tensor, n_tensor],
                out_numpy=True,
                is_valid=idx < loader.sampler.valid_length
            )
            
            for al, nl in zip(acc_list, not_nans_list):
                run_acc.update(float(al), n=int(nl))
        else:
            run_loss.update(loss.item(), n=args.batch_size)
            run_acc.update(acc, n=not_nans)
        
        # Logging
        if is_main_process:
            if sim is not None:
                print(
                    f"Epoch: {epoch}/{args.max_epochs} Iter: {idx}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"sim_loss: {sim_loss_value:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
            else:
                print(
                    f"Epoch: {epoch}/{args.max_epochs} Iter: {idx}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
        start_time = time.time()
    
    # Calcola accuracy per classe
    per_class_acc = {
        int(c): float(per_class_correct[c]) / max(1, int(per_class_total[c]))
        for c in range(num_classes)
    }
    
    # Confusion matrix
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    # Stampa riassunto
    if is_main_process:
        summary = ", ".join([f"c{c}: {per_class_acc[c]:.3f}" for c in range(num_classes)])
        print(f"[Train epoch {epoch}] avg_loss={run_loss.avg:.4f} avg_acc={run_acc.avg:.4f} | per-class [{summary}]")
    
    return run_loss.avg, float(run_acc.avg), per_class_acc, cm


def val_epoch(
    model,
    loader: DataLoader,
    epoch: int,
    acc_func,
    args,
) -> tuple[float, dict, np.ndarray]:
    """
    Validazione con pipeline di inferenza a patch usando forward_features.
    
    Returns:
        tuple: (avg_accuracy, per_class_accuracy_dict, confusion_matrix)
    """
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_acc = AverageMeter()
    
    # Contatori per classe
    num_classes = None
    per_class_correct = None
    per_class_total = None
    
    # Liste per confusion matrix
    all_preds = []
    all_targets = []
    
    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data e target
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Forward a patch per volume
            batch_logits = []
            B = data.shape[0]
            
            for b in range(B):
                vol = data[b:b+1]  # [1,C,D,H,W]
                vol = ensure_single_channel(vol, mode="first")  # [1,1,D,H,W]
                
                # Estrai patch
                patches, coords = extract_patches_5d_torch(
                    vol,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=(args.roi_z, args.roi_y, args.roi_x),
                    pad_value=0
                )
                
                # Converti patches una volta sola
                patches = patches.to(device).to(torch.float32)
                
                # Inferenza per ogni patch
                feat_list = []
                for i in range(patches.shape[0]):
                    patch = patches[i:i+1]
                    feats, _ = model.forward_features(patch)
                    feat_list.append(feats)
                
                # Ricostruisci volume features
                feats_cat = torch.cat(feat_list, dim=0)
                feats_tiled = tile_feature_patches(feats_cat, coords=coords)
                
                # Classificazione
                pooled = model.global_pool(feats_tiled)
                
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr":
                    pooled = pooled.flatten(1)
                
                logits_b = model.fc(pooled)  # [1,num_classes]
                batch_logits.append(logits_b)
            
            logits = torch.cat(batch_logits, dim=0)  # [B,num_classes]
            
            # Inizializza contatori per classe alla prima iterazione
            if num_classes is None:
                num_classes = logits.shape[1]
                per_class_correct = np.zeros(num_classes, dtype=np.int64)
                per_class_total = np.zeros(num_classes, dtype=np.int64)
            
            # Predizioni
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # [B]
            target_eval = target.view(-1) if target.ndim > 1 else target
            
            all_preds.append(preds.cpu())
            all_targets.append(target_eval.cpu())
            
            # Accuracy batch
            correct = (preds == target_eval).sum().item()
            not_nans = target_eval.numel()
            
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)
            
            # Per-class accuracy
            t_cpu = target_eval.cpu().numpy()
            p_cpu = preds.cpu().numpy()
            
            # Calcola correttezza per ogni sample
            mask = (p_cpu == t_cpu)
            
            # Conta totali e corretti per classe in un solo passaggio
            batch_total = np.bincount(t_cpu, minlength=num_classes)
            batch_correct = np.bincount(t_cpu[mask], minlength=num_classes)
            
            if is_distributed:
                # Verifica validità sample
                is_valid = idx < loader.sampler.valid_length if hasattr(loader.sampler, "valid_length") else True
                
                # All-gather per classe
                correct_vec = torch.tensor(batch_correct, device=device, dtype=torch.float32)
                total_vec = torch.tensor(batch_total, device=device, dtype=torch.float32)
                
                corr_list, tot_list = distributed_all_gather(
                    [correct_vec, total_vec],
                    out_numpy=True,
                    is_valid=is_valid
                )
                
                per_class_correct += np.sum(np.stack(corr_list, axis=0), axis=0).astype(np.int64)
                per_class_total += np.sum(np.stack(tot_list, axis=0), axis=0).astype(np.int64)
                
                # Aggregazione accuracy globale
                acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
                n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
                
                acc_list, not_nans_list = distributed_all_gather(
                    [acc_tensor, n_tensor],
                    out_numpy=True,
                    is_valid=is_valid
                )
                
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(float(al), n=int(nl))
            else:
                per_class_correct += batch_correct
                per_class_total += batch_total
                run_acc.update(acc, n=not_nans)
            
            # Logging
            if is_main_process:
                print(
                    f"Val {epoch}/{args.max_epochs} {idx}/{len(loader)}, "
                    f"Acc: {run_acc.avg:.4f}, time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()
    
    # Gestisci caso senza batch processati
    if num_classes is None:
        return float('nan'), {}, np.array([])
    
    # Calcola accuracy per classe
    per_class_acc = {
        int(c): float(per_class_correct[c]) / max(1, int(per_class_total[c]))
        for c in range(num_classes)
    }
    
    # Confusion matrix
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    # Stampa riassunto
    if is_main_process:
        summary = ", ".join([f"c{c}: {per_class_acc[c]:.3f}" for c in range(num_classes)])
        print(f"[Val epoch {epoch}] avg_acc={run_acc.avg:.4f} | per-class [{summary}]")
    
    return float(run_acc.avg), per_class_acc, cm

def val_epoch_new(
    model,
    loader: DataLoader,
    epoch: int,
    acc_func,
    args,
) -> tuple[float, dict, np.ndarray]:
    """
    Validazione con pipeline di inferenza a patch ottimizzata.
    
    Returns:
        tuple: (avg_accuracy, per_class_accuracy_dict, confusion_matrix)
    """
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_acc = AverageMeter()
    
    # Contatori per classe
    num_classes = None
    per_class_correct = None
    per_class_total = None
    
    # Liste per confusion matrix
    all_preds = []
    all_targets = []
    
    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    sw_batch_size = getattr(args, 'sw_batch_size', 4)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data e target
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]
            
            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Assicura single channel per tutto il batch
            data = ensure_single_channel(data, mode="first")  # [B,1,D,H,W]
            B = data.shape[0]
            
            # ============================================
            # STEP 1: Estrai TUTTE le patch di TUTTO il batch
            # ============================================
            all_patches = []
            all_coords = []
            batch_indices = []
            patches_per_sample = []
            
            for b in range(B):
                vol = data[b:b+1]  # [1,1,D,H,W]
                
                # Estrai patch con overlap per smooth blending
                patches, coords = extract_patches_5d_torch(
                    vol,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=args.step,
                    pad_value=0
                )
                
                all_patches.append(patches)
                all_coords.extend(coords)
                batch_indices.extend([b] * patches.shape[0])
                patches_per_sample.append(patches.shape[0])
            
            # Concatena tutte le patch in un mega-batch
            all_patches = torch.cat(all_patches, dim=0).to(torch.float32)  # [Total_N, 1, D, H, W]
            total_patches = all_patches.shape[0]
            
            # ============================================
            # STEP 2: Batch processing delle patch
            # ============================================
            all_feats = []
            
            # Processa le patch in batch invece che una alla volta
            for i in range(0, total_patches, sw_batch_size):
                end_idx = min(i + sw_batch_size, total_patches)
                batch_patches = all_patches[i:end_idx]  # [sw_batch_size, 1, D, H, W]
                
                # Forward pass su batch di patch (solo features, non serve hidden in validazione)
                feats, _ = model.forward_features(batch_patches)
                all_feats.append(feats)
            
            # Concatena tutti i risultati
            all_feats = torch.cat(all_feats, dim=0)  # [Total_N, Cf, fD, fH, fW]
            
            # ============================================
            # STEP 3: Ricostruisci per ogni sample del batch con blending
            # ============================================
            batch_logits = []
            
            start_idx = 0
            for b in range(B):
                num_patches = patches_per_sample[b]
                end_idx = start_idx + num_patches
                
                # Estrai features di questo sample
                b_feats = all_feats[start_idx:end_idx]  # [num_patches, Cf, fD, fH, fW]
                b_coords = all_coords[start_idx:end_idx]
                
                # Ricostruisci con blending (invece di tiling rigido)
                feats_tiled = tile_with_gaussian_blending(
                    b_feats,
                    b_coords,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=args.step
                )
                
                # Classificazione: global_pool → flatten → fc
                pooled = model.global_pool(feats_tiled)
                
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr":
                    pooled = pooled.flatten(1)
                
                logits_b = model.fc(pooled)  # [1, num_classes]
                batch_logits.append(logits_b)
                
                start_idx = end_idx
            
            logits = torch.cat(batch_logits, dim=0)  # [B, num_classes]
            
            # ============================================
            # STEP 4: Calcola metriche
            # ============================================
            
            # Inizializza contatori per classe alla prima iterazione
            if num_classes is None:
                num_classes = logits.shape[1]
                per_class_correct = np.zeros(num_classes, dtype=np.int64)
                per_class_total = np.zeros(num_classes, dtype=np.int64)
            
            # Predizioni
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)  # [B]
            target_eval = target.view(-1) if target.ndim > 1 else target
            
            all_preds.append(preds.cpu())
            all_targets.append(target_eval.cpu())
            
            # Accuracy batch
            correct = (preds == target_eval).sum().item()
            not_nans = target_eval.numel()
            
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)
            
            # Per-class accuracy
            t_cpu = target_eval.cpu().numpy()
            p_cpu = preds.cpu().numpy()
            
            # Calcola correttezza per ogni sample
            mask = (p_cpu == t_cpu)
            
            # Conta totali e corretti per classe in un solo passaggio
            batch_total = np.bincount(t_cpu, minlength=num_classes)
            batch_correct = np.bincount(t_cpu[mask], minlength=num_classes)
            
            if is_distributed:
                # Verifica validità sample
                is_valid = idx < loader.sampler.valid_length if hasattr(loader.sampler, "valid_length") else True
                
                # All-gather per classe
                correct_vec = torch.tensor(batch_correct, device=device, dtype=torch.float32)
                total_vec = torch.tensor(batch_total, device=device, dtype=torch.float32)
                
                corr_list, tot_list = distributed_all_gather(
                    [correct_vec, total_vec],
                    out_numpy=True,
                    is_valid=is_valid
                )
                
                per_class_correct += np.sum(np.stack(corr_list, axis=0), axis=0).astype(np.int64)
                per_class_total += np.sum(np.stack(tot_list, axis=0), axis=0).astype(np.int64)
                
                # Aggregazione accuracy globale
                acc_tensor = torch.tensor(acc, device=device, dtype=torch.float32)
                n_tensor = torch.tensor(not_nans, device=device, dtype=torch.float32)
                
                acc_list, not_nans_list = distributed_all_gather(
                    [acc_tensor, n_tensor],
                    out_numpy=True,
                    is_valid=is_valid
                )
                
                for al, nl in zip(acc_list, not_nans_list):
                    run_acc.update(float(al), n=int(nl))
            else:
                per_class_correct += batch_correct
                per_class_total += batch_total
                run_acc.update(acc, n=not_nans)
            
            # Logging
            if is_main_process:
                print(
                    f"Val {epoch}/{args.max_epochs} {idx}/{len(loader)}, "
                    f"Acc: {run_acc.avg:.4f}, time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()
    
    # Gestisci caso senza batch processati
    if num_classes is None:
        return float('nan'), {}, np.array([])
    
    # Calcola accuracy per classe
    per_class_acc = {
        int(c): float(per_class_correct[c]) / max(1, int(per_class_total[c]))
        for c in range(num_classes)
    }
    
    # Confusion matrix
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))
    
    # Stampa riassunto
    if is_main_process:
        summary = ", ".join([f"c{c}: {per_class_acc[c]:.3f}" for c in range(num_classes)])
        print(f"[Val epoch {epoch}] avg_acc={run_acc.avg:.4f} | per-class [{summary}]")
    
    return float(run_acc.avg), per_class_acc, cm



def save_checkpoint(model, epoch, args, filename="model.pt", best_acc=0, optimizer=None, scheduler=None):
    state_dict = model.state_dict() if not args.distributed else model.module.state_dict()
    save_dict = {"epoch": epoch, "best_acc": best_acc, "state_dict": state_dict}
    if optimizer is not None:
        save_dict["optimizer"] = optimizer.state_dict()
    if scheduler is not None:
        save_dict["scheduler"] = scheduler.state_dict()
    filename = os.path.join(args.final_output_dir, filename)
    torch.save(save_dict, filename)
    print("Saving checkpoint", filename)

    
def run_training(
    model,
    train_loader,
    val_loader,
    optimizer,
    loss_func,
    acc_func,
    args,
    scheduler=None,
    start_epoch=0,
    writer_dict=None,
    final_output_dir=None,
    logger=None,
) -> float:
    """
    Loop di training principale con validation, early stopping e logging.
    
    Returns:
        float: Best validation accuracy raggiunta
    """
    # Setup logging e writer
    writer = writer_dict.get("writer") if writer_dict is not None else None
    
    # Inizializza liste per metriche
    training_losses = []
    training_accuracies = []
    training_per_class_accuracies = []
    validation_accuracies = []
    validation_per_class_accuracies = []
    lr_history = []

    # Inizializza lo step
    args.step= (args.roi_z, int(args.roi_y * 2 // 3), int(args.roi_x * 2 // 3))
    
    # Setup directory output
    args.final_output_dir = final_output_dir
    if final_output_dir is None:
        time_str = time.strftime('%Y-%m-%d-%H-%M')
        name_file = f'{args.logdir}_{time_str}'
        final_output_dir = os.path.join(args.output_dir, name_file)
    
    # Crea struttura directory
    final_plots_dir = os.path.join(final_output_dir, "plots")
    sim_plots_dir = os.path.join(final_plots_dir, "similarity")
    cm_plots_dir = os.path.join(final_plots_dir, "confusion_matrix")
    metrics_plots_dir = os.path.join(final_plots_dir, "metrics_tables")
    
    os.makedirs(final_plots_dir, exist_ok=True)
    os.makedirs(sim_plots_dir, exist_ok=True)
    os.makedirs(cm_plots_dir, exist_ok=True)
    os.makedirs(metrics_plots_dir, exist_ok=True)
    
    args.final_plots_dir = final_plots_dir
    args.sim_plots_dir = sim_plots_dir
    
    # Cache attributi comuni
    is_main_process = args.rank == 0
    should_save = is_main_process and args.final_output_dir is not None and args.save_checkpoint
    use_telegram = args.telegram_log if hasattr(args, 'telegram_log') else False
    
    val_acc_max = args.best_acc if hasattr(args, 'best_acc') else 0.0
    last_cm = None
    last_metrics = None

    model = freeze_backbone_and_select_head_fixed_plus(model)
    
    # Freeze/unfreeze layers
    # if args.encoder10_pth is not None:
    #     model = freeze_backbone_and_select_head_fixed_plus(model)
    # else:
    #     model = freeze_backbone_and_select_head_fixed(model)
    #     if is_main_process:
    #         print("Loaded from checkpoint, layers unfrozen for fine-tuning")
    #         if logger:
    #             logger.info("Loaded from checkpoint, layers unfrozen for fine-tuning")
    
    # Setup early stopping
    early_stopping_val = None
    early_stopping_loss = None
    if args.early_stopping:
        early_stopping_val = EarlyStopping(
            mode='max', 
            patience=args.patience_val, 
            min_delta=args.min_delta_val, 
            restore_best=False, 
            verbose=True
        )
        early_stopping_loss = EarlyStopping(
            mode='min', 
            patience=args.patience_loss, 
            min_delta=args.min_delta_loss, 
            restore_best=False, 
            verbose=True
        )
    
    # Training loop
    for epoch in range(start_epoch, args.max_epochs):
        if args.distributed:
            train_loader.sampler.set_epoch(epoch)
            torch.distributed.barrier()
        
        if is_main_process:
            print(f"{args.rank} {time.ctime()} Epoch: {epoch}")
            if logger:
                logger.info(f"{args.rank} {time.ctime()} Epoch: {epoch}")
        
        # Training
        epoch_time = time.time()
        if args.patch_merging:
            train_loss, train_acc, train_per_class_acc, train_cm = train_epoch_new(
                model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, acc_func=acc_func, args=args
            )
        else:
            train_loss, train_acc, train_per_class_acc, train_cm = train_epoch(
                model, train_loader, optimizer, epoch=epoch, loss_func=loss_func, acc_func=acc_func, args=args
            )
        training_losses.append(train_loss)

        last_train_cm = train_cm
        train_metrics = metrics_from_confusion_matrix(train_cm)
        last_train_metrics = train_metrics
        train_metrics_str = format_print_metrics(train_metrics)

        training_accuracies.append(train_acc)
        training_per_class_accuracies.append(train_per_class_acc)

        # Learning rate attuale
        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)
        
        if is_main_process:
            train_time = time.time() - epoch_time
            msg = (
                f"Final training {epoch}/{args.max_epochs - 1}, "
                f"loss: {train_loss:.4f}, time {train_time:.2f}s, lr: {current_lr:.6f}"
                f"\n{train_metrics_str}\n"
                f"*----------------------------------------*"
            )
            print(msg)
            if logger:
                logger.info(msg)
            
            # Telegram notification ogni 10 epoche
            if use_telegram and epoch % 10 == 0:
                telegram_msg = (
                    f"*🏋 Training - Epoch {epoch}/{args.max_epochs - 1}*\n"
                    f"Train Loss: {train_loss:.4f}\n"
                    f"Train Acc: {train_acc:.4f}\n"
                    f"Best Val Acc: {val_acc_max:.4f}\n"
                    f"LR: {current_lr:.6f}\n"
                )
                _send_telegram_safe(args, telegram_msg)
            
            # Early stopping per loss
            if early_stopping_loss and early_stopping_loss.step(train_loss, model):
                print(f"[EarlyStopping] Stopping training for loss at epoch {epoch}")
                if logger:
                    logger.info(f"[EarlyStopping] Stopping training for loss at epoch {epoch}")
                if use_telegram:
                    _send_telegram_safe(args, f"*🛑 Early Stopping (Loss) at Epoch {epoch}*")
                break
        
        # Validation
        is_new_best = False
        if (epoch + 1) % args.val_every == 0:
            if args.distributed:
                torch.distributed.barrier()
            
            epoch_time = time.time()
            if args.patch_merging:
                val_acc, val_per_class, cm = val_epoch_new(
                    model, val_loader, epoch=epoch, acc_func=acc_func, args=args
                )
            else:
                val_acc, val_per_class, cm = val_epoch(
                    model, val_loader, epoch=epoch, acc_func=acc_func, args=args
                )

            last_cm = cm
            metrics = metrics_from_confusion_matrix(cm)
            last_metrics = metrics
            metrics_str = format_print_metrics(metrics)
            
            validation_accuracies.append(val_acc)
            validation_per_class_accuracies.append(val_per_class)
            
            if is_main_process:
                val_time = time.time() - epoch_time
                msg = (
                    f"Final validation {epoch}/{args.max_epochs - 1}, "
                    f"Val_acc: {val_acc:.4f}, time {val_time:.2f}s"
                    f"{metrics_str}\n"
                    f"*========================================*"
                )
                print(msg)
                if logger:
                    logger.info(msg)
                
                # Check new best
                if val_acc > val_acc_max:
                    print(f"New best ({val_acc_max:.6f} --> {val_acc:.6f})")
                    if logger:
                        logger.info(f"New best ({val_acc_max:.6f} --> {val_acc:.6f})")
                    
                    val_acc_max = val_acc
                    is_new_best = True
                    
                    # Salva plot del best model
                    class_names = [f"class {i}" for i in range(cm.shape[0])]
                    plot_confusion_matrix(
                        cm, 
                        class_names=class_names,
                        title=f'Confusion Matrix - Epoch {epoch}',
                        save_path=os.path.join(cm_plots_dir, f"best_confusion_matrix_epoch{epoch}.png")
                    )
                    plot_confusion_matrix(
                        train_cm,
                        class_names=class_names,
                        title=f'Confusion Matrix (Train) - Epoch {epoch} ',
                        save_path=os.path.join(cm_plots_dir, f"best_confusion_train_matrix_epoch{epoch}.png")
                    )
                    plot_metrics_table(
                        metrics,
                        class_names=class_names,
                        title=f'Metrics Table - Epoch {epoch}',
                        save_path=os.path.join(metrics_plots_dir, f"best_metrics_table_epoch{epoch}.png")
                    )
                    plot_metrics_table(
                        train_metrics,
                        class_names=class_names,
                        title=f'Metrics Table (Train) - Epoch {epoch}',
                        save_path=os.path.join(metrics_plots_dir, f"best_train_metrics_table_epoch{epoch}.png")
                    )

                # Telegram notification
                if use_telegram:
                    telegram_msg = (
                        f"*✅ Validation - Epoch {epoch}/{args.max_epochs - 1}*\n"
                        f"Val Acc: {val_acc:.4f}\n"
                        f"Best Val Acc: {val_acc_max:.4f}"
                    )
                    _send_telegram_safe(args, telegram_msg)
            
            # Salva checkpoint
            if should_save:
                save_checkpoint(
                    model, epoch, args, 
                    best_acc=val_acc_max, 
                    optimizer=optimizer, 
                    scheduler=scheduler,
                    filename="model_final.pt"
                )
                
                if is_new_best:
                    print("Copying best model to model.pt")
                    if logger:
                        logger.info("Copying best model to model.pt")
                    shutil.copyfile(
                        os.path.join(args.final_output_dir, "model_final.pt"),
                        os.path.join(args.final_output_dir, "model.pt")
                    )
            
            # Early stopping per validation
            if early_stopping_val and early_stopping_val.step(val_acc, model):
                print(f"[EarlyStopping] Stopping training for val accuracy at epoch {epoch}")
                if logger:
                    logger.info(f"[EarlyStopping] Stopping training for val accuracy at epoch {epoch}")
                if use_telegram:
                    _send_telegram_safe(args, f"*🛑 Early Stopping (Validation) at Epoch {epoch}*")
                break
        
        # Step scheduler
        if scheduler is not None:
            scheduler.step()
    
    # Fine training
    if is_main_process:
        print(f"Training Finished! Best Accuracy: {val_acc_max:.4f}")
        if logger:
            logger.info(f"Training Finished! Best Accuracy: {val_acc_max:.4f}")
            logger.info("=" * 100)
        
        if use_telegram:
            time_str = time.strftime('%Y/%m/%d %H:%M')
            telegram_msg = (
                f"*🏆 Training Finished!*\n"
                f"{time_str}\n"
                f"Best Val Acc: {val_acc_max:.4f}"
            )
            _send_telegram_safe(args, telegram_msg)
        
        # Salva plot finali
        if last_cm is not None and last_metrics is not None:
            print(f"Saving plots to: {final_plots_dir}")
            
            class_names = [f"class {i}" for i in range(last_cm.shape[0])]
            
            # Confusion matrix e metrics
            plot_confusion_matrix(
                last_cm,
                class_names=class_names,
                title='Confusion Matrix - Final Epoch',
                save_path=os.path.join(cm_plots_dir, "final_confusion_matrix.png")
            )
            plot_confusion_matrix(
                last_train_cm,
                class_names=class_names,
                title='Confusion Matrix (Train) - Final Epoch ',
                save_path=os.path.join(cm_plots_dir, "final_confusion_train_matrix.png")
            )
            plot_metrics_table(
                last_metrics,
                class_names=class_names,
                title='Metrics Table - Final Epoch',
                save_path=os.path.join(metrics_plots_dir, "final_metrics_table.png")
            )
            plot_metrics_table(
                last_train_metrics,
                class_names=class_names,
                title='Metrics Table (Train) - Final Epoch',
                save_path=os.path.join(metrics_plots_dir, "final_train_metrics_table.png")
            )
            
            # Training curves
            plot_training_curve(
                training_losses,
                metric_name="Loss",
                title="Training Curve - Loss",
                save_path=os.path.join(metrics_plots_dir, "training_loss_curve.png")
            )
            plot_training_curve(
                lr_history,
                metric_name="Learning Rate",
                title="Training Curve - Learning Rate",
                save_path=os.path.join(metrics_plots_dir, "learning_rate_curve.png")
            )
            plot_loss_lr(
                training_losses,
                lr_history,
                title="Training Curve - Loss vs Learning Rate",
                save_path=os.path.join(metrics_plots_dir, "loss_vs_lr_curve.png")
            )
            plot_multi_class_training_curve(
                training_accuracies,
                training_per_class_accuracies,
                title="Training Curve - Accuracy",
                save_path=os.path.join(metrics_plots_dir, "training_accuracy_curve.png")
            )
            plot_multi_class_training_curve(
                validation_accuracies,
                validation_per_class_accuracies,
                title="Validation Curve - Accuracy",
                save_path=os.path.join(metrics_plots_dir, "validation_accuracy_curve.png")
            )
            
            # Telegram plot notifications
            if use_telegram:
                _send_telegram_plots(args, metrics_plots_dir, cm_plots_dir)
    
    return val_acc_max


def _send_telegram_safe(args, message):
    """Helper per inviare messaggi Telegram con gestione errori."""
    try:
        asyncio.run(send_alert(args.oar_id, message, token_file=args.token))
    except Exception as e:
        print(f"[Warning] Telegram notification failed: {e}")


def _send_telegram_plots(args, plots_dir, cm_dir):
    """Helper per inviare plot via Telegram."""
    plots_to_send = [
        ("*Loss Curve*", os.path.join(plots_dir, "training_loss_curve.png"), None),
        ("*Accuracy Curve*", os.path.join(plots_dir, "validation_accuracy_curve.png"), None),
        ("*Learning Rate Curve*", os.path.join(plots_dir, "learning_rate_curve.png"), None),
        ("*Loss vs LR Curve*", os.path.join(plots_dir, "loss_vs_lr_curve.png"), None),
        ("*Confusion Matrix*", os.path.join(cm_dir, "final_confusion_matrix.png"), None),
        ("*Metrics Table*", os.path.join(plots_dir, "final_metrics_table.png"), None),
    ]
    
    for msg, img_path, text_suffix in plots_to_send:
        full_msg = f"{msg}\n{text_suffix}" if text_suffix else msg
        try:
            if img_path:
                asyncio.run(send_alert(args.oar_id, full_msg, token_file=args.token, image_path=img_path))
            else:
                asyncio.run(send_alert(args.oar_id, full_msg, token_file=args.token))
        except Exception as e:
            print(f"[Warning] Failed to send telegram plot {msg}: {e}")
