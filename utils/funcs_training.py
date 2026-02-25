# Standard library
import os
import time

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Local imports - Utils
from .utils import (
    AverageMeter,
    distributed_all_gather,
    extract_patches_5d_torch,
    ensure_single_channel,
    tile_feature_patches,
    tile_with_gaussian_blending,
)

# Local imports - Tools

from tools.similarity import (
    compute_similarity_matrix,
    plot_similarity_heatmap_new,
)
from tools.loss import (
    similarity_margin_loss,
    supervised_contrastive_from_similarity,
)

# Local imports - Other
from optimizers.early_stop import EarlyStopping
from dataset import get_train_transforms, selective_augmentation


def train_epoch_pm(model, loader, optimizer, epoch, loss_func, acc_func, args):
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

    # Liste per la visualizzazione degli errori
    all_errors_paths = []

    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    
    for idx, batch_data in enumerate(loader):
        # Estrai data e target
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]

        paths = batch_data["path"] if "path" in batch_data else None
        
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
            # Augmenta solo il 60% dei samples nel batch
            data = selective_augmentation(
                data, 
                train_transform,
                augmentation_ratio=0.7  # ← 30% originali, 70% augmentati
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
            # print(f"Sample {b}: num_patches={num_patches}, b_feats shape={b_feats.shape}")
            
            # Ricostruisci con blending (invece di tiling rigido)
            feats_tiled = tile_with_gaussian_blending(
                b_feats,
                b_coords,
                patch_size=(args.roi_z, args.roi_y, args.roi_x),
                step=args.step
            )

            # print(f"Sample {b}: feats_tiled shape={feats_tiled.shape}")
            
            
            
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
            # print(f"Sample {b}: pooled shape={pooled.shape}")
            
            if args.model_name == "swinunetr+ml_decoder":
                pooled = pooled.flatten(2)
            elif args.model_name == "swinunetr" or "resnet" in args.model_name or  "densenet" in args.model_name or "swinvit" in args.model_name:
                pooled = pooled.flatten(1)
            
            # print(f"Sample {b}: flattened pooled shape={pooled.shape}")
            pooled = model.dropout_head(pooled)
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
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_epoch{epoch+1}_iter{idx}.png")
                )
                
                hidden_concat = torch.cat(hidden_list_all, dim=0)
                hidden_flat = hidden_concat.view(hidden_concat.shape[0], -1)
                sim_hidden = compute_similarity_matrix(hidden_flat).cpu().detach().numpy()
                plot_similarity_heatmap_new(
                    sim_hidden, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_hidden_epoch{epoch+1}_iter{idx}.png")
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
            all_errors_paths.extend([(paths[i], preds[i], target_eval[i]) for i in range(len(paths)) if preds[i] != target_eval[i]])
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
            if sim_loss_value != 0.0:
                print(
                    f"Epoch: {epoch+1}/{args.max_epochs} Iter: {idx+1}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"sim_loss: {sim_loss_value:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
            else:
                print(
                    f"Epoch: {epoch+1}/{args.max_epochs} Iter: {idx+1}/{len(loader)} "
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
        print(f"[Train epoch {epoch+1}] avg_loss={run_loss.avg:.4f} avg_acc={run_acc.avg:.4f} | per-class [{summary}]")
    
    return run_loss.avg, float(run_acc.avg), per_class_acc, cm, all_errors_paths

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

    # Liste per la visualizzazione degli errori
    all_errors_paths = []
    
    # Cache per attributi args
    is_distributed = getattr(args, "distributed", False)
    is_main_process = getattr(args, "rank", 0) == 0
    
    for idx, batch_data in enumerate(loader):
        # Estrai data e target
        if isinstance(batch_data, list):
            data, target = batch_data
        else:
            data, target = batch_data["vol"], batch_data["label"]

        paths = batch_data["path"] if "path" in batch_data else None

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
                augmentation_ratio=0.7  # ← 30% originali, 70% augmentati
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
                step=(args.roi_z, args.roi_y, args.roi_x),
                pad_value=0
            )
            
            all_patches.append(patches)
            all_coords.extend(coords)
            batch_indices.extend([b] * patches.shape[0])
            patches_per_sample.append(patches.shape[0])
        
        # Concatena tutte le patch in un mega-batch
        all_patches = torch.cat(all_patches, dim=0).to(torch.float32)  # [Total_N, 1, D, H, W]
        total_patches = all_patches.shape[0]
        #print(f"Total patches:{total_patches}")
        
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
            feats_tiled = tile_feature_patches(b_feats,b_coords)

            
            
            
            feat_list_all.append(feats_tiled)

            if should_compute_sim:
                hidden_tiled = tile_feature_patches(b_hidden,b_coords)

                hidden_list_all.append(hidden_tiled)
            
            # Classificazione: global_pool → flatten → fc
            pooled = model.global_pool(feats_tiled)
            
            if args.model_name == "swinunetr+ml_decoder":
                pooled = pooled.flatten(2)
            elif args.model_name == "swinunetr" or "resnet" in args.model_name or  "densenet" in args.model_name or args.model_name == "swinvit":
                pooled = pooled.flatten(1)
            
            pooled = model.dropout_head(pooled)
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
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_epoch{epoch+1}_iter{idx}.png")
                )
                

                
                hidden_concat = torch.cat(hidden_list_all, dim=0)
                hidden_flat = hidden_concat.view(hidden_concat.shape[0], -1)
                sim_hidden = compute_similarity_matrix(hidden_flat).cpu().detach().numpy()
                plot_similarity_heatmap_new(
                    sim_hidden, 
                    target, 
                    save_path=os.path.join(args.sim_plots_dir, f"similarity_hidden_epoch{epoch+1}_iter{idx}.png")
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
            all_errors_paths.extend([(paths[i], preds[i], target_eval[i]) for i in range(len(paths)) if preds[i] != target_eval[i]])
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
            if sim_loss_value != 0.0:
                print(
                    f"Epoch: {epoch+1}/{args.max_epochs} Iter: {idx+1}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"sim_loss: {sim_loss_value:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
            else:
                print(
                    f"Epoch: {epoch+1}/{args.max_epochs} Iter: {idx+1}/{len(loader)} "
                    f"loss: {run_loss.avg:.4f} acc: {run_acc.avg:.4f} "
                    f"time {time.time() - start_time:.2f}s"
                )
                
            if time.time() - start_time > 5.0:  # Logga se ci sono batch particolarmente lenti
                print(f"Warning: Batch {idx+1} took {time.time() - start_time:.2f}s, which is longer than expected.")
                for i in range(0,args.batch_size):
                    print(f"  Sample patch {i+1} shape: {batch_patches[i].shape}")
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
        print(f"[Train epoch {epoch+1}] avg_loss={run_loss.avg:.4f} avg_acc={run_acc.avg:.4f} | per-class [{summary}]")
    
    return run_loss.avg, float(run_acc.avg), per_class_acc, cm, all_errors_paths
