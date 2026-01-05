# Standard library
import asyncio
import os
import shutil
import time
from tracemalloc import start

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

# Local imports - Utilss
from .utils import (
    AverageMeter,
    distributed_all_gather,
    extract_patches_5d_torch,
    ensure_single_channel,
    tile_feature_patches,
    tile_with_gaussian_blending,
)


# Local imports - Other
from optimizers.early_stop import EarlyStopping


def val_epoch_pm(model,loader: DataLoader,epoch: int,acc_func,loss_func,args,) -> tuple[float, dict, np.ndarray]:
    """
    Validazione con pipeline di inferenza a patch ottimizzata.
    
    Returns:
        tuple: (avg_accuracy, per_class_accuracy_dict, confusion_matrix)
    """
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_acc = AverageMeter()
    run_loss = AverageMeter()
    
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
    sw_batch_size = getattr(args, 'sw_batch_size', 4)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data e target
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]
            
            paths = batch_data["path"] if "path" in batch_data else None

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
                elif args.model_name == "swinunetr" or "resnet" in args.model_name or  "densenet" in args.model_name or "swinvit" in args.model_name:
                    pooled = pooled.flatten(1)
                
                logits_b = model.fc(pooled)  # [1, num_classes]
                batch_logits.append(logits_b)
                
                start_idx = end_idx
            
            logits = torch.cat(batch_logits, dim=0)  # [B, num_classes]
            loss = loss_func(logits, target)
            
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
                run_loss.update(loss.item(), n=not_nans)
            
            # Logging
            if is_main_process:
                print(
                    f"Val {epoch+1}/{args.max_epochs} {idx+1}/{len(loader)}, "
                    f"Acc: {run_acc.avg:.4f}, Loss: {run_loss.avg:.4f}, time {time.time() - start_time:.2f}s"
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
        print(f"[Val epoch {epoch+1}] avg_acc={run_acc.avg:.4f} | avg_loss={run_loss.avg:.4f} | per-class [{summary}]")

    return float(run_loss.avg), float(run_acc.avg), per_class_acc, cm, all_errors_paths

def val_epoch(model,loader: DataLoader,epoch: int,acc_func,loss_func,args,) -> tuple[float, dict, np.ndarray]:
    """
    Validazione con pipeline di inferenza a patch ottimizzata.
    
    Returns:
        tuple: (avg_accuracy, per_class_accuracy_dict, confusion_matrix)
    """
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    start_time = time.time()
    run_acc = AverageMeter()
    run_loss = AverageMeter()
    
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
    sw_batch_size = getattr(args, 'sw_batch_size', 4)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data e target
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]

            paths = batch_data["path"] if "path" in batch_data else None
            
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
                feats_tiled = tile_feature_patches(b_feats, coords=b_coords)
                
                # Classificazione: global_pool → flatten → fc
                pooled = model.global_pool(feats_tiled)
                
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr" or "resnet" in args.model_name or  "densenet" in args.model_name or "swinvit" in args.model_name:
                    pooled = pooled.flatten(1)
                
                logits_b = model.fc(pooled)  # [1, num_classes]
                batch_logits.append(logits_b)
                
                start_idx = end_idx
            
            logits = torch.cat(batch_logits, dim=0)  # [B, num_classes]
            loss = loss_func(logits, target)
            
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
                run_loss.update(loss.item(), n=not_nans)
            
            # Logging
            if is_main_process:
                print(
                    f"Val {epoch+1}/{args.max_epochs} {idx+1}/{len(loader)}, "
                    f"Acc: {run_acc.avg:.4f}, Loss: {run_loss.avg:.4f}, time {time.time() - start_time:.2f}s"
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
        print(f"[Val epoch {epoch+1}] avg_acc={run_acc.avg:.4f} | avg_loss={run_loss.avg:.4f} | per-class [{summary}]")

    return float(run_loss.avg), float(run_acc.avg), per_class_acc, cm, all_errors_paths
