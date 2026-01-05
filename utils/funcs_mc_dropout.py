import torch
import torch.nn as nn
import numpy as np
import time
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
import os
import time

# Third-party libraries
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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

def enable_mc_dropout(model):
    """
    Riattiva SOLO i layer di Dropout anche in eval mode.
    Mantiene BatchNorm in eval (statistiche globali).
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout3d)):
            module.train()

def disable_mc_dropout(model):
    """
    Disattiva MC-Dropout, tornando a eval mode classico.
    """
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout3d)):
            module.eval()


def mc_dropout_inference(
    model,
    feats_tiled,
    num_samples: int = 20,
    args=None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    MC-Dropout inference: forward pass multipli con dropout attivo.
    
    Args:
        model: ResNet_3D model (deve essere in eval mode base)
        feats_tiled: Feature map tiled [1, C, D, H, W]
        num_samples: Numero di MC samples
        args: Argomenti per il modello (model_name, etc.)
    
    Returns:
        tuple: (mean_logits, std_logits, mean_probs)
            - mean_logits: media logits [1, num_classes]
            - std_logits: std logits [1, num_classes]
            - mean_probs: media probabilità [1, num_classes]
    """
    device = feats_tiled.device
    
    # Riattiva SOLO i dropout
    enable_mc_dropout(model)
    
    all_logits = []
    
    with torch.no_grad():
        for _ in range(num_samples):
            # Global pooling
            pooled = model.global_pool(feats_tiled)  # [1, C, 1, 1, 1]
            
            # Flatten in base al modello
            if args and args.model_name == "swinunetr+ml_decoder":
                pooled = pooled.flatten(2)  # [1, C, 1]
            else:
                pooled = pooled.flatten(1)  # [1, C]
            
            # ✅ Dropout attivo (forzato da enable_mc_dropout)
            pooled = model.dropout_head(pooled)
            
            # FC classification
            logits = model.fc(pooled)  # [1, num_classes]
            all_logits.append(logits)
    
    # Disattiva MC-Dropout per tornare a eval mode classico
    disable_mc_dropout(model)
    
    # Stack: [num_samples, 1, num_classes]
    all_logits = torch.stack(all_logits, dim=0)
    
    # Statistiche
    mean_logits = all_logits.mean(dim=0)  # [1, num_classes]
    std_logits = all_logits.std(dim=0)    # [1, num_classes]
    
    # Probabilità dalla media dei logits
    mean_probs = torch.softmax(mean_logits, dim=1)
    
    return mean_logits, std_logits, mean_probs

def mc_dropout_batch_inference(
    model,
    all_feats: torch.Tensor,
    all_coords: list,
    patches_per_sample: list,
    num_samples: int = 20,
    args=None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    MC-Dropout inference per un intero batch di sample.
    Ritorna medie e incertezze per ogni sample.
    
    Args:
        model: ResNet_3D model
        all_feats: Feature map concatenate [Total_N, Cf, fD, fH, fW]
        all_coords: Coordinate di tutte le patch
        patches_per_sample: Numero di patch per ogni sample del batch
        num_samples: Numero di MC samples
        args: Argomenti del modello
    
    Returns:
        tuple: (mean_probs_all, std_probs_all, uncertainty_all)
            - mean_probs_all: media probs per sample [B, num_classes]
            - std_probs_all: std probs per sample [B, num_classes]
            - uncertainty_all: incertezza (entropy) per sample [B]
    """
    
    device = all_feats.device
    B = len(patches_per_sample)
    num_classes = None
    
    # Contenitori per risultati
    mean_probs_all = []
    std_probs_all = []
    uncertainty_all = []
    
    # Riattiva MC-Dropout
    enable_mc_dropout(model)
    
    start_idx = 0
    with torch.no_grad():
        for b in range(B):
            num_patches = patches_per_sample[b]
            end_idx = start_idx + num_patches
            
            # Estrai features di questo sample
            b_feats = all_feats[start_idx:end_idx]  # [num_patches, Cf, fD, fH, fW]
            b_coords = all_coords[start_idx:end_idx]
            
            # Ricostruisci con blending
            feats_tiled = tile_with_gaussian_blending(
                b_feats,
                b_coords,
                patch_size=(args.roi_z, args.roi_y, args.roi_x),
                step=args.step
            )
            
            # MC-Dropout forward pass multipli
            all_probs = []
            
            for _ in range(num_samples):
                pooled = model.global_pool(feats_tiled)
                
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                else:
                    pooled = pooled.flatten(1)
                
                # ✅ Dropout attivo
                pooled = model.dropout_head(pooled)
                logits = model.fc(pooled)  # [1, num_classes]
                probs = torch.softmax(logits, dim=1)
                all_probs.append(probs)
            
            # Stack e calcola statistiche
            all_probs = torch.stack(all_probs, dim=0)  # [num_samples, 1, num_classes]
            mean_probs = all_probs.mean(dim=0)  # [1, num_classes]
            std_probs = all_probs.std(dim=0)    # [1, num_classes]
            
            # Incertezza: entropia sulla media delle probabilità
            entropy = -(mean_probs * torch.log(mean_probs + 1e-8)).sum(dim=1)  # [1]
            
            mean_probs_all.append(mean_probs.cpu().numpy())
            std_probs_all.append(std_probs.cpu().numpy())
            uncertainty_all.append(entropy.cpu().numpy())
            
            start_idx = end_idx
            
            if num_classes is None:
                num_classes = mean_probs.shape[1]
    
    # Disattiva MC-Dropout
    disable_mc_dropout(model)
    
    # Concatena risultati
    mean_probs_all = np.concatenate(mean_probs_all, axis=0)  # [B, num_classes]
    std_probs_all = np.concatenate(std_probs_all, axis=0)    # [B, num_classes]
    uncertainty_all = np.concatenate(uncertainty_all, axis=0).flatten()  # [B]
    
    return mean_probs_all, std_probs_all, uncertainty_all

def plot_uncertainty_analysis(
    uncertainties: np.ndarray,
    targets: np.ndarray,
    preds: np.ndarray,
    save_dir: str,
):
    """
    Visualizza analisi dell'incertezza MC-Dropout.
    
    Args:
        uncertainties: Incertezze [N]
        targets: Target veri [N]
        preds: Predizioni [N]
        save_dir: Directory per salvare i plot
    """
    import matplotlib.pyplot as plt
    
    os.makedirs(save_dir, exist_ok=True)
    
    # Separazione correct/incorrect
    correct_mask = (preds == targets)
    correct_uncertainty = uncertainties[correct_mask]
    incorrect_uncertainty = uncertainties[~correct_mask]
    
    # Plot 1: Distribuzione incertezze
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].hist(correct_uncertainty, bins=30, alpha=0.7, label='Correct', color='green')
    axes[0].hist(incorrect_uncertainty, bins=30, alpha=0.7, label='Incorrect', color='red')
    axes[0].set_xlabel('MC-Dropout Uncertainty')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Uncertainty Distribution')
    axes[0].legend()
    
    # Plot 2: Box plot
    axes[1].boxplot([correct_uncertainty, incorrect_uncertainty], labels=['Correct', 'Incorrect'])
    axes[1].set_ylabel('MC-Dropout Uncertainty')
    axes[1].set_title('Uncertainty by Correctness')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'uncertainty_analysis.png'), dpi=150)
    plt.close()
    
    print(f"Uncertainty analysis saved to {save_dir}")

def test_epoch_mc_dropout(
    model,
    loader: DataLoader,
    epoch: int,
    acc_func,
    loss_func,
    args,
    num_mc_samples: int = 20,
    logger=None,
) -> tuple[float, dict, np.ndarray, np.ndarray, np.ndarray]:
    """
    Testing con MC-Dropout per stimare incertezza.
    
    Returns:
        tuple: (avg_loss, avg_accuracy, per_class_accuracy_dict, 
                confusion_matrix, uncertainties)
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
    
    # Liste per confusion matrix e incertezze
    all_preds = []
    all_targets = []
    all_uncertainties = []
    all_confidences = []
    
    # Cache per attributi args
    is_main_process = getattr(args, "rank", 0) == 0
    sw_batch_size = getattr(args, 'sw_batch_size', 4)
    
    # Riattiva MC-Dropout
    enable_mc_dropout(model)
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Estrai data e target
            if isinstance(batch_data, list):
                data, target = batch_data
            else:
                data, target = batch_data["vol"], batch_data["label"]

            data = data.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)
            
            # Assicura single channel
            data = ensure_single_channel(data, mode="first")  # [B,1,D,H,W]
            B = data.shape[0]
            
            # ============================================
            # STEP 1: Estrai patch
            # ============================================
            all_patches = []
            all_coords = []
            patches_per_sample = []
            
            for b in range(B):
                vol = data[b:b+1]
                patches, coords = extract_patches_5d_torch(
                    vol,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=args.step,
                    pad_value=0
                )
                all_patches.append(patches)
                all_coords.extend(coords)
                patches_per_sample.append(patches.shape[0])
            
            all_patches = torch.cat(all_patches, dim=0).to(torch.float32)
            total_patches = all_patches.shape[0]
            
            # ============================================
            # STEP 2: Batch processing patch
            # ============================================
            all_feats = []
            
            for i in range(0, total_patches, sw_batch_size):
                end_idx = min(i + sw_batch_size, total_patches)
                batch_patches = all_patches[i:end_idx]
                feats, _ = model.forward_features(batch_patches)
                all_feats.append(feats)
            
            all_feats = torch.cat(all_feats, dim=0)
            
            # ============================================
            # STEP 3: MC-Dropout inference per ogni sample
            # ============================================
            batch_logits = []
            batch_uncertainties = []
            batch_confidences = []
            
            start_idx = 0
            for b in range(B):
                num_patches = patches_per_sample[b]
                end_idx = start_idx + num_patches
                
                b_feats = all_feats[start_idx:end_idx]
                b_coords = all_coords[start_idx:end_idx]
                
                # Ricostruisci con blending
                feats_tiled = tile_with_gaussian_blending(
                    b_feats,
                    b_coords,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=args.step
                )
                
                # MC-Dropout: forward multipli
                all_probs = []
                
                for _ in range(num_mc_samples):
                    pooled = model.global_pool(feats_tiled)
                    
                    if args.model_name == "swinunetr+ml_decoder":
                        pooled = pooled.flatten(2)
                    else:
                        pooled = pooled.flatten(1)
                    
                    # ✅ Dropout attivo (forzato)
                    pooled = model.dropout_head(pooled)
                    logits = model.fc(pooled)  # [1, num_classes]
                    probs = torch.softmax(logits, dim=1)
                    all_probs.append(probs)
                
                # Statistiche MC-Dropout
                all_probs = torch.stack(all_probs, dim=0)  # [num_samples, 1, num_classes]
                mean_probs = all_probs.mean(dim=0)  # [1, num_classes]
                std_probs = all_probs.std(dim=0)    # [1, num_classes]
                
                # Logits dalla media
                mean_logits = torch.log(mean_probs + 1e-8)
                
                # Incertezza: entropia o std media
                uncertainty = std_probs.mean(dim=1)  # [1]
                confidence = mean_probs.max(dim=1)[0]  # [1]
                
                batch_logits.append(mean_logits)
                batch_uncertainties.append(uncertainty)
                batch_confidences.append(confidence)
                
                start_idx = end_idx
            
            logits = torch.cat(batch_logits, dim=0)  # [B, num_classes]
            uncertainties = torch.cat(batch_uncertainties, dim=0)  # [B]
            confidences = torch.cat(batch_confidences, dim=0)  # [B]
            
            loss = loss_func(logits, target)
            
            # ============================================
            # STEP 4: Metriche
            # ============================================
            
            if num_classes is None:
                num_classes = logits.shape[1]
                per_class_correct = np.zeros(num_classes, dtype=np.int64)
                per_class_total = np.zeros(num_classes, dtype=np.int64)
            
            # Predizioni
            probs = torch.softmax(logits, dim=1)
            preds = probs.argmax(dim=1)
            target_eval = target.view(-1) if target.ndim > 1 else target
            
            all_preds.append(preds.cpu())
            all_targets.append(target_eval.cpu())
            all_uncertainties.append(uncertainties.cpu())
            all_confidences.append(confidences.cpu())
            
            # Accuracy
            correct = (preds == target_eval).sum().item()
            not_nans = target_eval.numel()
            
            if acc_func is not None:
                acc = float(acc_func(logits, target_eval))
            else:
                acc = correct / max(1, not_nans)
            
            # Per-class accuracy
            t_cpu = target_eval.cpu().numpy()
            p_cpu = preds.cpu().numpy()
            mask = (p_cpu == t_cpu)
            batch_total = np.bincount(t_cpu, minlength=num_classes)
            batch_correct = np.bincount(t_cpu[mask], minlength=num_classes)
            
            per_class_correct += batch_correct
            per_class_total += batch_total
            run_acc.update(acc, n=not_nans)
            run_loss.update(loss.item(), n=not_nans)
            
            # Logging
            if is_main_process:
                print(
                    f"MC-Dropout Testing {idx+1}/{len(loader)}, "
                    f"Acc: {run_acc.avg:.4f}, Loss: {run_loss.avg:.4f}, "
                    f"time {time.time() - start_time:.2f}s"
                )
            start_time = time.time()
    
    # Disattiva MC-Dropout
    disable_mc_dropout(model)
    
    # Gestisci caso senza batch
    if num_classes is None:
        return float('nan'), {}, np.array([]), np.array([])
    
    # Calcola accuracy per classe
    per_class_acc = {
        int(c): float(per_class_correct[c]) / max(1, int(per_class_total[c]))
        for c in range(num_classes)
    }
    
    # Confusion matrix
    all_preds = torch.cat(all_preds, dim=0).numpy()
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_uncertainties = torch.cat(all_uncertainties, dim=0).numpy()
    
    cm = confusion_matrix(all_targets, all_preds, labels=np.arange(num_classes))

    plot_uncertainty_analysis(
        uncertainties=all_uncertainties,
        targets=all_targets,
        preds=all_preds,
        save_dir=args.final_output_dir 
    )
    
    # Stampa riassunto
    if is_main_process:
        if logger:
            logger.info("")
            logger.info("=" * 80)
            logger.info("MC-DROPOUT UNCERTAINTY ANALYSIS")
            logger.info("=" * 80)
            
            # 1. Correttezza vs Incertezza
            correct_mask = (all_preds == all_targets)
            correct_unc = all_uncertainties[correct_mask]
            incorrect_unc = all_uncertainties[~correct_mask]
            
            logger.info(f"Correct predictions:   mean_unc={correct_unc.mean():.4f} ± {correct_unc.std():.4f}")
            logger.info(f"Incorrect predictions: mean_unc={incorrect_unc.mean():.4f} ± {incorrect_unc.std():.4f}")
            logger.info(f"Uncertainty gap:       {incorrect_unc.mean() - correct_unc.mean():.4f}")
            logger.info("")
            
            # 2. Incertezza per classe
            logger.info("Per-class uncertainty:")
            for c in range(num_classes):
                mask = (all_targets == c)
                if mask.sum() > 0:
                    class_unc = all_uncertainties[mask]
                    logger.info(f"  Class {c}: {class_unc.mean():.4f} ± {class_unc.std():.4f} (n={mask.sum()})")
            logger.info("")
            
            # 3. Rejection analysis
            logger.info("Selective prediction (rejection thresholds):")
            for percentile in [50, 75, 90, 95]:
                thresh = np.percentile(all_uncertainties, percentile)
                keep_mask = all_uncertainties <= thresh
                if keep_mask.sum() > 0:
                    remaining_acc = (all_preds[keep_mask] == all_targets[keep_mask]).mean()
                    rejected = (~keep_mask).sum()
                    logger.info(f"  Reject top {100-percentile}% (>{thresh:.4f}): "
                        f"keep {keep_mask.sum()}/{len(all_preds)} samples, acc={remaining_acc:.4f}")
            logger.info("")
            
            # 4. Calibration bins
            logger.info("Calibration (accuracy by uncertainty bin):")
            bins = np.percentile(all_uncertainties, [0, 25, 50, 75, 100])
            bin_labels = ['Low (0-25%)', 'Medium (25-50%)', 'High (50-75%)', 'Very High (75-100%)']
            
            for i in range(len(bins) - 1):
                mask = (all_uncertainties >= bins[i]) & (all_uncertainties < bins[i+1])
                if mask.sum() > 0:
                    bin_acc = (all_preds[mask] == all_targets[mask]).mean()
                    logger.info(f"  {bin_labels[i]:20} [{bins[i]:.4f}, {bins[i+1]:.4f}): "
                        f"acc={bin_acc:.4f}, n={mask.sum()}")
            logger.info("")
            
            # 5. Hard cases (top-10 più incerti)
            logger.info("Top 10 most uncertain samples:")
            top_uncertain_idx = np.argsort(all_uncertainties)[-10:]
            for rank, idx in enumerate(top_uncertain_idx[::-1], 1):
                logger.info(f"  {rank}. Sample {idx}: unc={all_uncertainties[idx]:.4f}, "
                    f"pred={all_preds[idx]}, true={all_targets[idx]}, "
                    f"{'✓' if all_preds[idx] == all_targets[idx] else '✗'}")
            
            logger.info("=" * 80)
            logger.info("")




    return float(run_loss.avg), float(run_acc.avg), per_class_acc, cm, all_uncertainties

# ============================================
# Helper: AverageMeter (se non lo hai già)
# ============================================

class AverageMeter:
    """Computa e memorizza la media e la varianza corrente"""
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
        self.avg = self.sum / self.count
