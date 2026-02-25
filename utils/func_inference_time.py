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
import time
import torch
import psutil
import os
from pathlib import Path
from models.SpatialAbstracter import SpatialAbstracter

# Local imports - Utilss
from .utils import (
    AverageMeter,
    distributed_all_gather,
    extract_patches_5d_torch,
    ensure_single_channel,
    tile_feature_patches,
    tile_with_gaussian_blending,
)

def test_metrics_inference_pm(
    model,
    loader: DataLoader,
    args,
    num_samples: int = 20,  # Numero di sample da processare
    warmup_batches: int = 2   # Batch di warm-up da scartare
) -> dict:
    """
    Misura efficienza computazionale durante l'inferenza con patch matching.
    
    Args:
        model: Il modello da testare
        loader: DataLoader del test set
        args: Argomenti di configurazione
        num_samples: Numero di sample da processare (default 100)
        warmup_batches: Numero di batch di warm-up da scartare (default 2)
    
    Returns:
        dict: Metriche di efficienza computazionale
    """
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    # Cache attributi args
    sw_batch_size = getattr(args, 'sw_batch_size', 4)
    is_main_process = getattr(args, "rank", 0) == 0
    spatial_abstracter = SpatialAbstracter((256, 256))
    
    # ============================================
    # 1. STORAGE REQUIREMENTS
    # ============================================
    model_size_mb = 0
    if is_main_process: 
        temp_path = "/tmp/temp_model.pth"
        torch.save(model.state_dict(), temp_path)
        model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)  # MB
        os.remove(temp_path)
    
    # ============================================
    # 2. WARM-UP
    # ============================================
    if is_main_process:
        print(f"Running {warmup_batches} warm-up batches...")
    
    warmup_count = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if warmup_count >= warmup_batches:
                break
            
            if isinstance(batch_data, list):
                data, _ = batch_data
            else:
                data = batch_data["vol"]
            
            data = data.to(device, non_blocking=True)
            data = ensure_single_channel(data, mode="first")
            print(f"Processing warm-up batch {warmup_count+1} with {data.shape[0]} samples...")
            data = spatial_abstracter(data)
            print(f"After spatial abstraction, shape: {data.shape}")
            B = data.shape[0]
            
            # Inferenza di warm-up (stesso pipeline con patch matching)
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
            
            all_feats = []
            for i in range(0, total_patches, sw_batch_size):
                end_idx = min(i + sw_batch_size, total_patches)
                batch_patches = all_patches[i:end_idx]
                feats, _ = model.forward_features(batch_patches)
                all_feats.append(feats)
            
            all_feats = torch.cat(all_feats, dim=0)
            
            start_idx = 0
            for b in range(B):
                num_patches = patches_per_sample[b]
                end_idx = start_idx + num_patches
                b_feats = all_feats[start_idx:end_idx]
                b_coords = all_coords[start_idx:end_idx]
                feats_tiled = tile_with_gaussian_blending(
                    b_feats,
                    b_coords,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=args.step
                )
                pooled = model.global_pool(feats_tiled)
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr" or "resnet" in args.model_name or "densenet" in args.model_name or "swinvit" in args.model_name:
                    pooled = pooled.flatten(1)
                logits_b = model.fc(pooled)
                start_idx = end_idx
            
            warmup_count += 1
            
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
    
    if is_main_process:
        print("Warm-up complete. Starting measurements...")
    
    # ============================================
    # 3. MEASUREMENT PHASE
    # ============================================
    inference_times = []
    memory_usage = []
    num_patches_per_sample = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    
    sample_count = 0
    batch_idx = 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Salta i batch già usati per warm-up
            batch_start_time = time.time()
            if idx < warmup_batches:
                continue
            
            if sample_count >= num_samples:
                break
            
            if isinstance(batch_data, list):
                data, _ = batch_data
            else:
                data = batch_data["vol"]
            
            data = data.to(device, non_blocking=True)
            data = ensure_single_channel(data, mode="first")
            B = data.shape[0]
            
            # Sincronizza prima della misurazione
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
            print(f"Processing warm-up batch {warmup_count+1} with {data.shape[0]} samples...")
            data = spatial_abstracter(data)
            print(f"After spatial abstraction, shape: {data.shape}")
            
            # ============================================
            # INFERENZA con PATCH MATCHING
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
                num_patches_per_sample.append(patches.shape[0])
            
            all_patches = torch.cat(all_patches, dim=0).to(torch.float32)
            total_patches = all_patches.shape[0]
            
            # Batch processing patches
            all_feats = []
            for i in range(0, total_patches, sw_batch_size):
                end_idx = min(i + sw_batch_size, total_patches)
                batch_patches = all_patches[i:end_idx]
                feats, _ = model.forward_features(batch_patches)
                all_feats.append(feats)
            
            all_feats = torch.cat(all_feats, dim=0)
            
            # Ricostruzione con Gaussian blending
            batch_logits = []
            start_idx = 0
            for b in range(B):
                num_patches = patches_per_sample[b]
                end_idx = start_idx + num_patches
                
                b_feats = all_feats[start_idx:end_idx]
                b_coords = all_coords[start_idx:end_idx]
                
                feats_tiled = tile_with_gaussian_blending(
                    b_feats,
                    b_coords,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=args.step
                )
                
                pooled = model.global_pool(feats_tiled)
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr" or "resnet" in args.model_name or "densenet" in args.model_name or "swinvit" in args.model_name:
                    pooled = pooled.flatten(1)
                
                logits_b = model.fc(pooled)
                batch_logits.append(logits_b)
                start_idx = end_idx
            
            logits = torch.cat(batch_logits, dim=0)
            
            # Sincronizza dopo l'inferenza
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
            batch_time = time.time() - batch_start_time
            time_per_sample = batch_time / B
            inference_times.append(time_per_sample)
            
            # Misura memoria GPU
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
                memory_usage.append(mem_allocated)
                torch.cuda.reset_peak_memory_stats(device)
            
            sample_count += B
            batch_idx += 1
            
            if is_main_process:
                avg_patches = np.mean(patches_per_sample)
                print(f"Batch {batch_idx}: {sample_count}/{num_samples} samples, "
                      f"time/sample: {time_per_sample*1000:.2f}ms, "
                      f"memory: {mem_allocated:.1f}MB, "
                      f"patches: {avg_patches:.1f}")
    
    # ============================================
    # 4. CALCOLA STATISTICHE
    # ============================================
    metrics = {}
    
    if inference_times:
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        std_inference_time = np.std(inference_times) * 1000
        median_inference_time = np.median(inference_times) * 1000
        min_inference_time = np.min(inference_times) * 1000
        max_inference_time = np.max(inference_times) * 1000
        throughput = 1.0 / np.mean(inference_times)  # samples/sec
        
        metrics['avg_inference_time_ms'] = float(avg_inference_time)
        metrics['std_inference_time_ms'] = float(std_inference_time)
        metrics['median_inference_time_ms'] = float(median_inference_time)
        metrics['min_inference_time_ms'] = float(min_inference_time)
        metrics['max_inference_time_ms'] = float(max_inference_time)
        metrics['throughput_samples_per_sec'] = float(throughput)
    
    if memory_usage and torch.cuda.is_available():
        avg_memory = np.mean(memory_usage)
        std_memory = np.std(memory_usage)
        peak_memory = np.max(memory_usage)
        
        metrics['avg_memory_mb'] = float(avg_memory)
        metrics['std_memory_mb'] = float(std_memory)
        metrics['peak_memory_mb'] = float(peak_memory)
    else:
        metrics['avg_memory_mb'] = 0.0
        metrics['std_memory_mb'] = 0.0
        metrics['peak_memory_mb'] = 0.0
    
    metrics['model_size_mb'] = float(model_size_mb)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics['total_parameters'] = int(total_params)
        metrics['trainable_parameters'] = int(trainable_params)
        metrics['parameters_millions'] = float(total_params / 1e6)
    
    # Statistiche specifiche patch matching
    if num_patches_per_sample:
        metrics['avg_patches_per_sample'] = float(np.mean(num_patches_per_sample))
        metrics['std_patches_per_sample'] = float(np.std(num_patches_per_sample))
        metrics['max_patches_per_sample'] = int(np.max(num_patches_per_sample))
        metrics['min_patches_per_sample'] = int(np.min(num_patches_per_sample))
    
    # Stampa summary
    if is_main_process:
        print("\n" + "="*60)
        print("COMPUTATIONAL EFFICIENCY METRICS (Patch Matching)")
        print("="*60)
        print(f"Inference Time (excluding warm-up):")
        print(f"  - Mean: {metrics.get('avg_inference_time_ms', 0):.2f} ± {metrics.get('std_inference_time_ms', 0):.2f} ms/sample")
        print(f"  - Median: {metrics.get('median_inference_time_ms', 0):.2f} ms/sample")
        print(f"  - Range: [{metrics.get('min_inference_time_ms', 0):.2f}, {metrics.get('max_inference_time_ms', 0):.2f}] ms")
        print(f"  - Throughput: {metrics.get('throughput_samples_per_sec', 0):.2f} samples/sec")
        print(f"\nMemory Footprint:")
        print(f"  - Mean: {metrics.get('avg_memory_mb', 0):.2f} ± {metrics.get('std_memory_mb', 0):.2f} MB")
        print(f"  - Peak: {metrics.get('peak_memory_mb', 0):.2f} MB")
        print(f"\nStorage Requirements:")
        print(f"  - Model size: {metrics.get('model_size_mb', 0):.2f} MB")
        print(f"  - Total parameters: {metrics.get('parameters_millions', 0):.2f}M")
        print(f"\nPatch Statistics:")
        print(f"  - Avg patches/sample: {metrics.get('avg_patches_per_sample', 0):.1f} ± {metrics.get('std_patches_per_sample', 0):.1f}")
        print(f"  - Range: [{metrics.get('min_patches_per_sample', 0)}, {metrics.get('max_patches_per_sample', 0)}]")
        print("="*60 + "\n")
        
        # Salva summary su file
        output_file = Path(args.final_output_dir) / "inference_metrics_pm.txt"
        with open(output_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("COMPUTATIONAL EFFICIENCY METRICS (Patch Matching)\n")
            f.write("="*60 + "\n")
            f.write(f"Inference Time (excluding warm-up):\n")
            f.write(f"  - Mean: {metrics.get('avg_inference_time_ms', 0):.2f} ± {metrics.get('std_inference_time_ms', 0):.2f} ms/sample\n")
            f.write(f"  - Median: {metrics.get('median_inference_time_ms', 0):.2f} ms/sample\n")
            f.write(f"  - Range: [{metrics.get('min_inference_time_ms', 0):.2f}, {metrics.get('max_inference_time_ms', 0):.2f}] ms\n")
            f.write(f"  - Throughput: {metrics.get('throughput_samples_per_sec', 0):.2f} samples/sec\n")
            f.write(f"\nMemory Footprint:\n")
            f.write(f"  - Mean: {metrics.get('avg_memory_mb', 0):.2f} ± {metrics.get('std_memory_mb', 0):.2f} MB\n")
            f.write(f"  - Peak: {metrics.get('peak_memory_mb', 0):.2f} MB\n")
            f.write(f"\nStorage Requirements:\n")
            f.write(f"  - Model size: {metrics.get('model_size_mb', 0):.2f} MB\n")
            f.write(f"  - Total parameters: {metrics.get('parameters_millions', 0):.2f}M\n")
            f.write(f"\nPatch Statistics:\n")
            f.write(f"  - Avg patches/sample: {metrics.get('avg_patches_per_sample', 0):.1f} ± {metrics.get('std_patches_per_sample', 0):.1f}\n")
            f.write(f"  - Range: [{metrics.get('min_patches_per_sample', 0)}, {metrics.get('max_patches_per_sample', 0)}]\n")
            f.write("="*60 + "\n")
    
    return metrics


def test_metrics_inference(
    model,
    loader: DataLoader,
    args,
    num_samples: int = 20,  # Numero limitato di sample per test
    warmup_batches: int = 1  # Numero di batch di warm-up da scartare
) -> dict:
    """
    Misura efficienza computazionale durante l'inferenza.
    
    Args:
        model: Il modello da testare
        loader: DataLoader del test set
        args: Argomenti di configurazione
        num_samples: Numero di sample da processare (default 100)
        warmup_batches: Numero di batch di warm-up da scartare (default 2)
    
    Returns:
        dict: Metriche di efficienza computazionale
    """
    model.eval()
    device = torch.device("cuda", args.rank) if torch.cuda.is_available() else torch.device("cpu")
    
    # Cache attributi args
    sw_batch_size = getattr(args, 'sw_batch_size', 4)
    is_main_process = getattr(args, "rank", 0) == 0
    spatial_abstracter = SpatialAbstracter((128,128))
    
    # ============================================
    # 1. STORAGE REQUIREMENTS
    # ============================================
    model_size_mb = 0
    if is_main_process:
        temp_path = "/tmp/temp_model.pth"
        torch.save(model.state_dict(), temp_path)
        model_size_mb = os.path.getsize(temp_path) / (1024 * 1024)  # MB
        os.remove(temp_path)
    
    # ============================================
    # 2. WARM-UP
    # ============================================
    if is_main_process:
        print(f"Running {warmup_batches} warm-up batches...")
    
    warmup_count = 0
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            if warmup_count >= warmup_batches:
                break
            
            if isinstance(batch_data, list):
                data, _ = batch_data
            else:
                data = batch_data["vol"]
            
            data = data.to(device, non_blocking=True)
            data = ensure_single_channel(data, mode="first")
            print(f"Processing warm-up batch {warmup_count+1} with {data.shape[0]} samples...")
            print(f"Original shape: {data.shape}")
            data = spatial_abstracter(data)
            print(f"After spatial abstraction, shape: {data.shape}")
            B = data.shape[0]
            
            # Inferenza di warm-up (stesso pipeline)
            all_patches = []
            all_coords = []
            patches_per_sample = []
            
            for b in range(B):
                vol = data[b:b+1]
                patches, coords = extract_patches_5d_torch(
                    vol,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=(args.roi_z, args.roi_y, args.roi_x),
                    pad_value=0
                )
                all_patches.append(patches)
                all_coords.extend(coords)
                patches_per_sample.append(patches.shape[0])
            
            all_patches = torch.cat(all_patches, dim=0).to(torch.float32)
            total_patches = all_patches.shape[0]
            
            all_feats = []
            for i in range(0, total_patches, sw_batch_size):
                end_idx = min(i + sw_batch_size, total_patches)
                batch_patches = all_patches[i:end_idx]
                feats, _ = model.forward_features(batch_patches)
                all_feats.append(feats)
            
            all_feats = torch.cat(all_feats, dim=0)
            
            start_idx = 0
            for b in range(B):
                num_patches = patches_per_sample[b]
                end_idx = start_idx + num_patches
                b_feats = all_feats[start_idx:end_idx]
                b_coords = all_coords[start_idx:end_idx]
                feats_tiled = tile_feature_patches(b_feats, coords=b_coords)
                pooled = model.global_pool(feats_tiled)
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr" or "resnet" in args.model_name or "densenet" in args.model_name or "swinvit" in args.model_name:
                    pooled = pooled.flatten(1)
                logits_b = model.fc(pooled)
                start_idx = end_idx
            
            warmup_count += 1
            
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
    
    if is_main_process:
        print("Warm-up complete. Starting measurements...")
    
    # ============================================
    # 3. MEASUREMENT PHASE
    # ============================================
    inference_times = []
    memory_usage = []
    
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    
    sample_count = 0
    batch_idx = 0
    
    with torch.no_grad():
        for idx, batch_data in enumerate(loader):
            # Salta i batch già usati per warm-up
            batch_start_time = time.time()
            if idx < warmup_batches:
                continue
            
            if sample_count >= num_samples:
                break
            
            if isinstance(batch_data, list):
                data, _ = batch_data
            else:
                data = batch_data["vol"]
            
            data = data.to(device, non_blocking=True)
            data = ensure_single_channel(data, mode="first")
            B = data.shape[0]
            
            # Sincronizza prima della misurazione
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
            print(f"Processing warm-up batch {warmup_count+1} with {data.shape[0]} samples...")
            print(f"Original shape: {data.shape}")
            data = spatial_abstracter(data)
            print(f"After spatial abstraction, shape: {data.shape}")
            
            # ============================================
            # INFERENZA (stesso pipeline)
            # ============================================
            all_patches = []
            all_coords = []
            patches_per_sample = []
            
            for b in range(B):
                vol = data[b:b+1]
                patches, coords = extract_patches_5d_torch(
                    vol,
                    patch_size=(args.roi_z, args.roi_y, args.roi_x),
                    step=(args.roi_z, args.roi_y, args.roi_x),
                    pad_value=0
                )
                all_patches.append(patches)
                all_coords.extend(coords)
                patches_per_sample.append(patches.shape[0])
            
            all_patches = torch.cat(all_patches, dim=0).to(torch.float32)
            total_patches = all_patches.shape[0]
            
            all_feats = []
            for i in range(0, total_patches, sw_batch_size):
                end_idx = min(i + sw_batch_size, total_patches)
                batch_patches = all_patches[i:end_idx]
                feats, _ = model.forward_features(batch_patches)
                all_feats.append(feats)
            
            all_feats = torch.cat(all_feats, dim=0)
            
            batch_logits = []
            start_idx = 0
            for b in range(B):
                num_patches = patches_per_sample[b]
                end_idx = start_idx + num_patches
                b_feats = all_feats[start_idx:end_idx]
                b_coords = all_coords[start_idx:end_idx]
                feats_tiled = tile_feature_patches(b_feats, coords=b_coords)
                pooled = model.global_pool(feats_tiled)
                if args.model_name == "swinunetr+ml_decoder":
                    pooled = pooled.flatten(2)
                elif args.model_name == "swinunetr" or "resnet" in args.model_name or "densenet" in args.model_name or "swinvit" in args.model_name:
                    pooled = pooled.flatten(1)
                logits_b = model.fc(pooled)
                batch_logits.append(logits_b)
                start_idx = end_idx
            
            logits = torch.cat(batch_logits, dim=0)
            
            # Sincronizza dopo l'inferenza
            if torch.cuda.is_available():
                torch.cuda.synchronize(device)
            
            batch_time = time.time() - batch_start_time
            time_per_sample = batch_time / B
            inference_times.append(time_per_sample)
            
            # Misura memoria GPU
            if torch.cuda.is_available():
                mem_allocated = torch.cuda.max_memory_allocated(device) / (1024 ** 2)  # MB
                memory_usage.append(mem_allocated)
                torch.cuda.reset_peak_memory_stats(device)
            
            sample_count += B
            batch_idx += 1
            
            if is_main_process:
                print(f"Batch {batch_idx}: {sample_count}/{num_samples} samples, "
                      f"time/sample: {time_per_sample*1000:.2f}ms, "
                      f"memory: {mem_allocated:.1f}MB")
    
    # ============================================
    # 4. CALCOLA STATISTICHE
    # ============================================
    metrics = {}
    
    if inference_times:
        avg_inference_time = np.mean(inference_times) * 1000  # ms
        std_inference_time = np.std(inference_times) * 1000
        median_inference_time = np.median(inference_times) * 1000
        min_inference_time = np.min(inference_times) * 1000
        max_inference_time = np.max(inference_times) * 1000
        throughput = 1.0 / np.mean(inference_times)  # samples/sec
        
        metrics['avg_inference_time_ms'] = float(avg_inference_time)
        metrics['std_inference_time_ms'] = float(std_inference_time)
        metrics['median_inference_time_ms'] = float(median_inference_time)
        metrics['min_inference_time_ms'] = float(min_inference_time)
        metrics['max_inference_time_ms'] = float(max_inference_time)
        metrics['throughput_samples_per_sec'] = float(throughput)
    
    if memory_usage and torch.cuda.is_available():
        avg_memory = np.mean(memory_usage)
        std_memory = np.std(memory_usage)
        peak_memory = np.max(memory_usage)
        
        metrics['avg_memory_mb'] = float(avg_memory)
        metrics['std_memory_mb'] = float(std_memory)
        metrics['peak_memory_mb'] = float(peak_memory)
    else:
        metrics['avg_memory_mb'] = 0.0
        metrics['std_memory_mb'] = 0.0
        metrics['peak_memory_mb'] = 0.0
    
    metrics['model_size_mb'] = float(model_size_mb)
    
    if is_main_process:
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        metrics['total_parameters'] = int(total_params)
        metrics['trainable_parameters'] = int(trainable_params)
        metrics['parameters_millions'] = float(total_params / 1e6)
    
    # Stampa summary
    if is_main_process:
        print("\n" + "="*60)
        print("COMPUTATIONAL EFFICIENCY METRICS")
        print("="*60)
        print(f"Inference Time (excluding warm-up):")
        print(f"  - Mean: {metrics.get('avg_inference_time_ms', 0):.2f} ± {metrics.get('std_inference_time_ms', 0):.2f} ms/sample")
        print(f"  - Median: {metrics.get('median_inference_time_ms', 0):.2f} ms/sample")
        print(f"  - Range: [{metrics.get('min_inference_time_ms', 0):.2f}, {metrics.get('max_inference_time_ms', 0):.2f}] ms")
        print(f"  - Throughput: {metrics.get('throughput_samples_per_sec', 0):.2f} samples/sec")
        print(f"\nMemory Footprint:")
        print(f"  - Mean: {metrics.get('avg_memory_mb', 0):.2f} ± {metrics.get('std_memory_mb', 0):.2f} MB")
        print(f"  - Peak: {metrics.get('peak_memory_mb', 0):.2f} MB")
        print(f"\nStorage Requirements:")
        print(f"  - Model size: {metrics.get('model_size_mb', 0):.2f} MB")
        print(f"  - Total parameters: {metrics.get('parameters_millions', 0):.2f}M")
        print("="*60 + "\n")

        # Salva summary su file
        output_file = Path(args.final_output_dir) / "inference_metrics_pm.txt"
        with open(output_file, "w") as f:
            f.write("="*60 + "\n")
            f.write("COMPUTATIONAL EFFICIENCY METRICS (Patch Matching)\n")
            f.write("="*60 + "\n")
            f.write(f"Inference Time (excluding warm-up):\n")
            f.write(f"  - Mean: {metrics.get('avg_inference_time_ms', 0):.2f} ± {metrics.get('std_inference_time_ms', 0):.2f} ms/sample\n")
            f.write(f"  - Median: {metrics.get('median_inference_time_ms', 0):.2f} ms/sample\n")
            f.write(f"  - Range: [{metrics.get('min_inference_time_ms', 0):.2f}, {metrics.get('max_inference_time_ms', 0):.2f}] ms\n")
            f.write(f"  - Throughput: {metrics.get('throughput_samples_per_sec', 0):.2f} samples/sec\n")
            f.write(f"\nMemory Footprint:\n")
            f.write(f"  - Mean: {metrics.get('avg_memory_mb', 0):.2f} ± {metrics.get('std_memory_mb', 0):.2f} MB\n")
            f.write(f"  - Peak: {metrics.get('peak_memory_mb', 0):.2f} MB\n")
            f.write(f"\nStorage Requirements:\n")
            f.write(f"  - Model size: {metrics.get('model_size_mb', 0):.2f} MB\n")
            f.write(f"  - Total parameters: {metrics.get('parameters_millions', 0):.2f}M\n")
            f.write(f"\nPatch Statistics:\n")
            f.write(f"  - Avg patches/sample: {metrics.get('avg_patches_per_sample', 0):.1f} ± {metrics.get('std_patches_per_sample', 0):.1f}\n")
            f.write(f"  - Range: [{metrics.get('min_patches_per_sample', 0)}, {metrics.get('max_patches_per_sample', 0)}]\n")
            f.write("="*60 + "\n")
    
    return metrics
