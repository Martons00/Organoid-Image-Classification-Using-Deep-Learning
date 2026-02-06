#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test Adaptive Slice Selection on 5 samples with visualization
"""

import os
import argparse
import numpy as np
from pathlib import Path
from skimage import io
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.preprocessing import StandardScaler


class SliceSelector:
    """Selects most representative slices from 3D volume"""
    
    def __init__(self, method='feature_variance'):
        self.method = method
        
    def compute_slice_features(self, volume):
        """Compute feature vector for each slice"""
        D = volume.shape[0]
        features = []
        
        for z in range(D):
            slice_2d = volume[z]
            
            # Statistical features
            feat = [
                np.mean(slice_2d),
                np.std(slice_2d),
                np.median(slice_2d),
                np.percentile(slice_2d, 25),
                np.percentile(slice_2d, 75),
            ]
            
            # Texture features (gradient magnitude)
            grad_x = np.gradient(slice_2d, axis=0)
            grad_y = np.gradient(slice_2d, axis=1)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            feat.extend([
                np.mean(grad_mag),
                np.std(grad_mag),
                np.max(grad_mag)
            ])
            
            # Entropy
            hist, _ = np.histogram(slice_2d.flatten(), bins=256, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            feat.append(entropy)
            
            features.append(feat)
        
        return np.array(features)
    
    def compute_ssim_matrix(self, volume):
        """Compute pairwise SSIM between all slices"""
        D = volume.shape[0]
        ssim_matrix = np.zeros((D, D))
        
        for i in range(D):
            for j in range(i, D):
                if i == j:
                    ssim_matrix[i, j] = 1.0
                else:
                    slice_i = volume[i] / (volume[i].max() + 1e-8)
                    slice_j = volume[j] / (volume[j].max() + 1e-8)
                    score = ssim(slice_i, slice_j, data_range=1.0)
                    ssim_matrix[i, j] = score
                    ssim_matrix[j, i] = score
        
        return ssim_matrix
    
    def greedy_dissimilarity_selection(self, similarity_matrix, n_slices):
        """Greedy selection of most dissimilar slices with spatial coverage"""
        D = similarity_matrix.shape[0]
        
        # Ensure spatial coverage: divide into regions
        n_regions = min(n_slices // 4, 8)
        region_size = D // n_regions
        selected = []
        
        for r in range(n_regions):
            start = r * region_size
            end = (r + 1) * region_size if r < n_regions - 1 else D
            
            # Select slice with lowest average similarity in this region
            region_avg_sim = np.mean(similarity_matrix[start:end, :], axis=1)
            local_idx = np.argmin(region_avg_sim)
            selected.append(start + local_idx)
        
        # Greedy selection for remaining slices
        dissimilarity_matrix = 1 - similarity_matrix
        
        while len(selected) < n_slices:
            min_sim_to_selected = np.min(dissimilarity_matrix[:, selected], axis=1)
            min_sim_to_selected[selected] = -np.inf
            next_idx = np.argmax(min_sim_to_selected)
            selected.append(next_idx)
        
        return sorted(selected)
    
    def select_by_method(self, volume, n_slices=32):
        """Select representative slices using specified method"""
        D = volume.shape[0]
        
        if n_slices >= D:
            return list(range(D))
        
        if self.method == 'feature_variance':
            features = self.compute_slice_features(volume)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            distances = squareform(pdist(features_scaled, metric='euclidean'))
            max_dist = np.max(distances)
            similarity = 1 - (distances / (max_dist + 1e-8))
            selected = self.greedy_dissimilarity_selection(similarity, n_slices)
            
        elif self.method == 'ssim_dissimilarity':
            similarity = self.compute_ssim_matrix(volume)
            selected = self.greedy_dissimilarity_selection(similarity, n_slices)
            
        elif self.method == 'entropy':
            entropies = []
            for z in range(D):
                hist, _ = np.histogram(volume[z].flatten(), bins=256, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropies.append(entropy)
            
            n_regions = min(n_slices // 4, 8)
            region_size = D // n_regions
            selected = []
            
            for r in range(n_regions):
                start = r * region_size
                end = (r + 1) * region_size if r < n_regions - 1 else D
                n_from_region = n_slices // n_regions + (1 if r < n_slices % n_regions else 0)
                region_entropies = entropies[start:end]
                top_indices = np.argsort(region_entropies)[-n_from_region:]
                selected.extend([start + idx for idx in top_indices])
            
            selected = sorted(selected)[:n_slices]
            
        elif self.method == 'gradient':
            grad_magnitudes = []
            for z in range(D):
                grad_x = np.gradient(volume[z], axis=0)
                grad_y = np.gradient(volume[z], axis=1)
                grad_mag = np.sqrt(grad_x**2 + grad_y**2)
                grad_magnitudes.append(np.mean(grad_mag))
            
            n_regions = min(n_slices // 4, 8)
            region_size = D // n_regions
            selected = []
            
            for r in range(n_regions):
                start = r * region_size
                end = (r + 1) * region_size if r < n_regions - 1 else D
                n_from_region = n_slices // n_regions + (1 if r < n_slices % n_regions else 0)
                region_grads = grad_magnitudes[start:end]
                top_indices = np.argsort(region_grads)[-n_from_region:]
                selected.extend([start + idx for idx in top_indices])
            
            selected = sorted(selected)[:n_slices]
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return selected


def visualize_selected_slices(samples_data, n_slices=32, save_path=None):
    """
    Visualize selected slices for multiple samples in a single figure
    
    Args:
        samples_data: list of dicts with keys 'name', 'volume', 'selected_indices'
        n_slices: number of slices per sample
        save_path: path to save figure (optional)
    """
    n_samples = len(samples_data)
    
    # Create figure with subplots: rows=samples, cols=slices
    fig, axes = plt.subplots(n_samples, n_slices, figsize=(n_slices * 1.5, n_samples * 1.5))
    
    # Handle single sample case
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for sample_idx, sample in enumerate(samples_data):
        volume = sample['volume']
        selected_indices = sample['selected_indices']
        sample_name = sample['name']
        
        for slice_idx, z_idx in enumerate(selected_indices):
            ax = axes[sample_idx, slice_idx]
            
            # Display slice
            slice_2d = volume[z_idx]
            ax.imshow(slice_2d, cmap='gray', aspect='auto')
            ax.axis('off')
            
            # Add title only for first row (slice indices)
            if sample_idx == 0:
                ax.set_title(f'Z={z_idx}', fontsize=8)
        
        # Add sample name on the left
        axes[sample_idx, 0].text(-0.1, 0.5, sample_name, 
                                  transform=axes[sample_idx, 0].transAxes,
                                  fontsize=10, fontweight='bold',
                                  verticalalignment='center',
                                  horizontalalignment='right',
                                  rotation=0)
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0.02, hspace=0.05)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figure saved to {save_path}")
    
    plt.show()


def test_on_samples(input_dir, n_samples=5, n_slices=32, method='feature_variance', save_path=None):
    """
    Test slice selection on n_samples and visualize results
    
    Args:
        input_dir: directory containing TIFF files
        n_samples: number of samples to process
        n_slices: number of slices to select per sample
        method: selection method
        save_path: path to save visualization
    """
    input_path = Path(input_dir)
    
    # Find TIFF files
    tif_files = list(input_path.rglob("*.tif")) + list(input_path.rglob("*.tiff"))
    
    if len(tif_files) == 0:
        raise ValueError(f"No TIFF files found in {input_dir}")
    
    # Select first n_samples
    tif_files = tif_files[:n_samples]
    
    print(f"Processing {len(tif_files)} samples")
    print(f"Method: {method}")
    print(f"Target slices: {n_slices}")
    print("-" * 50)
    
    selector = SliceSelector(method=method)
    samples_data = []
    
    for tif_path in tif_files:
        print(f"Processing: {tif_path.name}")
        
        # Load volume
        volume = io.imread(str(tif_path))
        
        # Handle different shapes
        if volume.ndim == 4:
            volume = volume[..., 0]  # Take first channel
        
        D, H, W = volume.shape
        print(f"  Original shape: {volume.shape}")
        
        # Select slices
        if D <= n_slices:
            selected_indices = list(range(D))
            print(f"  Warning: Volume has only {D} slices, using all")
        else:
            selected_indices = selector.select_by_method(volume, n_slices)
            print(f"  Selected {len(selected_indices)} slices: {selected_indices[:5]}...{selected_indices[-5:]}")
        
        samples_data.append({
            'name': tif_path.stem,
            'volume': volume,
            'selected_indices': selected_indices
        })
    
    print("-" * 50)
    print("Creating visualization...")
    
    # Visualize
    visualize_selected_slices(samples_data, n_slices, save_path)


def main():
    parser = argparse.ArgumentParser(description='Test adaptive slice selection on samples')
    parser.add_argument('--input_dir', type=str, required=True,
                        help='Directory containing TIFF files')
    parser.add_argument('--n_samples', type=int, default=4,
                        help='Number of samples to process (default: 5)')
    parser.add_argument('--n_slices', type=int, default=8,
                        help='Number of slices to select (default: 32)')
    parser.add_argument('--method', type=str, default='feature_variance',
                        choices=['feature_variance', 'ssim_dissimilarity', 'entropy', 'gradient'],
                        help='Selection method (default: feature_variance)')
    parser.add_argument('--save_path', type=str, default='slice_selection_visualization.png',
                        help='Path to save visualization (default: slice_selection_visualization.png)')
    
    args = parser.parse_args()
    
    test_on_samples(
        input_dir=args.input_dir,
        n_samples=args.n_samples,
        n_slices=args.n_slices,
        method=args.method,
        save_path=args.save_path
    )


if __name__ == "__main__":
    main()
