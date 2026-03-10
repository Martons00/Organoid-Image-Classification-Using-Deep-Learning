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
    """Selects most representative slices from 3D volume along any axis"""
    
    def __init__(self, method='feature_variance'):
        self.method = method
    
    def compute_slice_features(self, slices_2d):
        """Compute features for batch of 2D slices"""
        n_slices = slices_2d.shape[0]
        features = []
        
        for i in range(n_slices):
            slice_2d = slices_2d[i]
            feat = [
                np.mean(slice_2d), np.std(slice_2d), np.median(slice_2d),
                np.percentile(slice_2d, 25), np.percentile(slice_2d, 75),
            ]
            
            grad_x = np.gradient(slice_2d, axis=0)
            grad_y = np.gradient(slice_2d, axis=1)
            grad_mag = np.sqrt(grad_x**2 + grad_y**2)
            feat.extend([np.mean(grad_mag), np.std(grad_mag), np.max(grad_mag)])
            
            hist, _ = np.histogram(slice_2d.flatten(), bins=256, density=True)
            hist = hist[hist > 0]
            entropy = -np.sum(hist * np.log2(hist + 1e-10))
            feat.append(entropy)
            
            features.append(feat)
        
        return np.array(features)
    
    def compute_similarity_matrix(self, slices_2d, method='ssim'):
        """Compute pairwise similarity between 2D slices"""
        n_slices = slices_2d.shape[0]
        sim_matrix = np.zeros((n_slices, n_slices))
        
        for i in range(n_slices):
            for j in range(i, n_slices):
                if i == j:
                    sim_matrix[i, j] = 1.0
                else:
                    s_i = slices_2d[i] / (slices_2d[i].max() + 1e-8)
                    s_j = slices_2d[j] / (slices_2d[j].max() + 1e-8)
                    if method == 'ssim':
                        score = ssim(s_i, s_j, data_range=1.0)
                    else:  # correlation
                        score = np.corrcoef(s_i.flatten(), s_j.flatten())[0, 1]
                    sim_matrix[i, j] = score
                    sim_matrix[j, i] = score
        
        return sim_matrix
    
    def greedy_dissimilarity_selection(self, similarity_matrix, n_slices):
        """Greedy selection maintaining spatial coverage"""
        D = similarity_matrix.shape[0]
        if n_slices >= D:
            return list(range(D))
        
        n_regions = min(n_slices // 4, 8)
        region_size = D // n_regions
        selected = []
        
        # Spatial coverage: one per region
        for r in range(n_regions):
            start = r * region_size
            end = (r + 1) * region_size if r < n_regions - 1 else D
            region_avg_sim = np.mean(similarity_matrix[start:end, :], axis=1)
            local_idx = np.argmin(region_avg_sim)
            selected.append(start + local_idx)
        
        # Greedy for remaining
        dissimilarity = 1 - similarity_matrix
        while len(selected) < n_slices:
            min_sim = np.min(dissimilarity[:, selected], axis=1)
            min_sim[selected] = -np.inf
            next_idx = np.argmax(min_sim)
            selected.append(next_idx)
        
        return sorted(selected)
    
    def select_axis(self, volume, axis, n_slices=32):
        """
        Select representative slices along specified axis
        Args:
            volume: [D, H, W] ndarray
            axis: 0=D, 1=H, 2=W
            n_slices: number to select
        Returns:
            selected indices along axis
        """
        # Move axis to first position for easier slicing
        vol_moved = np.moveaxis(volume, axis, 0)  # [axis_size, ...]
        n_axis = vol_moved.shape[0]
        
        if n_slices >= n_axis:
            return list(range(n_axis))
        
        # Extract 2D slices along axis
        slices_2d = vol_moved.reshape(n_axis, -1)  # Flatten spatial dims
        slices_2d = slices_2d.reshape(n_axis, *vol_moved.shape[1:])  # Unflatten back (2D)
        
        if self.method == 'feature_variance':
            features = self.compute_slice_features(slices_2d)
            scaler = StandardScaler()
            features_scaled = scaler.fit_transform(features)
            distances = squareform(pdist(features_scaled, metric='euclidean'))
            similarity = 1 - (distances / (np.max(distances) + 1e-8))
            selected = self.greedy_dissimilarity_selection(similarity, n_slices)
        
        elif self.method == 'ssim_dissimilarity':
            similarity = self.compute_similarity_matrix(slices_2d, method='ssim')
            selected = self.greedy_dissimilarity_selection(similarity, n_slices)
        
        elif self.method == 'entropy':
            entropies = []
            for i in range(n_axis):
                hist, _ = np.histogram(slices_2d[i].flatten(), bins=256, density=True)
                hist = hist[hist > 0]
                entropy = -np.sum(hist * np.log2(hist + 1e-10))
                entropies.append(entropy)
            
            n_regions = min(n_slices // 4, 8)
            region_size = n_axis // n_regions
            selected = []
            for r in range(n_regions):
                start = r * region_size
                end = (r + 1) * region_size if r < n_regions - 1 else n_axis
                n_from_region = n_slices // n_regions + (1 if r < n_slices % n_regions else 0)
                region_ent = entropies[start:end]
                top_idx = np.argsort(region_ent)[-n_from_region:]
                selected.extend([start + i for i in top_idx])
            selected = sorted(selected)[:n_slices]
        
        elif self.method == 'gradient':
            grad_mags = []
            for i in range(n_axis):
                gx = np.gradient(slices_2d[i], axis=0)
                gy = np.gradient(slices_2d[i], axis=1)
                grad_mag = np.sqrt(gx**2 + gy**2)
                grad_mags.append(np.mean(grad_mag))
            
            n_regions = min(n_slices // 4, 8)
            region_size = n_axis // n_regions
            selected = []
            for r in range(n_regions):
                start = r * region_size
                end = (r + 1) * region_size if r < n_regions - 1 else n_axis
                n_from_region = n_slices // n_regions + (1 if r < n_slices % n_regions else 0)
                region_grad = grad_mags[start:end]
                top_idx = np.argsort(region_grad)[-n_from_region:]
                selected.extend([start + i for i in top_idx])
            selected = sorted(selected)[:n_slices]
        
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        return selected
    
    def select_xyz(self, volume, n_d=64, n_h=128, n_w=128):
        """
        Select representative slices along all 3 axes
        Args:
            volume: [D, H, W]
            n_d, n_h, n_w: number of slices to keep per axis
        Returns:
            volume_thin: [n_d, n_h, n_w]
        """
        vol = volume.copy()
        
        # Select along D (axis 0)
        d_idx = self.select_axis(vol, axis=0, n_slices=n_d)
        vol = vol[d_idx, :, :]
        
        # Select along H (axis 1)
        h_idx = self.select_axis(vol, axis=1, n_slices=n_h)
        vol = vol[:, h_idx, :]
        
        # Select along W (axis 2)
        w_idx = self.select_axis(vol, axis=2, n_slices=n_w)
        vol = vol[:, :, w_idx]
        
        return vol


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
        # axes[sample_idx, 0].text(-0.1, 0.5, sample_name, 
        #                           transform=axes[sample_idx, 0].transAxes,
        #                           fontsize=10, fontweight='bold',
        #                           verticalalignment='center',
        #                           horizontalalignment='right',
        #                           rotation=0)
    
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
    import random
    import numpy as np

    # Seleziona n_samples casuali (senza ripetizioni)
    tif_files = random.sample(tif_files, n_samples)

    # Oppure con numpy (più efficiente per liste grandi)
    indices = np.random.choice(len(tif_files), size=n_samples, replace=False)
    tif_files = [tif_files[i] for i in indices]
    
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
            selected_indices = selector.select_axis(volume, axis=0, n_slices=n_slices)
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
