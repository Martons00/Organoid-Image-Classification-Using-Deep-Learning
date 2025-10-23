from pathlib import Path
from typing import Optional, List, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff
from monai.transforms import Compose
from monai.transforms import (
    RandFlip, RandRotate90, RandAffine, RandGaussianNoise,
    RandAdjustContrast, RandShiftIntensity, RandZoom, RandGibbsNoise
)

CLASSES = {
    "chouxfleurs": 0,
    "compact": 1,
    "cystiques": 2,
}


def label_from_substring_or_none(p: str) -> Optional[int]:
    s = p.lower()
    if "chouxfleurs" in s: return CLASSES["chouxfleurs"]
    if "compact"     in s: return CLASSES["compact"]
    if "cystiques"   in s: return CLASSES["cystiques"]
    return None


def label_from_exact_parent_dir_or_none(p: str) -> Optional[int]:
    parent = Path(p).parent.name.lower()
    return CLASSES.get(parent, None)


def get_train_transforms():
    """
    Augmentation pipeline per TRAINING.
    Applicato on-the-fly, diverso ogni volta.
    """
    return Compose([
        # ============================================
        # Spatial Transformations
        # ============================================
        # Flip su tutti gli assi
        RandFlip(spatial_axis=0, prob=0.5),  # z-axis (depth)
        RandFlip(spatial_axis=1, prob=0.5),  # y-axis (height)
        RandFlip(spatial_axis=2, prob=0.5),  # x-axis (width)
        
        # Rotazioni di 90° (conservativo per organoid)
        RandRotate90(prob=0.5, max_k=3),
        
        # Affine trasformation (rotazione + zoom + traslazione)
        RandAffine(
            prob=0.5,
            rotate_range=(0.2, 0.2, 0.2),  # ±11.5° in radianti
            translate_range=(15, 15, 15),  # max 15 pixel shift
            scale_range=(0.1, 0.1, 0.1),  # scale ±10%
            shear_range=(0.05, 0.05, 0.05),  # mild shear
            mode="bilinear",
            padding_mode="border",
        ),
        
        # Zoom casuale (lieve)
        RandZoom(
            prob=0.3,
            min_zoom=0.95,
            max_zoom=1.05,
            mode="trilinear"
        ),
        
        # ============================================
        # Intensity-based Transformations
        # ============================================
        # Gaussian noise
        RandGaussianNoise(prob=0.2, mean=0.0, std=0.05),
        
        # Gibbs noise (artefatti biologici realistici)
        RandGibbsNoise(prob=0.1, alpha=(0.0, 0.3)),
        
        # Shift intensità
        RandShiftIntensity(prob=0.2, offsets=0.1),
        
        # Contrast adjustment
        RandAdjustContrast(prob=0.2, gamma=(0.8, 1.2)),
    ])


def get_val_transforms():
    """
    Transformazioni per VALIDAZIONE/TEST.
    Solo normalization, NO augmentation.
    """
    return Compose([])  # Niente augmentation per val/test


class OrganoidsINRIA3D(Dataset):
    """
    Dataset PyTorch con data augmentation on-the-fly.
    
    Args:
        root: Cartella radice del dataset
        exact_class_dir: Se True, usa il nome della cartella padre per la classe
        is_train: Se True, applica augmentation; se False, no
    """
    def __init__(
        self,
        root: str,
        exact_class_dir: bool = False,
        is_train: bool = True,
    ):
        self.root = Path(root)
        self.is_train = is_train
        
        # Prepara transform (on-the-fly, diverso ogni volta)
        self.transform = get_train_transforms() if is_train else get_val_transforms()

        # Raccoglie i file .tif/.tiff
        all_paths = sorted({
            *(str(p) for p in self.root.rglob("*.tif")),
            *(str(p) for p in self.root.rglob("*.tiff"))
        })

        if len(all_paths) == 0:
            raise RuntimeError(f"Nessun .tif valido trovato sotto {self.root}")

        # Filtering + precompute labels
        paths: List[str] = []
        labels: List[int] = []

        if exact_class_dir:
            lab_fn = label_from_exact_parent_dir_or_none
        else:
            lab_fn = label_from_substring_or_none

        for p in all_paths:
            y = lab_fn(p)
            if y is not None:
                paths.append(p)
                labels.append(int(y))

        if len(paths) == 0:
            raise RuntimeError(
                f"Nessun file appartiene alle classi {list(CLASSES.keys())} "
                f"con exact_class_dir={exact_class_dir}"
            )

        self.paths = paths
        self.labels = np.asarray(labels, dtype=np.int64)
        self.exact_class_dir = exact_class_dir

        print(f"[{'TRAIN' if is_train else 'VAL'}] Loaded {len(self.paths)} organoid volumes")
        for cls_name, cls_idx in CLASSES.items():
            count = (self.labels == cls_idx).sum()
            print(f"  {cls_name}: {count}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]

        # Caricamento lazy del volume
        vol_np = tiff.imread(p)

        # Se 2D, alza a [D,H,W] per uniformare
        if vol_np.ndim == 2:
            vol_np = vol_np[None, ...]

        # Salva dimensioni originali
        size = np.array(vol_np.shape, dtype=np.int64)

        # Normalizzazione a float32 in [0, 1]
        if vol_np.dtype == np.uint8:
            vol_np = vol_np.astype(np.float32) / 255.0
        else:
            vol_np = vol_np.astype(np.float32)
            # Normalizza a [0, 1] se necessario
            if vol_np.max() > 1.0:
                vol_np = vol_np / vol_np.max()

        # Aggiunge dimensione canale in testa → [C, D, H, W]
        vol_np = np.expand_dims(vol_np, axis=0)
        vol = torch.from_numpy(vol_np)
        
        _, D, H, W = vol.shape
        max_D = 128

        # ============================================
        # Normalizzazione dimensione Z a 128
        # ============================================
        if D > max_D:
            # Selezione equispaziata di esattamente max_D indici
            idx_d = torch.linspace(
                10, D - 1, steps=max_D, device=vol.device
            ).round().to(torch.long)
            idx_d = torch.clamp(idx_d, 10, D - 1)
            vol = vol.index_select(1, idx_d)  # [C, max_D, H, W]
        else:
            # Padding a 128 z
            pad_z = max_D - D
            pad_z_front = pad_z // 2
            pad_z_back = pad_z - pad_z_front
            vol = torch.nn.functional.pad(
                vol, 
                (0, 0, 0, 0, pad_z_front, pad_z_back),
                mode='constant',
                value=0
            )

        # ============================================
        # AUGMENTATION on-the-fly (solo training)
        # ============================================
        if self.is_train and self.transform is not None:
            vol = self.transform(vol)

        y = int(self.labels[idx])
        label = torch.tensor(y, dtype=torch.long)
        name = os.path.splitext(os.path.basename(p))[0]

        return {
            "vol": vol,          # [C, D, H, W] = [1, 128, H, W]
            "label": label,      # scalar
            "name": name,
            "size": size,
            "path": p,
        }


if __name__ == "__main__":
    # Test del dataset con augmentation
    import matplotlib.pyplot as plt
    
    # Dataset TRAINING con augmentation
    train_ds = OrganoidsINRIA3D(
        root="/home/mraffael/martone_project/Organoids_Dataset",
        exact_class_dir=False,
        is_train=True,  # ✅ Augmentation attiva
    )
    
    # Dataset VALIDATION senza augmentation
    val_ds = OrganoidsINRIA3D(
        root="/home/mraffael/martone_project/Organoids_Dataset",
        exact_class_dir=False,
        is_train=False,  # ❌ Augmentation disattiva
    )
    
    # ============================================
    # Visualizza augmentations
    # ============================================
    print("\n=== Visualizzando augmentations ===")
    sample_idx = 0
    
    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    
    # Stessa immagine, diversi augmentations
    for i in range(6):
        # OGNI chiamata a __getitem__ applica diversa augmentation
        sample = train_ds[sample_idx]
        vol = sample["vol"]  # [1, 128, H, W]
        
        # Visualizza slice centrale
        mid_slice = vol.shape[1] // 2
        img = vol[0, mid_slice, :, :].numpy()
        
        row = i // 3
        col = i % 3
        axes[row, col].imshow(img, cmap='gray')
        axes[row, col].set_title(f'Augmentation {i+1}')
        axes[row, col].axis('off')
    
    plt.suptitle(f'Data Augmentation (stessa immagine, diverse trasformazioni)')
    plt.tight_layout()
    plt.savefig('augmentations_visualization.png', dpi=100, bbox_inches='tight')
    print("✅ Salvato: augmentations_visualization.png")
    
    # ============================================
    # Test DataLoader
    # ============================================
    print("\n=== Testing DataLoaders ===")
    
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=2,
        shuffle=True,
        num_workers=2,  # ✅ Parallel augmentation
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_ds,
        batch_size=2,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Itera un batch
    for batch_idx, batch_data in enumerate(train_loader):
        vol = batch_data["vol"]  # [B, C, D, H, W]
        label = batch_data["label"]  # [B]
        names = batch_data["name"]
        
        print(f"Batch {batch_idx}:")
        print(f"  Volume shape: {vol.shape}")
        print(f"  Labels: {label}")
        print(f"  Names: {names}")
        
        if batch_idx >= 2:
            break
    
    print("\n✅ Dataset setup completed!")
