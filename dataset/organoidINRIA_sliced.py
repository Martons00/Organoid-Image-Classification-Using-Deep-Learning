from pathlib import Path
from typing import Optional, List, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset

from monai.transforms import Compose
from monai.transforms import (
    RandFlip, RandRotate90, RandAffine, RandGaussianNoise,
    RandAdjustContrast, RandShiftIntensity, RandZoom, OneOf, EnsureChannelFirst
)
import tifffile as tiff

# Dentro training/train.py
import os
import sys
from .sliceSelector import SliceSelector

# Calcola il path assoluto di project_root/tools
TOOLS_PATH = os.path.abspath(os.path.join(__file__, '..', '..', 'tools'))
if TOOLS_PATH not in sys.path:
    sys.path.insert(0, TOOLS_PATH)


from similarity import compute_similarity_matrix, plot_similarity_heatmap


CLASSES = {
    "chouxfleurs": 0,
    "compact": 1,
    "cystiques": 2,
}
# ============================================
import monai
import monai.transforms.compose
import monai.transforms.transform
import numpy as np
# ============================================
# PATCH MONAI - ESEGUITO UNA SOLA VOLTA AL MODULO IMPORT
# ============================================

# 1. Patch per monai.utils.get_seed
_original_get_seed = monai.utils.get_seed

def _safe_get_seed():
    seed = _original_get_seed()
    return seed % (2**31 - 1) if seed is not None else None

monai.utils.get_seed = _safe_get_seed

# 2. Patch per Compose.set_random_state
_original_compose_set_random_state = monai.transforms.compose.Compose.set_random_state

def _safe_compose_set_random_state(self, seed=None, state=None):
    """Wrapper sicuro che evita overflow nel seed"""
    if seed is not None:
        seed = int(seed) % (2**31 - 1)
    # Chiama il metodo ORIGINALE SALVATO, non il wrapper
    return _original_compose_set_random_state(self, seed=seed, state=state)

monai.transforms.compose.Compose.set_random_state = _safe_compose_set_random_state

# 3. Patch per Randomizable.set_random_state
_original_randomizable_set_random_state = monai.transforms.transform.Randomizable.set_random_state

def _safe_randomizable_set_random_state(self, seed=None, state=None):
    """Wrapper sicuro che evita overflow nel seed"""
    if seed is not None:
        if not isinstance(seed, (int, np.integer)):
            seed = int(seed)
        seed = seed % (2**31 - 1)
    # Chiama il metodo ORIGINALE SALVATO, non il wrapper
    return _original_randomizable_set_random_state(self, seed=seed, state=state)

monai.transforms.transform.Randomizable.set_random_state = _safe_randomizable_set_random_state

# 4. Patch per il MAX_SEED stesso (se esiste)
if hasattr(monai.transforms.transform, 'MAX_SEED'):
    monai.transforms.transform.MAX_SEED = 2**31 - 1

# ============================================
# FUNZIONE - SENZA PATCH (Già fatto sopra!)
# ============================================

def get_train_transforms():
    """
    Augmentation pipeline per TRAINING.
    
    ✅ Il monkey-patching è stato fatto UNA SOLA VOLTA al module import,
       quindi non c'è ricorsione infinita.
    """
    return Compose([
        # Flip 3D (specifica gli assi spaziali, non i canali)
        RandFlip(spatial_axis=0, prob=0.5),  # depth
        RandFlip(spatial_axis=1, prob=0.5),  # height  
        RandFlip(spatial_axis=2, prob=0.5),  # width
        
        # Rotazioni 90° per 3D
        RandRotate90(prob=0.5, max_k=3, spatial_axes=(2, 3)),  # Specifica gli assi
        
        # Intensità (one-of)
        OneOf([
            RandGaussianNoise(prob=1.0, std=0.03),
            RandAdjustContrast(prob=1.0, gamma=(0.8, 1.25)),
            RandShiftIntensity(prob=1.0, offsets=0.1),
        ], weights=[0.4, 0.3, 0.3]),
    ])




def label_from_substring_or_none(p: str) -> Optional[int]:
    s = p.lower()
    if "chouxfleurs" in s: return CLASSES["chouxfleurs"]
    if "compact"     in s: return CLASSES["compact"]
    if "cystiques"   in s: return CLASSES["cystiques"]
    return None  # Nessuna classe → escludere

def label_from_exact_parent_dir_or_none(p: str) -> Optional[int]:
    parent = Path(p).parent.name.lower()
    return CLASSES.get(parent, None)  # None se la cartella non è una delle 3

class OrganoidsINRIA3D(Dataset):
    """
    Dataset PyTorch che indicizza solo i file .tif/.tiff appartenenti alle 3 classi CLASSES,
    senza alcuna classe di default. I file non riconosciuti vengono esclusi a monte.
    """
    def __init__(self, root: str, exact_class_dir: bool = False,slice_selection: bool = True,n_slices: int = 32):
        self.root = Path(root)

        # Raccoglie i file .tif/.tiff
        all_paths = sorted({*(str(p) for p in self.root.rglob("*.tif")),
                            *(str(p) for p in self.root.rglob("*.tiff"))})

        if len(all_paths) == 0:
            raise RuntimeError(f"Nessun .tif valido trovato sotto {self.root}")

        # Filtering + precompute labels (senza I/O immagini)
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
                f"Nessun file appartiene alle classi {list(CLASSES.keys())} con exact_class_dir={exact_class_dir}"
            )

        self.paths = paths
        # array int64 per compatibilità con torch long
        self.labels = np.asarray(labels, dtype=np.int64)
        self.exact_class_dir = exact_class_dir
        self.slice_selection = slice_selection
        if slice_selection:
            self.n_slices = n_slices
            self.selector = SliceSelector(method='feature_variance')

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        vol_np = tiff.imread(p)

        y = int(self.labels[idx])
        label = torch.tensor(y, dtype=torch.long)
        name = os.path.splitext(os.path.basename(p))[0]

        # [D, H, W] garantito
        if vol_np.ndim == 2:
            vol_np = vol_np[None, ...]
        size = np.array(vol_np.shape, dtype=np.int64)

        # Normalizzazione
        if vol_np.dtype == np.uint8:
            vol_np = vol_np.astype(np.float32) / 255.0
        else:
            vol_np = vol_np.astype(np.float32)


        # OLD LOGIC: resize to 128 slices
        D = vol_np.shape[0]
        max_D = 128
        
        if D > max_D:
            # Downsampling equispaziato
            indices = np.linspace(10, D - 1, num=max_D, dtype=np.int64)
            indices = np.clip(indices, 10, D - 1)
            vol_np = vol_np[indices, ...]
        else:
            # Padding a 128
            pad_z = max_D - D
            pad_z_front = pad_z // 2
            pad_z_back = pad_z - pad_z_front
            vol_np = np.pad(vol_np, ((pad_z_front, pad_z_back), (0, 0), (0, 0)), 
                        mode='constant', constant_values=0)
        
        assert vol_np.shape[0] == max_D, \
            f"Shape mismatch after old logic: {vol_np.shape[0]} != {max_D}"

        # Adattamento per SliceSelector3D con axis-based selection
        if self.slice_selection:
            # ========== SELEZIONE ADATTIVA SU 3 ASSI ==========
            target_D, target_H, target_W = self.n_slices,self.n_slices,self.n_slices  # es. (32, 32, 32)
            # 1) SELEZIONE SLICE (D - axis 0)
            D, H, W = vol_np.shape
            if D > target_D:
                d_idx = self.selector.select_axis(vol_np, axis=0, n_slices=target_D)
                vol_np = vol_np[d_idx, :, :]
            elif D < target_D:
                pad = target_D - D
                pad_front = pad // 2
                pad_back = pad - pad_front
                vol_np = np.pad(vol_np, ((pad_front, pad_back), (0, 0), (0, 0)), 
                                mode='constant', constant_values=0.0)

            # 2) SELEZIONE ROWS (H - axis 1)
            if H > target_H:
                h_idx = self.selector.select_axis(vol_np, axis=1, n_slices=target_H)
                vol_np = vol_np[:, h_idx, :]
            elif H < target_H:
                pad = target_H - H
                pad_top = pad // 2
                pad_bottom = pad - pad_top
                vol_np = np.pad(vol_np, ((0, 0), (pad_top, pad_bottom), (0, 0)), 
                                mode='constant', constant_values=0.0)

            # 3) SELEZIONE COLS (W - axis 2)
            if W > target_W:
                w_idx = self.selector.select_axis(vol_np, axis=2, n_slices=target_W)
                vol_np = vol_np[:, :, w_idx]
            elif W < target_W:
                pad = target_W - W
                pad_left = pad // 2
                pad_right = pad - pad_left
                vol_np = np.pad(vol_np, ((0, 0), (0, 0), (pad_left, pad_right)), 
                                mode='constant', constant_values=0.0)

            # Verifica finale
            assert vol_np.shape == (target_D, target_H, target_W), \
                f"Shape mismatch finale: {vol_np.shape} != {(target_D, target_H, target_W)}"


        # Canale
        vol_np = np.expand_dims(vol_np, axis=0)  # [1,D,H,W]
        vol = torch.from_numpy(vol_np)

        return {"vol": vol, "label": label, "name": name, "size": size, "path": p}



import torch

def selective_augmentation(data, transform, augmentation_ratio=0.5):
    B = data.shape[0]
    k = int(B * augmentation_ratio)
    if k == 0 or transform is None:
        return data

    idx = torch.randperm(B, device=data.device)[:k]
    out = data.clone()
    aug_sub = transform(data.index_select(0, idx))  # [k, C, ...]
    out.index_copy_(0, idx, aug_sub)
    return out







if __name__ == "__main__":
    # Matching per sottostringa: include solo i file che contengono una delle 3 classi nel percorso
    ds = OrganoidsINRIA3D(
        root="/home/mraffael/martone_project/Organoids_Dataset",
        exact_class_dir=False,
        slice_selection=True,
        n_slices=64
    )
