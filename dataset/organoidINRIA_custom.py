from pathlib import Path
from typing import Optional, List, Tuple
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff
# Dentro training/train.py
import os
import sys

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
    def __init__(self, root: str, exact_class_dir: bool = False):
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

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]

        # Caricamento lazy dell'immagine/volume
        vol_np = tiff.imread(p)

        # Se 2D, alza a [1,H,W] per uniformare
        if vol_np.ndim == 2:
            vol_np = vol_np[None, ...]

        # Salva dimensioni originali prima della normalizzazione/canale
        size = np.array(vol_np.shape, dtype=np.int64)

        # Normalizzazione a float32
        if vol_np.dtype == np.uint8:
            vol_np = vol_np.astype(np.float32) / 255.0
        else:
            vol_np = vol_np.astype(np.float32)

        # Aggiunge dimensione canale in testa → [1, D, H, W] o [1, 1, H, W] se 2D
        vol_np = np.expand_dims(vol_np, axis=0)
        vol = torch.from_numpy(vol_np)
        _, D, _, _ = vol.shape
        max_D = 128
        if D > max_D:
            # Selezione equispaziata di esattamente max_D indici
            idx_d = torch.linspace(10, D - 1, steps=max_D, device=vol.device).round().to(torch.long)
            idx_d = torch.clamp(idx_d, 10, D - 1)
            vol = vol.index_select(1, idx_d)  # [B,C,max_D,H,W]
        else:
            # Padding a 128 z
            pad_z = max_D - D
            pad_z_front = pad_z // 2
            pad_z_back = pad_z - pad_z_front
            vol = torch.nn.functional.pad(vol, (0, 0, 0, 0, pad_z_front, pad_z_back), mode='constant', value=0)
        


        y = int(self.labels[idx])
        label = torch.tensor(y, dtype=torch.long)
        name = os.path.splitext(os.path.basename(p))[0]

        return {"vol": vol, "label": label, "name": name, "size": size, "path": p}



if __name__ == "__main__":
    # Matching per sottostringa: include solo i file che contengono una delle 3 classi nel percorso
    ds = OrganoidsINRIA3D(
        root="/home/mraffael/martone_project/Organoids_Dataset",
        exact_class_dir=False
    )
    dl = torch.utils.data.DataLoader(ds, batch_size=1, shuffle=True)
    samples = [dl.dataset[i]['vol'].view(1, -1) for i in range(100)]
    samples = torch.cat(samples, dim=0)  # [N,C*D*H*W]
    print(samples.shape)
    sim = compute_similarity_matrix(samples)
    print(f"AVG SIMILARITY: {sim.mean()}")
    plot_similarity_heatmap(sim, save_path="similarity_example.png")

