from pathlib import Path
import os
import numpy as np
import torch
from torch.utils.data import Dataset
import tifffile as tiff

CLASSES = {
    "chouxfleurs": 0,
    "compact": 1,
    "cystiques": 2,
}

def label_from_substring(p: str, default_other: int = 1) -> int:
    s = p.lower()
    if "chouxfleurs" in s: return 0
    if "compact"     in s: return 1
    if "cystiques"   in s: return 2
    return default_other

def label_from_exact_parent_dir(p: str, default_other: int = 1) -> int:
    parent = Path(p).parent.name.lower()  # nome cartella immediata
    return CLASSES.get(parent, default_other)

class OrganoidsINRIA3D(Dataset):
    """
    Se exact_class_dir=True, tiene solo i .tif la cui cartella immediata è
    esattamente una delle classi (chouxfleurs/compact/cystiques) e assegna
    la label in base al nome cartella. Altrimenti usa il matching per sottostringa.
    """
    def __init__(self, root: str, default_other: int = 1, exact_class_dir: bool = False):
        self.root = Path(root)
        # raccoglie .tif e .tiff
        paths = sorted({*(str(p) for p in self.root.rglob("*.tif")),
                        *(str(p) for p in self.root.rglob("*.tiff"))})

        if exact_class_dir:
            # filtra: tiene solo quelli con parent.name in CLASSES
            keep = []
            for p in paths:
                parent = Path(p).parent.name.lower()
                if parent in CLASSES:
                    keep.append(p)
            self.paths = keep
        else:
            self.paths = paths

        if len(self.paths) == 0:
            raise RuntimeError(f"Nessun .tif valido trovato sotto {self.root}")

        self.default_other = default_other
        self.exact_class_dir = exact_class_dir

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx: int):
        p = self.paths[idx]
        vol_np = tiff.imread(p)
        if vol_np.ndim == 2:
            vol_np = vol_np[None, ...]
        size = np.array(vol_np.shape, dtype=np.int64)

        if vol_np.dtype == np.uint8:
            vol_np = vol_np.astype(np.float32) / 255.0
        else:
            vol_np = vol_np.astype(np.float32)

        vol_np = np.expand_dims(vol_np, axis=0)  # [1,D,H,W]
        vol = torch.from_numpy(vol_np)

        if self.exact_class_dir:
            y = label_from_exact_parent_dir(p, self.default_other)
        else:
            y = label_from_substring(p, self.default_other)

        label = torch.tensor(y, dtype=torch.long)
        name = os.path.splitext(os.path.basename(p))[0]
        return {"vol": vol, "label": label, "name": name, "size": size, "path": p}

if __name__ == "__main__":
    # Matching per sottostringa (comportamento attuale):
    ds = OrganoidsINRIA3D(root="/home/mraffael/martone_project/Organoids_Dataset", default_other=3, exact_class_dir=False)
    print(f"Trovati {len(ds)} volumi .tif/.tiff")
    '''
    print(f"Numero di sample per classe:")
    for i in range(4):
        count = sum(item['label'].item() == i for item in ds)
        print(f" - Classe {i}: {count}")


    # Matching esatto sul nome cartella:
    ds_exact = OrganoidsINRIA3D(root="/home/mraffael/martone_project/Organoids_Dataset", default_other=3, exact_class_dir=True)
    print(f"Trovati {len(ds_exact)} volumi .tif/.tiff con matching esatto sul nome cartella")
    print(f"Numero di sample per classe:")
    for i in range(4):
        count = sum(item['label'].item() == i for item in ds_exact)
        print(f" - Classe {i}: {count}")
        '''
    
    # Matching esatto sul nome cartella:
    ds_exact = OrganoidsINRIA3D(root="/home/mraffael/martone_project/Organoids_Dataset", default_other=3, exact_class_dir=True)
    print(f"Trovati {len(ds_exact)} volumi .tif/.tiff con matching esatto sul nome cartella")