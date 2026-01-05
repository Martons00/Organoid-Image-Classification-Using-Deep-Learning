# Standard library
import json
import math
import os
from collections import Counter
from typing import Dict, List, Sequence, Tuple, Union

# Third-party libraries
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset, random_split
from sklearn.model_selection import StratifiedKFold, train_test_split
from monai import data, transforms



class Sampler(torch.utils.data.Sampler):
    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=True, make_even=True):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.shuffle = shuffle
        self.make_even = make_even
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        self.total_size = self.num_samples * self.num_replicas
        indices = list(range(len(self.dataset)))
        self.valid_length = len(indices[self.rank : self.total_size : self.num_replicas])

    def __iter__(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))
        if self.make_even:
            if len(indices) < self.total_size:
                if self.total_size - len(indices) < len(indices):
                    indices += indices[: (self.total_size - len(indices))]
                else:
                    extra_ids = np.random.randint(low=0, high=len(indices), size=self.total_size - len(indices))
                    indices += [indices[ids] for ids in extra_ids]
            assert len(indices) == self.total_size
        indices = indices[self.rank : self.total_size : self.num_replicas]
        self.num_samples = len(indices)
        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        self.epoch = epoch

def datafold_read(datalist, basedir, fold=0, key="training"):
    with open(datalist) as f:
        json_data = json.load(f)

    json_data = json_data[key]

    for d in json_data:
        for k, v in d.items():
            if isinstance(d[k], list):
                d[k] = [os.path.join(basedir, iv) for iv in d[k]]
            elif isinstance(d[k], str):
                d[k] = os.path.join(basedir, d[k]) if len(d[k]) > 0 else d[k]

    tr = []
    val = []
    for d in json_data:
        if "fold" in d and d["fold"] == fold:
            val.append(d)
        else:
            tr.append(d)

    return tr, val

def get_loader(args):
    data_dir = args.data_dir
    datalist_json = args.json_list
    train_files, validation_files = datafold_read(datalist=datalist_json, basedir=data_dir, fold=args.fold)
    train_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.CropForegroundd(
                keys=["image", "label"], source_key="image", k_divisible=[args.roi_x, args.roi_y, args.roi_z], allow_smaller=True
            ),
            transforms.RandSpatialCropd(
                keys=["image", "label"], roi_size=[args.roi_x, args.roi_y, args.roi_z], random_size=False
            ),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=0),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=1),
            transforms.RandFlipd(keys=["image", "label"], prob=0.5, spatial_axis=2),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.RandScaleIntensityd(keys="image", factors=0.1, prob=1.0),
            transforms.RandShiftIntensityd(keys="image", offsets=0.1, prob=1.0),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    test_transform = transforms.Compose(
        [
            transforms.LoadImaged(keys=["image", "label"]),
            transforms.ConvertToMultiChannelBasedOnBratsClassesd(keys="label"),
            transforms.NormalizeIntensityd(keys="image", nonzero=True, channel_wise=True),
            transforms.ToTensord(keys=["image", "label"]),
        ]
    )

    if args.test_mode:
        val_ds = data.Dataset(data=validation_files, transform=test_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        test_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )

        loader = test_loader
    else:
        train_ds = data.Dataset(data=train_files, transform=train_transform)

        train_sampler = Sampler(train_ds) if args.distributed else None
        train_loader = data.DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=(train_sampler is None),
            num_workers=args.workers,
            sampler=train_sampler,
            pin_memory=True,
        )
        val_ds = data.Dataset(data=validation_files, transform=val_transform)
        val_sampler = Sampler(val_ds, shuffle=False) if args.distributed else None
        val_loader = data.DataLoader(
            val_ds, batch_size=1, shuffle=False, num_workers=args.workers, sampler=val_sampler, pin_memory=True
        )
        loader = [train_loader, val_loader]

    return loader

def split_dataset_percentage(
    dataset: Dataset,
    split_fracs: Union[Sequence[float], Dict[Union[int, str], float]] = (0.4, 0.2, 0.4),
    val_size: Union[int, float] = 0.2,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Crea uno split train/val stratificato dove il numero di campioni per classe
    è proporzionale alle frazioni desiderate e la suddivisione train/val avviene
    all'interno di ciascuna classe.
    
    Parametri:
      - split_fracs: sequence (nell'ordine di unique_classes) oppure dict {label: frazione}.
                     Le frazioni vengono normalizzate se non sommano a 1.
      - val_size: frazione (0<val<=1) oppure numero intero di campioni PER CLASSE nel validation.
      - seed: riproducibilità.
    
    Restituisce:
      - (train_subset, val_subset)
    """
    labels = np.asarray(dataset.labels)
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_classes)

    # Mappa frazioni ai label, accetta sequence o dict
    if isinstance(split_fracs, dict):
        fracs = np.array([float(split_fracs.get(cls, 0.0)) for cls in unique_classes], dtype=float)
    else:
        fracs = np.array(split_fracs, dtype=float)
        if len(fracs) != n_classes:
            raise ValueError(f"split_fracs length {len(fracs)} != n_classes {n_classes}")

    # Normalizza frazioni (e valida)
    fracs = np.maximum(fracs, 0.0)
    total = fracs.sum()
    if total <= 0:
        raise ValueError("split_fracs deve contenere almeno una frazione > 0")
    fracs = fracs / total

    # Se tutte le frazioni sono zero (dopo clip) errore già gestito; altrimenti calcola T massimo
    # T = min_c floor(N_c / f_c) per f_c > 0
    positive = fracs > 0
    if not np.any(positive):
        raise ValueError("Tutte le frazioni risultano 0 dopo la normalizzazione")
    T_candidates = np.floor(class_counts[positive] / fracs[positive])
    if T_candidates.size == 0 or np.min(T_candidates) < 1:
        raise ValueError("Frazioni troppo ambiziose rispetto ai conteggi: impossibile selezionare almeno 1 campione")

    T = int(np.min(T_candidates))

    # Conteggi base e resti (Hamilton / largest remainder), con cap per disponibilità
    raw = fracs * T
    base = np.floor(raw).astype(int)
    rema = raw - base
    # Cap di disponibilità: non superare N_c
    cap = class_counts - base
    # Quanti restano da assegnare per raggiungere T totale
    R = T - int(base.sum())
    if R > 0:
        order = np.argsort(-rema)  # decrescente per resto
        i = 0
        while R > 0 and np.any(cap > 0):
            idx = order[i % n_classes]
            if cap[idx] > 0 and fracs[idx] > 0:
                base[idx] += 1
                cap[idx] -= 1
                R -= 1
            i += 1

    target_per_class = base
    # Verifica finale: non superare class_counts
    target_per_class = np.minimum(target_per_class, class_counts)

    rng = np.random.default_rng(seed)
    train_indices = []
    val_indices = []

    # Split per classe
    for cls, n_target in zip(unique_classes, target_per_class):
        cls_idx = np.where(labels == cls)[0]
        rng.shuffle(cls_idx)

        if n_target <= 0:
            continue

        selected = cls_idx[:n_target]

        # Calcola val per classe
        if isinstance(val_size, float):
            v = int(round(val_size * n_target))
        else:
            v = int(val_size)

        # Vincoli: 0 <= v <= n_target-1, lasciando almeno 1 in train quando possibile
        if n_target >= 2:
            v = max(1, min(v, n_target - 1))
        else:
            v = 0  # se c'è un solo elemento, tutto in train

        t = n_target - v

        # Assegna
        val_indices.extend(selected[:v].tolist())
        train_indices.extend(selected[v:v + t].tolist())

    # Shuffle globale
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    rng.shuffle(train_indices)
    rng.shuffle(val_indices)

    # Report
    chosen_counts = []
    for cls in unique_classes:
        in_train = np.sum(labels[train_indices] == cls)
        in_val = np.sum(labels[val_indices] == cls)
        chosen_counts.append((cls, in_train, in_val))
    print("Stratified proportional split per classe (train, val):", chosen_counts)
    print(f"Total: {len(train_indices)} train, {len(val_indices)} val")

    return Subset(dataset, train_indices.tolist()), Subset(dataset, val_indices.tolist())

def split_dataset_balanced(
    dataset: Dataset,
    val_size: Union[int, float] = 0.2,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Divide un Dataset PyTorch in train/val con lo stesso numero di campioni per classe.
    val_size: frazione (0<val<=1) oppure numero intero di campioni PER CLASSE nel validation.
    Restituisce (train_subset, val_subset) perfettamente bilanciati.
    """
    labels = dataset.labels  # Assume che il dataset abbia un attributo .labels (np.array)
    unique_classes, class_counts = np.unique(labels, return_counts=True)
    n_classes = len(unique_classes)
    
    # Trova la classe con meno campioni per determinare il limite
    min_class_samples = np.min(class_counts)
    
    if isinstance(val_size, float):
        val_samples_per_class = int(round(val_size * min_class_samples))
    else:
        val_samples_per_class = int(val_size)
    
    # Assicurati che rimangano abbastanza campioni per il training
    val_samples_per_class = max(1, min(val_samples_per_class, min_class_samples - 1))
    train_samples_per_class = min_class_samples - val_samples_per_class
    
    train_indices = []
    val_indices = []
    
    np.random.seed(seed)
    
    for cls in unique_classes:
        # Trova tutti gli indici per questa classe
        cls_indices = np.where(labels == cls)[0]
        
        # Prendi solo i primi min_class_samples per bilanciare
        cls_indices = cls_indices[:min_class_samples]
        
        # Shuffle gli indici per questa classe
        np.random.shuffle(cls_indices)
        
        # Split train/val per questa classe
        train_indices.extend(cls_indices[:train_samples_per_class])
        val_indices.extend(cls_indices[train_samples_per_class:train_samples_per_class + val_samples_per_class])
    
    # Converti in liste e shuffle finale
    train_indices = np.array(train_indices)
    val_indices = np.array(val_indices)
    
    np.random.shuffle(train_indices)
    np.random.shuffle(val_indices)
    
    train_subset = Subset(dataset, train_indices.tolist())
    val_subset = Subset(dataset, val_indices.tolist())
    
    print(f"Balanced split: {train_samples_per_class} train samples per class, {val_samples_per_class} val samples per class")
    print(f"Total: {len(train_subset)} train, {len(val_subset)} val")
    
    return train_subset, val_subset

def create_balanced_debug_subset(
    dataset_subset: Subset, 
    original_labels: np.ndarray,
    samples_per_class: int, 
    seed: int = 42
) -> Subset:
    """
    Crea un subset bilanciato per il debug con lo stesso numero di campioni per classe.
    samples_per_class: numero di campioni da prendere per ogni classe.
    """
    # Ottieni le label corrispondenti agli indici del subset
    subset_labels = original_labels[dataset_subset.indices]
    unique_classes, class_counts = np.unique(subset_labels, return_counts=True)
    
    # Verifica che ogni classe abbia abbastanza campioni
    min_available = np.min(class_counts)
    samples_per_class = min(samples_per_class, min_available)
    
    if samples_per_class <= 0:
        print(f"Warning: Not enough samples per class. Using {min_available} samples per class.")
        samples_per_class = min_available
    
    debug_indices = []
    
    np.random.seed(seed)
    
    for cls in unique_classes:
        # Trova gli indici nel subset per questa classe
        cls_mask = subset_labels == cls
        cls_subset_indices = np.where(cls_mask)[0]
        
        # Seleziona samples_per_class campioni casuali
        np.random.shuffle(cls_subset_indices)
        selected_indices = cls_subset_indices[:samples_per_class]
        debug_indices.extend(selected_indices)
    
    # Shuffle finale
    debug_indices = np.array(debug_indices)
    np.random.shuffle(debug_indices)
    
    # Mappa gli indici del subset agli indici originali del dataset
    original_indices = [dataset_subset.indices[i] for i in debug_indices]
    
    print(f"Debug subset: {samples_per_class} samples per class, {len(debug_indices)} total samples")
    
    return Subset(dataset_subset.dataset, original_indices)

def verify_balance(subset: Subset, original_labels: np.ndarray) -> None:
    """
    Verifica e stampa la distribuzione delle classi in un subset.
    """
    subset_labels = original_labels[subset.indices]
    unique_classes, class_counts = np.unique(subset_labels, return_counts=True)
    
    print("Class distribution:")
    for cls, count in zip(unique_classes, class_counts):
        print(f"  Class {cls}: {count} samples")

def split_dataset_random(
    dataset: Dataset,
    val_size: Union[int, float] = 0.2,
    seed: int = 42
) -> Tuple[Dataset, Dataset]:
    """
    Divide un Dataset PyTorch in train/val in modo casuale.
    val_size: frazione (0<val<=1) oppure numero intero di campioni.
    Restituisce (train_subset, val_subset).
    """
    n = len(dataset)
    if isinstance(val_size, float):
        val_len = int(round(val_size * n))
    else:
        val_len = int(val_size)
    val_len = max(1, min(n - 1, val_len))
    train_len = n - val_len

    g = torch.Generator().manual_seed(seed)
    train_subset, val_subset = random_split(dataset, [train_len, val_len], generator=g)
    return train_subset, val_subset

def split_dataset_stratified(
    dataset: Dataset,
    val_size: Union[int, float] = 0.2,
    seed: int = 42
) -> Tuple[Subset, Subset]:
    """
    Divide un Dataset PyTorch in train/val mantenendo la distribuzione delle classi.
    val_size: frazione (0<val<=1) oppure numero intero di campioni.
    Restituisce (train_subset, val_subset).
    """
    n = len(dataset)
    labels = dataset.labels  # Assume che il dataset abbia un attributo .labels (np.array)
    
    if isinstance(val_size, float):
        val_len = int(round(val_size * n))
    else:
        val_len = int(val_size)
    val_len = max(1, min(n - 1, val_len))
    
    # Split stratificato usando sklearn
    indices = np.arange(n)
    train_idx, val_idx = train_test_split(
        indices, 
        test_size=val_len, 
        stratify=labels, 
        random_state=seed
    )
    
    train_subset = Subset(dataset, train_idx.tolist())
    val_subset = Subset(dataset, val_idx.tolist())
    
    return train_subset, val_subset

def create_stratified_debug_subset(
    dataset_subset: Subset, 
    original_labels: np.ndarray,
    n_samples: int, 
    seed: int = 42
) -> Subset:
    """
    Crea un subset stratificato per il debug da un Subset esistente.
    """
    if len(dataset_subset) <= n_samples:
        return dataset_subset
    
    # Ottieni le label corrispondenti agli indici del subset
    subset_labels = original_labels[dataset_subset.indices]
    
    # Verifica che ci siano abbastanza campioni per classe
    unique_classes, class_counts = np.unique(subset_labels, return_counts=True)
    min_samples_per_class = max(1, n_samples // len(unique_classes))
    
    # Se qualche classe ha troppo pochi campioni, usa un approccio diverso
    if np.any(class_counts < min_samples_per_class):
        # Prendi almeno 1 campione per classe, poi riempi casualmente
        debug_indices = []
        remaining_slots = n_samples
        
        for cls in unique_classes:
            cls_indices = np.where(subset_labels == cls)[0]
            n_take = min(len(cls_indices), max(1, remaining_slots // len(unique_classes)))
            np.random.seed(seed + cls)  # Seed diverso per ogni classe per riproducibilità
            selected = np.random.choice(cls_indices, size=n_take, replace=False)
            debug_indices.extend(selected)
            remaining_slots -= n_take
        
        # Riempi gli slot rimanenti casualmente
        if remaining_slots > 0:
            all_indices = np.arange(len(dataset_subset))
            available = np.setdiff1d(all_indices, debug_indices)
            if len(available) > 0:
                np.random.seed(seed)
                extra = np.random.choice(available, size=min(remaining_slots, len(available)), replace=False)
                debug_indices.extend(extra)
        
        debug_indices = np.array(debug_indices)
    else:
        # Split stratificato normale
        subset_indices = np.arange(len(dataset_subset))
        debug_indices, _ = train_test_split(
            subset_indices,
            train_size=n_samples,
            stratify=subset_labels,
            random_state=seed
        )
    
    # Mappa gli indici del subset agli indici originali del dataset
    original_indices = [dataset_subset.indices[i] for i in debug_indices]
    return Subset(dataset_subset.dataset, original_indices)

def create_kfold_splits_stratified(dataset, n_splits=5, random_state=42, shuffle=True):
    """
    Crea splits K-Fold stratificati per il dataset
    
    Args:
        dataset: Il dataset OrganoidsINRIA3D
        n_splits: Numero di fold (default: 5)
        random_state: Seed per riproducibilità
        shuffle: Se fare shuffle dei dati prima di dividere
    
    Returns:
        List[Tuple]: Lista di tuple (train_indices, val_indices) per ogni fold
    """
    # Ottieni le labels dal dataset
    labels = np.array(dataset.labels)  # Assumendo che dataset.labels sia disponibile
    
    # Verifica la distribuzione delle classi
    print("Distribuzione classi originale:")
    class_counts = Counter(labels)
    for class_id, count in sorted(class_counts.items()):
        print(f"  Classe {class_id}: {count} campioni ({count/len(labels)*100:.1f}%)")
    
    # Crea il StratifiedKFold
    skf = StratifiedKFold(
        n_splits=n_splits,
        shuffle=shuffle,
        random_state=random_state
    )
    
    # Genera i fold
    folds = []
    for fold_idx, (train_idx, val_idx) in enumerate(skf.split(np.arange(len(labels)), labels)):
        # Verifica la distribuzione in ogni fold
        train_labels = labels[train_idx]
        val_labels = labels[val_idx]
        
        print(f"\nFold {fold_idx + 1}:")
        print(f"  Training: {len(train_idx)} campioni")
        train_dist = Counter(train_labels)
        for class_id in sorted(train_dist.keys()):
            print(f"    Classe {class_id}: {train_dist[class_id]} ({train_dist[class_id]/len(train_idx)*100:.1f}%)")
        
        print(f"  Validation: {len(val_idx)} campioni")
        val_dist = Counter(val_labels)
        for class_id in sorted(val_dist.keys()):
            print(f"    Classe {class_id}: {val_dist[class_id]} ({val_dist[class_id]/len(val_idx)*100:.1f}%)")
        
        folds.append((train_idx, val_idx))
    
    return folds

def create_kfold_splits_balanced(
    dataset: Dataset,
    n_splits: int = 5,
    seed: int = 42,
    shuffle: bool = True
) -> List[Tuple[Subset, Subset]]:
    """
    Crea K fold bilanciati: ogni fold ha lo STESSO numero di campioni per classe,
    limitando tutti alla classe minoritaria. Ogni fold è random (seed/ shuffle).
    
    Ritorna lista di (train_subset, val_subset) per ciascun fold.
    """
    labels = np.array(dataset.labels)
    classes, counts = np.unique(labels, return_counts=True)
    min_per_class = counts.min()

    # Limita per bilanciamento: seleziona min_per_class indici per ogni classe
    rng = np.random.default_rng(seed)
    balanced_indices = []
    balanced_labels = []
    for c in classes:
        idx_c = np.where(labels == c)[0]
        # Shuffle e prendi min_per_class
        idx_sel = rng.choice(idx_c, size=min_per_class, replace=False)
        balanced_indices.append(idx_sel)
        balanced_labels.append(np.full(min_per_class, c, dtype=labels.dtype))
    balanced_indices = np.concatenate(balanced_indices)
    balanced_labels = np.concatenate(balanced_labels)

    # Shuffle globale dei bilanciati (per non avere blocchi di classe)
    perm = rng.permutation(len(balanced_indices))
    balanced_indices = balanced_indices[perm]
    balanced_labels = balanced_labels[perm]

    # StratifiedKFold su set perfettamente bilanciato
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=seed)

    folds = []
    for train_idx_rel, val_idx_rel in skf.split(np.arange(len(balanced_labels)), balanced_labels):
        # Mappa agli indici originali del dataset
        train_idx = balanced_indices[train_idx_rel]
        val_idx = balanced_indices[val_idx_rel]

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
        folds.append((train_subset, val_subset))

    # Info utili
    per_fold = (min_per_class * len(classes)) // n_splits
    print(f"[Balanced KFold] classi={len(classes)}, min_per_class={min_per_class}, "
          f"n_splits={n_splits}, ~val_per_fold={per_fold // n_splits if n_splits>0 else 0} per class")

    return folds

def create_fold_dataloaders(dataset, train_idx, val_idx, batch_size=1, num_workers=1):
    """
    Crea i DataLoader per training e validation per un fold specifico
    
    Args:
        dataset: Il dataset completo
        train_idx: Indici per il training set
        val_idx: Indici per il validation set
        batch_size: Batch size
        num_workers: Numero di worker per il DataLoader
    
    Returns:
        Tuple: (train_loader, val_loader)
    """
    # Crea i subset
    train_subset = Subset(dataset, train_idx)
    val_subset = Subset(dataset, val_idx)
    
    # Crea i DataLoader
    train_loader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,  # Shuffle per training
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    val_loader = DataLoader(
        val_subset,
        batch_size=1,  # Spesso batch_size=1 per validation
        shuffle=False,  # No shuffle per validation
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    
    return train_loader, val_loader

def create_kfold_splits_random(
    dataset: Dataset,
    n_splits: int = 5,
    seed: int = 42
) -> List[Tuple[Subset, Subset]]:
    """
    Crea K fold con divisioni casuali (non stratificate).
    Ogni fold usa una porzione diversa del dataset come validation e il resto come training.
    
    Args:
        dataset: Dataset PyTorch
        n_splits: numero di fold
        seed: seed per riproducibilità
    
    Returns:
        List[Tuple[Subset, Subset]]: lista di (train_subset, val_subset) per ogni fold
    """
    n = len(dataset)
    if n_splits < 2:
        raise ValueError("n_splits deve essere >= 2")
    if n_splits > n:
        raise ValueError("n_splits non può superare il numero di campioni del dataset")

    rng = np.random.default_rng(seed)
    indices = np.arange(n)
    rng.shuffle(indices)  # permutazione casuale degli indici

    # Suddividi gli indici in n_splits blocchi il più possibile uguali
    fold_sizes = [n // n_splits] * n_splits
    for i in range(n % n_splits):
        fold_sizes[i] += 1

    folds = []
    current = 0
    for k in range(n_splits):
        val_idx = indices[current: current + fold_sizes[k]]
        current += fold_sizes[k]
        train_idx = np.setdiff1d(indices, val_idx, assume_unique=False)

        train_subset = Subset(dataset, train_idx.tolist())
        val_subset = Subset(dataset, val_idx.tolist())
        folds.append((train_subset, val_subset))

    return folds