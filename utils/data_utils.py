# Copyright 2020 - 2022 MONAI Consortium
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import math
import os

from typing import Tuple, Union, Optional
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, random_split
from sklearn.model_selection import train_test_split

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

from typing import Tuple, Union, Optional, List
import numpy as np
import torch
from torch.utils.data import Dataset, Subset, DataLoader
from sklearn.model_selection import train_test_split

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
