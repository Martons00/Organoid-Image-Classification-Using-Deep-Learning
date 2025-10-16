# Aggiungi all'inizio di tools/similarity.py
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def compute_similarity_matrix(features, metric='cosine'):
    """Calcola matrice di similarità tra feature"""
    features = F.normalize(features, p=2, dim=1)  # Normalizza per coseno
    
    if metric == 'cosine':
        # Cosine similarity: A·B / (||A|| ||B||)
        similarity_matrix = torch.mm(features, features.t())
    elif metric == 'euclidean':
        # Distanza euclidea convertita in similarità
        dist_matrix = torch.cdist(features, features, p=2)
        similarity_matrix = 1 / (1 + dist_matrix)  # Converti distanza in similarità
    
    return similarity_matrix.numpy()

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Sequence, Optional, Mapping

def plot_similarity_heatmap(
    similarity_matrix: np.ndarray,
    labels: Optional[Sequence] = None,          # etichette per sample (id o string)
    class_names: Optional[Sequence[str]] = None,# nomi già allineati ai sample
    class_name_map: Optional[Mapping] = None,   # mapping id->nome se labels sono id
    sort_by_class: bool = True,                 # ordina alfabeticamente per class name
    casefold: bool = True,                      # ordina in modo case-insensitive
    save_path: Optional[str] = None,
    title: str = "Feature Similarity Matrix (Cosine Similarity)",
    cmap: str = "coolwarm",
    figsize: tuple = (12, 10),
    show_ticks: bool = True,                    # mostra i tick con i nomi classe
    annot: bool = False,                        # annotazioni nelle celle (spesso troppo denso)
) -> np.ndarray:
    """
    Plotta la heatmap della matrice di similarità riordinando (opzionale) per nome di classe.

    Ritorna l'array di indici `order` applicato a righe/colonne per eventuale riuso.
    """
    sim = np.asarray(similarity_matrix)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity_matrix deve essere quadrata (N x N).")
    n = sim.shape[0]

    # Costruisci i nomi di classe per ogni sample
    # Priorità: class_names -> labels con class_name_map -> labels string -> fallback "Class {label}"
    if class_names is not None:
        if len(class_names) != n:
            raise ValueError("class_names deve avere lunghezza N (numero di sample).")
        names = list(class_names)
    elif labels is not None:
        if len(labels) != n:
            raise ValueError("labels deve avere lunghezza N (numero di sample).")
        if class_name_map is not None:
            names = [str(class_name_map[l]) for l in labels]
        else:
            # Se labels sono già stringhe, usale; altrimenti fallback "Class {id}"
            if len(labels) > 0 and isinstance(labels[0], str):
                names = [str(l) for l in labels]
            else:
                names = [f"Class {l}" for l in labels]
    else:
        # Nessuna informazione: non possiamo ordinare per class name né etichettare
        names = None

    # Calcola l'ordine
    if sort_by_class and names is not None:
        keys = [nm.casefold() if casefold else nm for nm in names]
        order = np.argsort(keys, kind="stable")  # ordina alfabeticamente in modo stabile
    else:
        order = np.arange(n)

    # Applica riordinamento a matrice e nomi
    sim_sorted = sim[np.ix_(order, order)]
    names_sorted = [names[i] for i in order] if names is not None else None

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        sim_sorted,
        cmap=cmap,
        center=0,
        square=True,
        annot=annot,
        fmt=".2f",
        cbar_kws={"shrink": .8},
        xticklabels=(names_sorted if (show_ticks and names_sorted is not None) else False),
        yticklabels=(names_sorted if (show_ticks and names_sorted is not None) else False),
    )
    plt.title(title)
    plt.xlabel("Samples (ordinati per class name)" if sort_by_class else "Samples")
    plt.ylabel("Samples (ordinati per class name)" if sort_by_class else "Samples")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return order
def plot_similarity_heatmap_new(
    similarity_matrix: np.ndarray,
    labels: Optional[Sequence] = None,          # etichette per sample (id o string)
    class_names: Optional[Sequence[str]] = None,# nomi già allineati ai sample
    class_name_map: Optional[Mapping] = None,   # mapping id->nome se labels sono id
    sort_by_class: bool = True,                 # ordina alfabeticamente per class name
    casefold: bool = True,                      # ordina in modo case-insensitive
    save_path: Optional[str] = None,
    title: str = "Feature Similarity Matrix (Cosine Similarity)",
    cmap: str = "coolwarm",
    figsize: tuple = (12, 10),
    show_ticks: bool = True,                    # mostra i tick con i nomi classe
    annot: bool = False,                        # annotazioni nelle celle (spesso troppo denso)
) -> np.ndarray:
    """
    Plotta la heatmap della matrice di similarità riordinando (opzionale) per nome di classe.

    Ritorna l'array di indici `order` applicato a righe/colonne per eventuale riuso.
    """
    import numpy as np
    import seaborn as sns
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    sim = np.asarray(similarity_matrix)
    if sim.ndim != 2 or sim.shape[0] != sim.shape[1]:
        raise ValueError("similarity_matrix deve essere quadrata (N x N).")
    n = sim.shape[0]

    # Costruisci i nomi di classe per ogni sample
    if class_names is not None:
        if len(class_names) != n:
            raise ValueError("class_names deve avere lunghezza N (numero di sample).")
        names = list(class_names)
    elif labels is not None:
        if len(labels) != n:
            raise ValueError("labels deve avere lunghezza N (numero di sample).")
        if class_name_map is not None:
            names = [str(class_name_map[l]) for l in labels]
        else:
            if len(labels) > 0 and isinstance(labels[0], str):
                names = [str(l) for l in labels]
            else:
                names = [f"Class {l}" for l in labels]
    else:
        names = None

    # Calcola l'ordine
    if sort_by_class and names is not None:
        keys = [nm.casefold() if casefold else nm for nm in names]
        order = np.argsort(keys, kind="stable")  # ordina alfabeticamente in modo stabile
    else:
        order = np.arange(n)

    # Applica riordinamento a matrice e nomi
    sim_sorted = sim[np.ix_(order, order)]
    names_sorted = [names[i] for i in order] if names is not None else None

    # Calibra la colorbar sui fuori-diagonale per evitare saturazione
    off = sim_sorted.copy()
    np.fill_diagonal(off, np.nan)
    # Percentili robusti per non farsi dominare da outlier/residui
    vmin = np.nanpercentile(off, 2)
    vmax = np.nanpercentile(off, 98)

    # Se i dati attraversano 0, usa una normalizzazione diverging centrata a 0; altrimenti usa vmin/vmax
    crosses_zero = (np.nanmin(off) < 0.0) and (np.nanmax(off) > 0.0)
    heatmap_kwargs = {}
    if crosses_zero:
        heatmap_kwargs["norm"] = TwoSlopeNorm(vmin=min(vmin, 0.0), vcenter=0.0, vmax=max(vmax, 0.0))
    else:
        heatmap_kwargs["vmin"] = vmin
        heatmap_kwargs["vmax"] = vmax

    # Plot
    plt.figure(figsize=figsize)
    sns.heatmap(
        sim_sorted,
        cmap=cmap,
        square=True,
        annot=annot,
        fmt=".2f",
        # rimuoviamo center=0; viene usato solo se serve via norm
        cbar_kws={"shrink": .8},
        xticklabels=(names_sorted if (show_ticks and names_sorted is not None) else False),
        yticklabels=(names_sorted if (show_ticks and names_sorted is not None) else False),
        **heatmap_kwargs
    )
    plt.title(title)
    plt.xlabel("Samples (ordinati per class name)" if sort_by_class else "Samples")
    plt.ylabel("Samples (ordinati per class name)" if sort_by_class else "Samples")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()
    else:
        plt.show()

    return order

# Uso
#similarity_matrix = compute_similarity_matrix(features, metric='cosine')
#plot_similarity_heatmap(similarity_matrix, labels)
