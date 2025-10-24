import numpy as np
from typing import Dict, Any, List, Optional, Sequence
import matplotlib.pyplot as plt

def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: Optional[Sequence] = None,
    normalize: Optional[str] = None,   # opzioni: None, "true", "pred", "all"
    title: Optional[str] = None,
    figsize: tuple = (8, 6),
    save_path: Optional[str] = None,
    cmap: str = "Blues",
    show_colorbar: bool = True,
    annot: bool = True,
    fmt: str = ".2f",
    rotate_xticks: int = 45
) -> None:
    """
    Plotta una confusion matrix già calcolata.

    Args:
        cm: matrice di confusione shape (num_classes, num_classes).
        class_names: etichette asse x/y; se None, usa range(num_classes).
        normalize: None (conteggi), "true" (normalizza per riga), 
                   "pred" (normalizza per colonna), "all" (normalizza globale).
        title: titolo del grafico; default "Confusion Matrix" (+ suffisso normalizzazione).
        figsize: dimensioni figura (width, height).
        save_path: percorso file per salvare il grafico; se None, mostra a schermo.
        cmap: colormap Matplotlib.
        show_colorbar: se True, mostra la colorbar.
        annot: se True, scrive i valori nelle celle.
        fmt: formato dei valori annotati quando normalizzati (es. ".2f").
        rotate_xticks: rotazione etichette asse x in gradi.
    """
    cm = np.asarray(cm)
    assert cm.ndim == 2 and cm.shape[0] == cm.shape[1], "cm deve essere una matrice quadrata"

    num_classes = cm.shape[0]
    if class_names is None:
        class_names = np.arange(num_classes)

    # Normalizzazione opzionale
    cm_plot = cm.astype(np.float64)
    norm_suffix = ""
    if normalize is not None:
        if normalize == "true":
            denom = cm_plot.sum(axis=1, keepdims=True)
            cm_plot = np.divide(cm_plot, denom, out=np.zeros_like(cm_plot), where=denom > 0)
            norm_suffix = " (Normalized by True)"
        elif normalize == "pred":
            denom = cm_plot.sum(axis=0, keepdims=True)
            cm_plot = np.divide(cm_plot, denom, out=np.zeros_like(cm_plot), where=denom > 0)
            norm_suffix = " (Normalized by Pred)"
        elif normalize == "all":
            total = cm_plot.sum()
            cm_plot = cm_plot / total if total > 0 else cm_plot
            norm_suffix = " (Normalized Overall)"
        else:
            raise ValueError("normalize deve essere None, 'true', 'pred' o 'all'")

    # Titolo
    if title is None:
        title = f"Confusion Matrix{norm_suffix}"

    # Plot
    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_plot, interpolation="nearest", cmap=cmap)
    ax.set_title(title)
    ax.set_xlabel("Predetta")
    ax.set_ylabel("Reale")

    # Tick e etichette
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    plt.setp(ax.get_xticklabels(), rotation=rotate_xticks, ha="right")

    # Colorbar
    if show_colorbar:
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    # Annotazioni nelle celle
    if annot:
        # Soglia per colore del testo
        thresh = cm_plot.max() / 2.0 if cm_plot.size > 0 else 0.0
        for i in range(num_classes):
            for j in range(num_classes):
                val = cm_plot[i, j]
                if normalize is None:
                    text_str = f"{int(cm[i, j])}"
                else:
                    text_str = format(val, fmt)
                ax.text(
                    j, i, text_str,
                    ha="center", va="center",
                    color="white" if val > thresh else "black"
                )

    fig.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()


def metrics_from_confusion_matrix(cm: np.ndarray, zero_division: float = 0.0) -> Dict[str, Any]:
    """
    Calcola metriche di classificazione da una matrice di confusione N×N.
    Restituisce:
      - accuracy
      - per_class: dict con precision, recall (sensibilità), specificity (TNR), f1 per ciascuna classe
      - macro_avg, weighted_avg, micro_avg: aggregazioni delle metriche
    Parametri:
      - cm: matrice di confusione (shape: [n_classes, n_classes])
      - zero_division: valore da usare quando il denominatore è zero
    """
    cm = np.asarray(cm, dtype=np.int64)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1]:
        raise ValueError("La matrice di confusione deve essere quadrata N×N.")

    n_classes = cm.shape[0]
    total = cm.sum()
    tp = np.diag(cm).astype(np.float64)
    fp = cm.sum(axis=0).astype(np.float64) - tp
    fn = cm.sum(axis=1).astype(np.float64) - tp
    tn = total - (tp + fp + fn)

    def safe_div(num, den):
        out = np.divide(num, den, out=np.full_like(num, fill_value=zero_division, dtype=np.float64), where=(den != 0))
        return out

    # Per-classe
    precision_c = safe_div(tp, tp + fp)
    recall_c    = safe_div(tp, tp + fn)  # sensitivity/TPR
    specificity_c = safe_div(tn, tn + fp)  # TNR
    f1_c = safe_div(2 * precision_c * recall_c, precision_c + recall_c)

    # Accuracy globale
    accuracy = float(tp.sum() / total) if total > 0 else float(zero_division)

    # Support (per true class)
    support = cm.sum(axis=1).astype(np.float64)
    weights = safe_div(support, support.sum())

    # Macro / Weighted
    macro = {
        "precision": float(np.nanmean(precision_c)),
        "recall": float(np.nanmean(recall_c)),
        "specificity": float(np.nanmean(specificity_c)),
        "f1": float(np.nanmean(f1_c)),
    }
    weighted = {
        "precision": float(np.nansum(precision_c * weights)),
        "recall": float(np.nansum(recall_c * weights)),
        "specificity": float(np.nansum(specificity_c * weights)),
        "f1": float(np.nansum(f1_c * weights)),
    }

    # Micro (global)
    TPg = tp.sum()
    FPg = fp.sum()
    FNg = fn.sum()
    TNg = tn.sum()
    micro_precision = float(TPg / (TPg + FPg)) if (TPg + FPg) > 0 else float(zero_division)
    micro_recall = float(TPg / (TPg + FNg)) if (TPg + FNg) > 0 else float(zero_division)
    micro_specificity = float(TNg / (TNg + FPg)) if (TNg + FPg) > 0 else float(zero_division)
    micro_f1 = (2 * micro_precision * micro_recall / (micro_precision + micro_recall)
                if (micro_precision + micro_recall) > 0 else float(zero_division))

    per_class = {
        int(i): {
            "precision": float(precision_c[i]),
            "recall": float(recall_c[i]),
            "specificity": float(specificity_c[i]),
            "f1": float(f1_c[i]),
            "support": int(support[i]),
        }
        for i in range(n_classes)
    }

    return {
        "accuracy": accuracy,
        "per_class": per_class,
        "macro_avg": macro,
        "weighted_avg": weighted,
        "micro_avg": {
            "precision": micro_precision,
            "recall": micro_recall,
            "specificity": micro_specificity,
            "f1": micro_f1,
        },
    }


def format_print_metrics(metrics: Dict[str, Any],
                         class_names: Optional[List[str]] = None,
                         digits: int = 3,
                         print_out: bool = False) -> str:
    """
    Formatta e (opzionalmente) stampa:
      - Accuracy globale
      - Tabella per-classe con: precision, recall, specificity (TNR), F1, support
    Parametri:
      - metrics: dizionario come restituito da metrics_from_confusion_matrix(...)
      - class_names: nomi da visualizzare per ciascuna classe (opzionale)
      - digits: cifre decimali per i float
      - print_out: se True esegue print, altrimenti solo restituisce la stringa
    """
    if not isinstance(metrics, dict):
        raise ValueError("metrics deve essere un dict prodotto dal calcolo sulle metriche.")
    if "per_class" not in metrics or "accuracy" not in metrics:
        raise ValueError("metrics deve contenere 'accuracy' e 'per_class'.")

    per_class = metrics["per_class"]
    # Ordina per indice di classe se le chiavi sono numeriche
    try:
        idxs = sorted(per_class.keys())
    except Exception:
        idxs = list(per_class.keys())

    n_classes = len(idxs)
    if class_names is not None and len(class_names) != n_classes:
        raise ValueError(f"class_names ha lunghezza {len(class_names)} ma ci sono {n_classes} classi.")

    names = class_names if class_names is not None else [str(i) for i in idxs]

    # Header tabella
    header = f"{'class':<20}{'precision':>12}{'recall':>10}{'specificity':>14}{'f1':>8}{'support':>10}"
    line = "-" * len(header)

    # Righe per-classe
    rows = [header, line]
    for pos, i in enumerate(idxs):
        c = per_class[i]
        rows.append(
            f"{names[pos]:<20}"
            f"{c.get('precision', 0.0):>12.{digits}f}"
            f"{c.get('recall', 0.0):>10.{digits}f}"
            f"{c.get('specificity', 0.0):>14.{digits}f}"
            f"{c.get('f1', 0.0):>8.{digits}f}"
            f"{int(c.get('support', 0)):>10d}"
        )

    acc = metrics["accuracy"]
    out = []
    out.append(f"Accuracy: {acc:.{digits}f}")
    out.extend(rows)
    formatted = "\n".join(out)

    if print_out:
        print(formatted)

    return formatted

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors, cm
from typing import Dict, Any, List, Optional, Tuple

def plot_metrics_table(metrics: Dict[str, Any],
                       class_names: Optional[List[str]] = None,
                       show_averages: bool = True,
                       decimals: int = 3,
                       cmap: str = "Blues",
                       figsize: Optional[Tuple[float, float]] = None,
                       title: Optional[str] = None,
                       save_path: Optional[str] = None) -> None:
    """
    Crea una tabella con Matplotlib che mostra per classe:
    precision, recall (sensibilità), specificity (TNR), F1 e support; opzionalmente aggiunge macro/weighted/micro.
    """
    if "per_class" not in metrics or "accuracy" not in metrics:
        raise ValueError("Il dict metrics deve contenere 'accuracy' e 'per_class'.")
    per_class = metrics["per_class"]
    idxs = sorted(per_class.keys(), key=lambda x: int(x) if isinstance(x, (int, np.integer)) or str(x).isdigit() else str(x))
    n_classes = len(idxs)

    if class_names is not None and len(class_names) != n_classes:
        raise ValueError(f"class_names ha lunghezza {len(class_names)} ma ci sono {n_classes} classi.")
    names = class_names if class_names is not None else [str(i) for i in idxs]

    # Colonne
    col_labels = ["class", "precision", "recall", "specificity", "f1", "support"]

    # Righe per-classe
    cell_text = []
    for pos, i in enumerate(idxs):
        c = per_class[i]
        row = [
            names[pos],
            f"{c.get('precision', 0.0):.{decimals}f}",
            f"{c.get('recall', 0.0):.{decimals}f}",
            f"{c.get('specificity', 0.0):.{decimals}f}",
            f"{c.get('f1', 0.0):.{decimals}f}",
            f"{int(c.get('support', 0))}",
        ]
        cell_text.append(row)

    # Aggiungi medie
    if show_averages:
        # Riga vuota separatrice
        cell_text.append(["", "", "", "", "", ""])
        def fmt_avg(block, label):
            return [
                label,
                f"{block.get('precision', 0.0):.{decimals}f}",
                f"{block.get('recall', 0.0):.{decimals}f}",
                f"{block.get('specificity', 0.0):.{decimals}f}",
                f"{block.get('f1', 0.0):.{decimals}f}",
                f"{sum(int(per_class[i].get('support', 0)) for i in idxs)}",
            ]
        if "macro_avg" in metrics:
            cell_text.append(fmt_avg(metrics["macro_avg"], "macro avg"))
        if "weighted_avg" in metrics:
            cell_text.append(fmt_avg(metrics["weighted_avg"], "weighted avg"))
        if "micro_avg" in metrics:
            # micro support = totale esempi
            total_support = sum(int(per_class[i].get('support', 0)) for i in idxs)
            micro = metrics["micro_avg"]
            cell_text.append([
                "micro avg",
                f"{micro.get('precision', 0.0):.{decimals}f}",
                f"{micro.get('recall', 0.0):.{decimals}f}",
                f"{micro.get('specificity', 0.0):.{decimals}f}",
                f"{micro.get('f1', 0.0):.{decimals}f}",
                f"{total_support}",
            ])

    # Figura
    n_rows = len(cell_text) + 1  # + header
    if figsize is None:
        # altezza proporzionale alle righe
        fig_h = max(2.5, 0.4 * n_rows )
        figsize = (10, fig_h)
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis("off")

    # Tabella
    tbl = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center", colLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(10)
    tbl.scale(1, 1.2)

    # Colori per celle numeriche nelle colonne [precision, recall, specificity, f1]
    rate_cols = [1, 2, 3, 4]
    norm = colors.Normalize(vmin=0.0, vmax=1.0)
    cmap_obj = cm.get_cmap(cmap)

    # Evidenzia le metriche di tasso (0..1); lascia 'class' e 'support' bianche
    for r in range(1, n_rows):  # r=0 è l'header
        for c in rate_cols:
            text_val = tbl[(r, c)].get_text().get_text()
            try:
                val = float(text_val)
            except ValueError:
                continue
            tbl[(r, c)].set_facecolor(cmap_obj(norm(val)))
        # 'support' in grigio chiaro
        c_support = len(col_labels) - 1
        tbl[(r, c_support)].set_facecolor("#F5F5F5")

    # Titolo con accuracy
    acc = metrics.get("accuracy", None)
    t = title if title is not None else "Classification metrics (per class)"
    if acc is not None:
        t = f"{t} — Accuracy: {acc:.{decimals}f}"
    ax.set_title(t, fontsize=12, pad=10)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
    else:
        plt.show()
