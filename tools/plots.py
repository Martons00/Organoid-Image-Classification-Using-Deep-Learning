import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Dict, Optional

def plot_training_curve(
    values: Sequence[float],
    metric_name: str = "Loss",
    epochs: Sequence[int] = None,
    title: str = None,
    figsize: tuple = (8, 5),
    save_path: str = None,
    max_xticks: int = 20
) -> None:
    """
    Plotta la curva di training di loss o accuracy in funzione delle epoche.

    Args:
        values: lista o array di valori (loss o accuracy) per ogni epoca.
        metric_name: nome del metrica mostrata sull'asse y ("Loss" o "Accuracy").
        epochs: lista o array di numeri di epoca; se None, usa range(len(values)).
        title: titolo del grafico; se None, usa f"{metric_name} vs Epoch".
        figsize: dimensione della figura (width, height).
        save_path: percorso file per salvare il grafico; se None, mostra a schermo.
        max_xticks: numero massimo di tick sull'asse x (default: 10).
    """
    if epochs is None:
        epochs = list(range(1, len(values) + 1))
    if title is None:
        title = f"{metric_name} vs Epoche"

    plt.figure(figsize=figsize)
    plt.plot(epochs, values, marker='o', linestyle='-', color='C0')
    plt.xlabel("Epoca")
    plt.ylabel(metric_name)
    plt.title(title)
    plt.grid(True)
    
    # Gestione intelligente degli xticks
    if len(epochs) <= max_xticks:
        plt.xticks(epochs)
    else:
        # Calcola step per avere circa max_xticks tick
        step = max(1, len(epochs) // max_xticks)
        tick_positions = epochs[::step]
        # Assicurati di includere sempre l'ultima epoca
        if epochs[-1] not in tick_positions:
            tick_positions.append(epochs[-1])
        plt.xticks(tick_positions)
    
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300)
        plt.close()
    else:
        plt.show()



def plot_multi_class_training_curve(
    avg_acc: Sequence[float],
    per_class_acc: Sequence[Dict[str, float]],
    epochs: Sequence[int] = None,
    title: str = None,
    figsize: tuple = (10, 6),
    save_path: str = None,
    max_xticks: int = 20
) -> None:
    """
    Plotta le curve di training per accuracy media e per-class in funzione delle epoche.
    
    Args:
        avg_acc: lista o array di valori di accuracy media per ogni epoca.
        per_class_acc: lista di dizionari contenenti le accuracies per classe per ogni epoca.
                      Formato: [{'c0': 0.0, 'c1': 0.0, 'c2': 1.0}, ...]
        epochs: lista o array di numeri di epoca; se None, usa range(len(avg_acc)).
        title: titolo del grafico; se None, usa "Training Accuracy vs Epoche".
        figsize: dimensione della figura (width, height).
        save_path: percorso file per salvare il grafico; se None, mostra a schermo.
        max_xticks: numero massimo di tick sull'asse x (default: 20).
    """
    if epochs is None:
        epochs = list(range(1, len(avg_acc) + 1))
    if title is None:
        title = "Training Accuracy vs Epoche"
    
    # Estrai i nomi delle classi dal primo dizionario
    class_names = list(per_class_acc[0].keys())
    
    # Crea arrays per ogni classe
    class_accuracies = {}
    for class_name in class_names:
        class_accuracies[class_name] = [epoch_acc[class_name] for epoch_acc in per_class_acc]
    
    plt.figure(figsize=figsize)
    
    # Plotta accuracy media
    plt.plot(epochs, avg_acc, marker='o', linestyle='-', linewidth=2, 
             label='Average Accuracy', color='black')
    
    # Plotta accuracies per classe
    colors = ['C1', 'C2', 'C3', 'C4', 'C5', 'C6', 'C7', 'C8', 'C9']  # Colori diversi per ogni classe
    for i, (class_name, acc_values) in enumerate(class_accuracies.items()):
        color = colors[i % len(colors)]
        plt.plot(epochs, acc_values, marker='s', linestyle='--', 
                 label=f'Class {class_name}', color=color)
    
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.legend()
    
    # Gestione intelligente degli xticks
    if len(epochs) <= max_xticks:
        plt.xticks(epochs)
    else:
        # Calcola step per avere circa max_xticks tick
        step = max(1, len(epochs) // max_xticks)
        tick_positions = epochs[::step]
        # Assicurati di includere sempre l'ultima epoca
        if epochs[-1] not in tick_positions:
            tick_positions.append(epochs[-1])
        plt.xticks(tick_positions)
    
    # Imposta i limiti dell'asse y da 0 a 1 per le accuracies
    plt.ylim(0, 1.05)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()

import matplotlib.pyplot as plt
import numpy as np
from typing import Sequence, Optional

def _ema(x: np.ndarray, alpha: float) -> np.ndarray:
    y = np.empty_like(x, dtype=float)
    y[0] = x[0]
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i-1]
    return y

def plot_loss_lr(
    loss: Sequence[float],
    lr: Sequence[float],
    epochs: Optional[Sequence[int]] = None,
    title: Optional[str] = None,
    figsize: tuple = (12, 4),
    save_path: Optional[str] = None,
    max_xticks: int = 20,
    kind: str = "all",           # "twin", "loss_vs_lr", "delta_vs_lr", "all"
    smooth_alpha: Optional[float] = None,  # 0<alpha<1 per smoothing EMA della loss
) -> None:
    """
    Crea plot informativi a partire da loss per epoca e learning rate per epoca.

    kind:
      - "twin": loss (asse sinistro) + lr (asse destro) vs epoche
      - "loss_vs_lr": loss in funzione del lr (asse x in log)
      - "delta_vs_lr": -Δloss per epoca vs lr
      - "all": pannello 1x3 con tutte le viste
    """
    loss = np.asarray(loss, dtype=float)
    lr = np.asarray(lr, dtype=float)
    assert loss.ndim == 1 and lr.ndim == 1, "loss e lr devono essere 1-D"
    assert len(loss) == len(lr), "loss e lr devono avere la stessa lunghezza"
    n = len(loss)

    if epochs is None:
        epochs = np.arange(1, n + 1)
    else:
        epochs = np.asarray(epochs)
        assert len(epochs) == n, "epochs deve avere la stessa lunghezza di loss/lr"

    # Smoothing opzionale sulla loss (EMA)
    loss_plot = loss.copy()
    if smooth_alpha is not None:
        if not (0 < smooth_alpha < 1):
            raise ValueError("smooth_alpha deve essere in (0,1)")
        loss_plot = _ema(loss_plot, smooth_alpha)

    # Delta loss per epoca
    delta_loss = np.diff(loss, prepend=loss[0])

    def _set_xticks(ax):
        if len(epochs) <= max_xticks:
            ax.set_xticks(epochs)
        else:
            step = max(1, len(epochs) // max_xticks)
            ticks = epochs[::step]
            if epochs[-1] not in ticks:
                ticks = np.append(ticks, epochs[-1])
            ax.set_xticks(ticks)

    if kind == "twin":
        fig, ax1 = plt.subplots(figsize=figsize)
        ax2 = ax1.twinx()
        ax1.plot(epochs, loss_plot, color='C0', marker='o', label='Loss')
        ax2.plot(epochs, lr,        color='C1', marker='s', label='LR')
        ax1.set_xlabel('Epoca'); ax1.set_ylabel('Loss', color='C0')
        ax2.set_ylabel('LR', color='C1')
        _set_xticks(ax1)
        ax1.grid(True, alpha=0.3)
        if title is None: title = 'Loss e LR per epoca'
        ax1.set_title(title)
        lines = ax1.get_lines() + ax2.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best')
        fig.tight_layout()

    elif kind == "loss_vs_lr":
        plt.figure(figsize=figsize)
        plt.semilogx(lr, loss_plot, marker='o', color='C0')
        plt.xlabel('Learning Rate (log)')
        plt.ylabel('Loss')
        plt.grid(True, which='both', alpha=0.3)
        if title is None: title = 'Loss vs Learning Rate'
        plt.title(title)
        plt.tight_layout()

    elif kind == "delta_vs_lr":
        plt.figure(figsize=figsize)
        plt.plot(lr, -delta_loss, marker='s', color='C2')
        plt.xlabel('Learning Rate')
        plt.ylabel('-ΔLoss per epoca')
        plt.grid(True, alpha=0.3)
        if title is None: title = '-ΔLoss vs Learning Rate'
        plt.title(title)
        plt.tight_layout()

    elif kind == "all":
        fig, axes = plt.subplots(1, 3, figsize=(figsize[0]*1.6, figsize[1]))
        # 1) Twin axes
        ax1 = axes[0]; ax1b = ax1.twinx()
        ax1.plot(epochs, loss_plot, color='C0', marker='o', label='Loss')
        ax1b.plot(epochs, lr,        color='C1', marker='s', label='LR')
        ax1.set_xlabel('Epoca'); ax1.set_ylabel('Loss', color='C0')
        ax1b.set_ylabel('LR', color='C1')
        _set_xticks(ax1)
        ax1.grid(True, alpha=0.3)
        ax1.set_title('Loss & LR (assi gemelli)')
        lines = ax1.get_lines() + ax1b.get_lines()
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='best', fontsize=9)
        # 2) Loss vs LR (log x)
        ax2 = axes[1]
        ax2.semilogx(lr, loss_plot, marker='o', color='C0')
        ax2.set_xlabel('Learning Rate (log)'); ax2.set_ylabel('Loss')
        ax2.grid(True, which='both', alpha=0.3)
        ax2.set_title('Loss vs LR (log x)')
        # 3) -ΔLoss vs LR
        ax3 = axes[2]
        ax3.plot(lr, -delta_loss, marker='s', color='C2')
        ax3.set_xlabel('Learning Rate'); ax3.set_ylabel('-ΔLoss per epoca')
        ax3.grid(True, alpha=0.3)
        ax3.set_title('-ΔLoss vs LR')
        if title is None: title = 'Diagnostica LR e dinamica della Loss'
        fig.suptitle(title, y=1.02)
        fig.tight_layout()

    else:
        raise ValueError("kind deve essere uno tra {'twin','loss_vs_lr','delta_vs_lr','all'}")

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
