import torch
import numpy as np
from copy import deepcopy

class EarlyStopping:
    def __init__(self, mode='min', patience=10, min_delta=0.0, restore_best=True, verbose=True):
        """
        mode: 'min' per loss, 'max' per accuracy/F1
        patience: epoche senza miglioramento prima di fermare
        min_delta: miglioramento minimo per essere considerato
        restore_best: se True, ripristina i pesi migliori al termine
        """
        assert mode in ('min', 'max')
        self.mode = mode
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.verbose = verbose

        self.best_score = None
        self.num_bad_epochs = 0
        self.should_stop = False
        self.best_state = None

    def step(self, current, model=None):
        """
        Aggiorna lo stato con la metrica corrente.
        current: valore della metrica monitorata
        model: modello da salvare per il best state
        """
        score = current if self.mode == 'max' else -current  # max-ify

        if self.best_score is None:
            self.best_score = score
            if model is not None and self.restore_best:
                self.best_state = deepcopy(model.state_dict())
            if self.verbose:
                print(f"[EarlyStopping] init best={current:.6f}")
            return False

        improvement = score - self.best_score
        if improvement > self.min_delta:
            self.best_score = score
            self.num_bad_epochs = 0
            if model is not None and self.restore_best:
                self.best_state = deepcopy(model.state_dict())
            if self.verbose:
                print(f"[EarlyStopping] improved to {current:.6f}")
        else:
            self.num_bad_epochs += 1
            if self.verbose:
                print(f"[EarlyStopping] no improve ({self.num_bad_epochs}/{self.patience})")
            if self.num_bad_epochs >= self.patience:
                self.should_stop = True
        return self.should_stop

    def restore(self, model):
        if self.restore_best and self.best_state is not None:
            model.load_state_dict(self.best_state)
            if self.verbose:
                print("[EarlyStopping] restored best weights")

