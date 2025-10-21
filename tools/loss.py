import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        # Usa CrossEntropyLoss senza reduction per ottenere loss per sample
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)  # probabilità predetta per la classe corretta
        
        # Applica focal weight
        focal_weight = (1 - pt) ** self.gamma
        focal_loss = focal_weight * ce_loss
        
        # Applica class weights usando gather per selezionare il peso corretto
        if self.alpha is not None:
            if isinstance(self.alpha, (list, tuple)):
                alpha_tensor = torch.tensor(self.alpha, dtype=inputs.dtype, device=inputs.device)
            else:
                alpha_tensor = self.alpha
            
            # CORREZIONE: usa gather per selezionare i pesi corretti per ogni sample
            alpha_t = alpha_tensor.gather(0, targets.long())
            focal_loss = alpha_t * focal_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class LabelSmoothingLoss(nn.Module):
    def __init__(self, classes, smoothing=0.1, weight=None):
        super(LabelSmoothingLoss, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.cls = classes
        self.weight = weight
        
    def forward(self, pred, target):
        pred = pred.log_softmax(dim=-1)
        
        with torch.no_grad():
            true_dist = torch.zeros_like(pred)
            true_dist.fill_(self.smoothing / (self.cls - 1))
            true_dist.scatter_(1, target.data.unsqueeze(1), self.confidence)
            
        loss = torch.sum(-true_dist * pred, dim=-1)
        
        if self.weight is not None:
            weight_expanded = self.weight[target]
            loss = loss * weight_expanded
            
        return loss.mean()


class DiversityLoss(nn.Module):
    def __init__(self, base_loss, diversity_weight=0.1):
        super(DiversityLoss, self).__init__()
        self.base_loss = base_loss
        self.diversity_weight = diversity_weight
        
    def forward(self, logits, targets):
        # Loss principale
        main_loss = self.base_loss(logits, targets)
        
        # Calcola distribuzione delle predizioni nel batch
        probs = F.softmax(logits, dim=1)
        batch_mean_probs = probs.mean(dim=0)  # media delle prob per classe
        
        # Penalità per bassa entropia (predizioni troppo concentrate)
        entropy_loss = -torch.sum(batch_mean_probs * torch.log(batch_mean_probs + 1e-8))
        diversity_penalty = -entropy_loss  # vogliamo massimizzare l'entropia
        
        return main_loss + self.diversity_weight * diversity_penalty


class CombinedLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, diversity_weight=0.1):
        super(CombinedLoss, self).__init__()
        self.focal_loss = FocalLoss(alpha=alpha, gamma=gamma)
        self.diversity_weight = diversity_weight
        
    def forward(self, logits, targets):
        # Focal loss (ora corretta)
        focal = self.focal_loss(logits, targets)
        
        # Diversity penalty
        probs = F.softmax(logits, dim=1)
        batch_dist = probs.mean(dim=0)
        entropy = -torch.sum(batch_dist * torch.log(batch_dist + 1e-8))
        diversity_penalty = -entropy
        
        return focal + self.diversity_weight * diversity_penalty



class CenterLoss(nn.Module):
    def __init__(self, num_classes, feat_dim, lambda_c=0.5):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.lambda_c = lambda_c
        self.centers = nn.Parameter(torch.randn(num_classes, feat_dim))
        
    def forward(self, features, labels):
        batch_size = features.size(0)
        centers_batch = self.centers[labels]
        criterion = nn.MSELoss()
        center_loss = criterion(features, centers_batch)
        return center_loss

# Uso combinato
def combined_loss(logits, features, targets):
    ce_loss = F.cross_entropy(logits, targets)
    center_loss = center_criterion(features, targets)
    return ce_loss + 0.5 * center_loss

import torch
import torch.nn.functional as F

def similarity_margin_loss(sim: torch.Tensor,
                           labels: torch.Tensor,
                           pos_margin: float = 0.5,
                           neg_margin: float = 0.0,
                           exclude_diag: bool = True,
                           reduction: str = "mean",
                           balance: str = "sqrt"):
    """
    sim: (N,N) similarità (es. cosine su embedding L2-normalizzati) in [-1,1] o [0,1]
    labels: (N,) int64
    pos_margin m+: target min per coppie positive
    neg_margin m-: target max per coppie negative
    balance: 'none' | 'linear' | 'sqrt' per compensare #positivi/#negativi
    """
    assert sim.ndim == 2 and sim.shape[0] == sim.shape[1], "sim deve essere NxN"
    N = sim.shape[0]
    device = sim.device
    y = labels.view(-1, 1)
    pos_mask = (y == y.T).to(sim.dtype)
    neg_mask = 1.0 - pos_mask

    if exclude_diag:
        eye = torch.eye(N, device=device, dtype=sim.dtype)
        pos_mask = pos_mask - eye
        pos_mask = pos_mask.clamp_min(0.0)
        neg_mask = neg_mask.clamp_min(0.0)

    # Hinge quadratico
    pos_term = F.relu(pos_margin - sim)**2
    neg_term = F.relu(sim - neg_margin)**2

    # Pesi per bilanciare
    n_pos = pos_mask.sum().clamp(min=1.0)
    n_neg = neg_mask.sum().clamp(min=1.0)

    if balance == "linear":
        w_pos = (n_pos + n_neg) / n_pos
        w_neg = (n_pos + n_neg) / n_neg
    elif balance == "sqrt":
        w_pos = torch.sqrt((n_pos + n_neg) / n_pos)
        w_neg = torch.sqrt((n_pos + n_neg) / n_neg)
    else:
        w_pos = 1.0
        w_neg = 1.0

    loss_pos = (pos_term * pos_mask).sum() / n_pos
    loss_neg = (neg_term * neg_mask).sum() / n_neg
    loss = w_pos * loss_pos + w_neg * loss_neg

    if reduction == "sum":
        return loss * (n_pos + n_neg) / (n_pos + n_neg)  # identità per compatibilità
    elif reduction == "none":
        # Restituisce mappa pesata per debug
        return w_pos * pos_term * pos_mask + w_neg * neg_term * neg_mask
    else:
        return loss

def supervised_contrastive_from_similarity(sim: torch.Tensor,
                                           labels: torch.Tensor,
                                           temperature: float = 0.07,
                                           exclude_diag: bool = True,
                                           eps: float = 1e-12):
    """
    sim: (N,N) matrice di similarità (es. cosine) non necessariamente normalizzata
    labels: (N,)
    """
    assert sim.ndim == 2 and sim.shape[0] == sim.shape[1], "sim deve essere NxN"
    N = sim.shape[0]
    device = sim.device
    y = labels.view(-1, 1)

    # Maschere
    pos_mask = (y == y.T).to(sim.dtype)
    if exclude_diag:
        pos_mask = pos_mask - torch.eye(N, device=device, dtype=sim.dtype)
        pos_mask = pos_mask.clamp_min(0.0)

    # Logits con temperatura
    logits = sim / max(temperature, eps)

    # Maschera per escludere self-contrast nel denominatore
    denom_mask = torch.ones_like(logits, dtype=logits.dtype, device=device)
    if exclude_diag:
        denom_mask = denom_mask - torch.eye(N, device=device, dtype=logits.dtype)
        denom_mask = denom_mask.clamp_min(0.0)

    # Log-softmax su j con maschera
    # Per evitare overflow: sottraiamo il max per riga prima di exp
    logits = logits - logits.max(dim=1, keepdim=True).values
    exp_logits = torch.exp(logits) * denom_mask
    denom = exp_logits.sum(dim=1, keepdim=True).clamp_min(eps)
    log_prob = logits - torch.log(denom)

    # Media su positivi per anchor
    pos_count = pos_mask.sum(dim=1)  # |P(i)|
    # Evita divisione per zero: consideriamo solo anchor con >=1 positivo
    valid = pos_count > 0
    if valid.any():
        loss_i = -(pos_mask[valid] * log_prob[valid]).sum(dim=1) / pos_count[valid].clamp_min(1)
        return loss_i.mean()
    else:
        # Nessuna coppia positiva nel batch
        return torch.tensor(0.0, device=device, requires_grad=True)


if __name__ == "__main__":
    # Esempio di test
    logits = torch.randn(8, 3, requires_grad=True)
    features = torch.randn(8, 128, requires_grad=True)
    targets = torch.randint(0, 3, (8,))


    # Uso nel tuo codice
    focal_alpha = [1.0, 1.2, 0.8]  # pesi per le 3 classi
    loss_func = FocalLoss(alpha=focal_alpha, gamma=2.0)


    # Uso
    loss_func = LabelSmoothingLoss(classes=3, smoothing=0.1, weight=weights)


    # Uso
    base_loss = nn.CrossEntropyLoss(weight=weights.cuda(args.gpu))
    loss_func = DiversityLoss(base_loss, diversity_weight=0.15)


    # Uso finale
    class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(labels), y=labels[train_indices])
    focal_weights = torch.tensor(class_weights, dtype=torch.float).cuda(args.gpu)

    loss_func = CombinedLoss(
        alpha=focal_weights, 
        gamma=2.0, 
        diversity_weight=0.15
    )
    
    center_criterion = CenterLoss(num_classes=3, feat_dim=128)
    loss = combined_loss(logits, features, targets)
    print("Combined Loss:", loss.item())

    # E: (N,d) embeddings del modello su un batch
    E = F.normalize(E, p=2, dim=1)
    sim = E @ E.T  # cosine perché E è L2-normalizzato
    loss = similarity_margin_loss(sim, labels, pos_margin=0.5, neg_margin=0.0)

    # oppure
    loss_supcon = supervised_contrastive_from_similarity(sim, labels, temperature=0.07)
