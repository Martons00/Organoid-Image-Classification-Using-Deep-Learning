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