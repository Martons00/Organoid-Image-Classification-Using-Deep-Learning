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

def plot_similarity_heatmap(similarity_matrix, labels=None, save_path=None):
    """Plotta heatmap della matrice di similarità"""
    plt.figure(figsize=(12, 10))
    
    # Crea annotazioni con etichette se disponibili
    if labels is not None:
        class_names = [f"Class {l}" for l in labels]
        annot = False  # Troppo denso con tutte le annotazioni
    else:
        annot = False
        class_names = None
    
    # Plot heatmap
    sns.heatmap(
        similarity_matrix, 
        cmap='coolwarm', 
        center=0,
        square=True,
        annot=annot,
        fmt='.2f',
        cbar_kws={"shrink": .8},
        xticklabels=class_names if class_names else False,
        yticklabels=class_names if class_names else False
    )
    
    plt.title('Feature Similarity Matrix (Cosine Similarity)')
    plt.xlabel('Samples')
    plt.ylabel('Samples')
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

# Uso
#similarity_matrix = compute_similarity_matrix(features, metric='cosine')
#plot_similarity_heatmap(similarity_matrix, labels)
