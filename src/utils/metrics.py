import numpy as np
from sklearn.metrics import roc_auc_score

def calculate_auc(labels, preds):
    """
    Calcule l'AUC-ROC moyenne et par classe pour un problème multi-label.
    
    Args:
        labels (np.ndarray): Ground truth labels de shape (n_samples, n_classes).
        preds (np.ndarray): Prédictions (probabilités) de shape (n_samples, n_classes).
        
    Returns:
        tuple: (mean_auc, class_aucs)
    """
    aucs = []
    for i in range(labels.shape[1]):
        try:
            auc = roc_auc_score(labels[:, i], preds[:, i])
            aucs.append(auc)
        except ValueError:
            aucs.append(0.5) # Fallback si une seule classe est présente dans le batch
    return np.mean(aucs), aucs
