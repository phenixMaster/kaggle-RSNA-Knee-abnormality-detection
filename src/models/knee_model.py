import torch
import torch.nn as nn
import torchvision.models as models

class KneeAbnormalityModel(nn.Module):
    """
    Modèle de classification multi-label pour la détection d'anomalies du genou.
    Utilise un backbone pré-entraîné (EfficientNet-B0) adapté pour des images 
    en niveaux de gris (1 canal).
    """
    def __init__(self, num_classes=12):
        super(KneeAbnormalityModel, self).__init__()
        
        # Utilisation d'EfficientNet-B0 comme backbone
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Pour l'approche 2.5D, on utilise les 3 canaux natifs d'EfficientNet
        # On ne modifie plus la première couche Conv2d
        
        # Remplacement du classifieur final
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Forward pass du modèle.
        Args:
            x (torch.Tensor): Image d'entrée de shape (batch, 3, 384, 384).
        Returns:
            torch.Tensor: Logits de sortie de shape (batch, num_classes).
        """
        return self.backbone(x)

def get_model(num_classes=12):
    """
    Fonction utilitaire pour instancier le modèle.
    """
    return KneeAbnormalityModel(num_classes=num_classes)
