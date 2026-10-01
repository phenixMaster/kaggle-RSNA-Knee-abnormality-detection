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
        # weights=models.EfficientNet_B0_Weights.DEFAULT charge les poids pré-entraînés sur ImageNet
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # Adaptation de la première couche pour accepter 1 canal (gris) au lieu de 3 (RGB)
        # On moyenne les poids des 3 canaux originaux pour conserver l'information
        original_weights = self.backbone.features[0][0].weight.data
        summed_weights = torch.sum(original_weights, dim=1, keepdim=True)
        self.backbone.features[0][0] = nn.Conv2d(
            1, 
            self.backbone.features[0][0].out_channels, 
            kernel_size=self.backbone.features[0][0].kernel_size, 
            stride=self.backbone.features[0][0].stride, 
            padding=self.backbone.features[0][0].padding, 
            bias=self.backbone.features[0][0].bias
        )
        self.backbone.features[0][0].weight.data = summed_weights
        
        # Remplacement du classifieur final
        # EfficientNet-B0 a un classifier avec 'avgpool' suivi d'une couche 'classifier'
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
            x (torch.Tensor): Image d'entrée de shape (batch, 1, 224, 224).
        Returns:
            torch.Tensor: Logits de sortie de shape (batch, num_classes).
        """
        return self.backbone(x)

def get_model(num_classes=12):
    """
    Fonction utilitaire pour instancier le modèle.
    """
    return KneeAbnormalityModel(num_classes=num_classes)
