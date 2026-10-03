import torch
import torch.nn as nn
import torchvision.models as models

class KneeAbnormalityModel(nn.Module):
    """
    Modèle 2.5D Avancé : Utilise EfficientNet-B0 pour extraire des caractéristiques 
    de plusieurs vues et les agrège via un Max-Pooling.
    """
    def __init__(self, num_classes=12):
        super(KneeAbnormalityModel, self).__init__()
        
        # Backbone EfficientNet-B0 pré-entraîné
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        # On récupère les features juste avant le classifieur final
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier = nn.Identity() # On retire le classifieur d'origine
        
        # Nouveau classifieur pour l'agrégation
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Shape (batch, num_views * 3, H, W) ou (batch, 3, H, W)
        Returns:
            torch.Tensor: Logits (batch, num_classes)
        """
        if x.shape[1] > 3:
            batch_size = x.shape[0]
            num_views = x.shape[1] // 3
            x = x.view(-1, 3, x.shape[2], x.shape[3])
            features = self.backbone(x)
            features = features.view(batch_size, num_views, -1)
            features, _ = torch.max(features, dim=1)
        else:
            features = self.backbone(x)
            
        return self.classifier(features)

def get_model(num_classes=12, model_type='2.5D'):
    return KneeAbnormalityModel(num_classes=num_classes)
