import torch
import torch.nn as nn
import torchvision.models as models

class KneeAbnormalityModel(nn.Module):
    """
    Modèle 2.5D Avancé utilisant ConvNeXt-Tiny.
    ConvNeXt offre un meilleur compromis entre performance et stabilité que EfficientNet.
    """
    def __init__(self, num_classes=12):
        super(KneeAbnormalityModel, self).__init__()
        
        # Utilisation de ConvNeXt-Tiny pré-entraîné
        self.backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        
        # ConvNeXt a un classifieur final sous la forme d'une couche Linear dans self.backbone.classifier[2]
        in_features = self.backbone.classifier[2].in_features
        self.backbone.classifier = nn.Identity() 
        
        self.classifier = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
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

