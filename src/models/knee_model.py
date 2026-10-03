import torch
import torch.nn as nn
import torchvision.models as models

class ResNet3DBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResNet3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv3d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class KneeAbnormalityModel3D(nn.Module):
    """
    Modèle 3D pour la détection d'anomalies du genou.
    Traite des volumes d'images (Depth, Height, Width).
    """
    def __init__(self, num_classes=12, input_channels=1):
        super(KneeAbnormalityModel3D, self).__init__()
        
        self.prep = nn.Sequential(
            nn.Conv3d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        )
        
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(ResNet3DBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(ResNet3DBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        # x shape: (batch, channels, depth, height, width)
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

class KneeAbnormalityModel(nn.Module):
    """
    Modèle de classification multi-label pour la détection d'anomalies du genou.
    Utilise un backbone pré-entraîné (EfficientNet-B0) adapté pour des images 
    en niveaux de gris (1 canal).
    """
    def __init__(self, num_classes=12):
        super(KneeAbnormalityModel, self).__init__()
        
        self.backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
        
        in_features = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)

def get_model(num_classes=12, model_type='2.5D'):
    if model_type == '3D':
        return KneeAbnormalityModel3D(num_classes=num_classes)
    return KneeAbnormalityModel(num_classes=num_classes)
