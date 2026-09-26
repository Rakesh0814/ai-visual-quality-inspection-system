import torch
from torch import nn
from torchvision import models
class QualityClassifier(nn.Module):
    def __init__(self,pretrained=True):
        super().__init__()
        weights=models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        self.backbone=models.mobilenet_v3_small(weights=weights)
        n=self.backbone.classifier[-1].in_features
        self.backbone.classifier[-1]=nn.Linear(n,1)
    def forward(self,x): return self.backbone(x).squeeze(1)
