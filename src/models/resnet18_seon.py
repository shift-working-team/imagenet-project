import torch.nn as nn
from torchvision import models


class EncoderResnet18(nn.Module):
    def __init__(self, num_class=50):
        super().__init__()
        model = models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]

        self.backbone = nn.Sequential(*modules)
        self.fc = nn.Linear(model.fc.in_features, num_class)

    def forward(self, images):
        features = self.backbone(images)
        features = features.flatten(1)

        tgt = self.fc(features)
        
        return tgt, features