import torch.nn as nn
from torchvision import models


class ResNet18Encoder(nn.Module):

    def __init__(self, embed_size=512):
        super().__init__()
        model = models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        self.linear = nn.Linear(
            model.fc.in_features,
            embed_size
        )

    def forward(self, images):
        features = self.backbone(images)
        features = features.flatten(1)
        features = self.linear(features)
        return features