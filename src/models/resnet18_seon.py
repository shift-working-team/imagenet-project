import torch.nn as nn
from torchvision import models


class EncoderResnet18(nn.Module):
    def __init__(self, num_class=50, embed_size=512):
        super().__init__()
        model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        modules = list(model.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        for param in self.backbone.parameters():
            param.requires_grad = False

        self.fc = nn.Linear(model.fc.in_features, num_class)

        self.projection = nn.Sequential(
            nn.Linear(model.fc.in_features, embed_size),
            nn.BatchNorm1d(embed_size)
        )

    def forward(self, images):
        features = self.backbone(images)

        logits = self.fc(features)

        features = features.view(features.size(0), -1)
        features = self.projection(features)

        return logits, features