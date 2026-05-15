import torch.nn as nn
from torchvision import models


class EncoderResnet18(nn.Module):
    def __init__(self, num_classes=50, embed_size=512):
        super().__init__()
        model = models.resnet18(
            weights=models.ResNet18_Weights.DEFAULT
        )
        modules = list(model.children())[:-1]
        self.backbone = nn.Sequential(*modules)
        for param in self.backbone.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(
            model.fc.in_features,
            num_classes
        )

        self.projector = nn.Linear(
            model.fc.in_features,
            embed_size
        )

    def forward(
        self,
        images,
        return_features=False
    ):

        features = self.backbone(images)
        features = features.view(
            features.size(0),
            -1
        )

        logits = self.classifier(features)
        features = self.projector(features)

        # classification
        if not return_features:
            return logits

        # captioning
        return features