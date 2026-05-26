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

        cap_modules = list(model.children())[:-2]
        self.cap_backbone = nn.Sequential(*cap_modules)
        for param in self.cap_backbone.parameters():
            param.requires_grad = False

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

        cap_features = self.cap_backbone(images)
        cap_features = cap_features.flatten(2)
        cap_features = cap_features.permute(0, 2, 1)

        # classification
        if not return_features:
            return logits

        # captioning
        return cap_features