import torch.nn as nn
from torchvision import models


class EncoderEfficientNetB0(nn.Module):
    def __init__(self, num_classes=50, embed_size=512):
        super().__init__()

        model = models.efficientnet_b0(
            weights=models.EfficientNet_B0_Weights.DEFAULT
        )

        self.backbone = model.features
        self.pool = nn.AdaptiveAvgPool2d(1)

        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = model.classifier[1].in_features

        self.classifier = nn.Linear(
            in_features,
            num_classes
        )

        self.projector = nn.Linear(
            in_features,
            embed_size
        )

    def forward(
        self,
        images,
        return_features=False
    ):

        features = self.backbone(images)
        features = self.pool(features)

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