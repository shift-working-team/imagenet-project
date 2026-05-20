import torch
import torch.nn as nn
from torchvision import models


class EncoderViTB16(nn.Module):
    def __init__(self, num_classes=50, embed_size=512):
        super().__init__()

        model = models.vit_b_16(
            weights=models.ViT_B_16_Weights.DEFAULT
        )

        self.backbone = model

        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = model.heads.head.in_features

        self.backbone.heads = nn.Identity()

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

        if isinstance(features, tuple):
            features = features[0]

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