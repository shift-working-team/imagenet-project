import torch.nn as nn
from torchvision import models


class EncoderSwinTiny(nn.Module):
    def __init__(self, num_classes=50, embed_size=512):
        super().__init__()

        model = models.swin_t(
            weights=models.Swin_T_Weights.DEFAULT
        )

        self.backbone = model

        for param in self.backbone.parameters():
            param.requires_grad = False

        in_features = model.head.in_features

        self.backbone.head = nn.Identity()

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