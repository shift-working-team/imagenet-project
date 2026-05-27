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


        self.cap_backbone = model.features # B, 7*7, 768
        for param in self.cap_backbone.parameters():
            param.requires_grad = False
        self.projector = nn.Linear(
            in_features, # 768
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

        # 특성 추출
        cap_features = self.cap_backbone(images) # B, 7*7, 768
        cap_features = cap_features.flatten(1, 2) # B, 49, 768
        cap_features = self.projector(cap_features) # B, 49, embedding

        # classification
        if not return_features:
            return logits

        # captioning
        return cap_features