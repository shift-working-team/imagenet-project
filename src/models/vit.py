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

        # 특성 추출
        cap_features = self.backbone._process_input(images) # B, 196, 768
        cap_features = cap_features + self.backbone.encoder.pos_embedding[:, 1:, :] # 위치 임베딩
        for layer in self.backbone.encoder.layers: # B, 196, 768
            cap_features = layer(cap_features)
        cap_features = self.backbone.encoder.ln(cap_features) # LayerNorm
        cap_features = self.projector(cap_features) # B, 196, d_model

        # classification
        if not return_features:
            return logits

        # captioning
        return cap_features