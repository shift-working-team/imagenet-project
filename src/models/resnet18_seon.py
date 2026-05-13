import torch.nn as nn
from torchvision import models


class EncoderResnet18(nn.Module):

    def __init__(self, num_classes=50, embed_size=512):
        super().__init__()
        model = models.resnet18(pretrained=True)
        modules = list(model.children())[:-1]
        self.backbone = nn.Sequential(*modules)

        self.classifier = nn.Linear(model.fc.in_features, num_classes)

        self.projector = nn.Linear(model.fc.in_features, embed_size)

    def forward(self, images):
        features = self.backbone(images)
        features = features.flatten(1)
        logits = self.classifier(features)
        embeddings = self.projector(features)
        return logits, embeddings