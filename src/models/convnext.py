import torch.nn as nn
from torchvision import models


def get_convnext_tiny(
    num_classes=50
):

    model = models.convnext_tiny(pretrained=True)

    model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    return model