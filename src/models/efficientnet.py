import torch.nn as nn
from torchvision import models


def get_efficientnet_b0(
    num_classes=50
):

    model = models.efficientnet_b0(pretrained=True)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    return model