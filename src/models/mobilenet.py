import torch.nn as nn
from torchvision import models


def get_mobilenet_v3_small(
    num_classes=50
):

    model = models.mobilenet_v3_small(pretrained=True)
    model.classifier[3] = nn.Linear(model.classifier[3].in_features, num_classes)

    return model