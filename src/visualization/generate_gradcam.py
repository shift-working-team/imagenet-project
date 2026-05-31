import os
import cv2
import torch
import torch.nn as nn
import numpy as np

from PIL import Image

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import (
    show_cam_on_image
)

from transforms.image_transform import (
    get_classification_valid_transform
)




class SwinClassifierWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, images):

        features = self.model.backbone(images)

        features = features.view(
            features.size(0),
            -1
        )

        logits = self.model.classifier(features)

        return logits




def reshape_transform(tensor):

    # Swin-T feature output: B, H, W, C
    # Grad-CAM expects: B, C, H, W
    if tensor.ndim == 4:

        tensor = tensor.permute(
            0,
            3,
            1,
            2
        )

    return tensor




def save_gradcam(
    model,
    image_path,
    save_path,
    device
):

    model.eval()

    for param in model.backbone.parameters():
        param.requires_grad = True

    for param in model.classifier.parameters():
        param.requires_grad = True

    gradcam_model = SwinClassifierWrapper(
        model
    ).to(device)

    gradcam_model.eval()

    transform = (
        get_classification_valid_transform()
    )

    image = Image.open(
        image_path
    ).convert("RGB")

    image = image.resize(
        (224, 224)
    )

    image_np = (
        np.array(image)
        .astype(np.float32)
        / 255.0
    )

    tensor = transform(
        image
    ).unsqueeze(0).to(device)

    target_layer = (
        model.backbone.features[-1][-1].norm2
    )

    cam = GradCAM(
        model=gradcam_model,
        target_layers=[target_layer],
        reshape_transform=reshape_transform
    )

    grayscale_cam = cam(
        input_tensor=tensor
    )[0]

    visualization = show_cam_on_image(
        image_np,
        grayscale_cam,
        use_rgb=True
    )

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    cv2.imwrite(
        save_path,
        cv2.cvtColor(
            visualization,
            cv2.COLOR_RGB2BGR
        )
    )