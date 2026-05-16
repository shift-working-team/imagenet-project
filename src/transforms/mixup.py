import torch
import numpy as np


def mixup_data(
    images,
    labels,
    alpha=1.0
):

    if alpha > 0:
        lam = np.random.beta(alpha, alpha)

    else:
        lam = 1

    batch_size = images.size(0)

    index = torch.randperm(
        batch_size
    ).to(images.device)

    mixed_images = (
        lam * images
        + (1 - lam) * images[index]
    )

    labels_a = labels
    labels_b = labels[index]

    return (
        mixed_images,
        labels_a,
        labels_b,
        lam
    )