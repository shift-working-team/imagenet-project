import torch
import numpy as np


def rand_bbox(
    size,
    lam
):

    W = size[2]
    H = size[3]

    cut_rat = np.sqrt(1. - lam)

    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)

    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)

    return x1, y1, x2, y2


def cutmix_data(
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

    labels_a = labels
    labels_b = labels[index]

    x1, y1, x2, y2 = rand_bbox(
        images.size(),
        lam
    )

    images[:, :, x1:x2, y1:y2] = (
        images[index, :, x1:x2, y1:y2]
    )

    lam = 1 - (
        (x2 - x1) * (y2 - y1)
        / (images.size(-1) * images.size(-2))
    )

    return (
        images,
        labels_a,
        labels_b,
        lam
    )