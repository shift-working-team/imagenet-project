from torchmetrics.classification import (
    MulticlassAccuracy
)

from transforms.mixup import mixup_data
from transforms.cutmix import cutmix_data


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    num_classes,
    augmentation=None
):

    model.train()

    metric = MulticlassAccuracy(
        num_classes=num_classes
    ).to(device)

    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        if augmentation == "mixup":
            images, labels_a, labels_b, lam = mixup_data(
                images,
                labels
            )

        elif augmentation == "cutmix":
            images, labels_a, labels_b, lam = cutmix_data(
                images,
                labels
            )

        outputs = model(images)

        if augmentation in ["mixup", "cutmix"]:
            loss = (
                lam * criterion(outputs, labels_a)
                + (1 - lam) * criterion(outputs, labels_b)
            )

        else:
            loss = criterion(
                outputs,
                labels
            )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)

        metric.update(
            preds,
            labels
        )

    acc = metric.compute().item()

    return total_loss / len(loader), acc