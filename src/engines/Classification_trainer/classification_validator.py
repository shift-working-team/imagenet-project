import torch

from torchmetrics.classification import (
    MulticlassAccuracy,
    MulticlassF1Score
)


def validation_one_epoch(
    model,
    loader,
    criterion,
    device,
    num_classes
):

    model.eval()

    acc_metric = MulticlassAccuracy(
        num_classes=num_classes
    ).to(device)

    f1_metric = MulticlassF1Score(
        num_classes=num_classes,
        average="macro"
    ).to(device)

    # precision_metric = MulticlassPrecision(
    #     num_classes=num_classes,
    #     average="macro"
    # ).to(device)

    # recall_metric = MulticlassRecall(
    #     num_classes=num_classes,
    #     average="macro"
    # ).to(device)

    total_loss = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()
            preds = outputs.argmax(dim=1)
            acc_metric.update(
                preds,
                labels
            )

            f1_metric.update(
                preds,
                labels
            )

            # precision_metric.update(
            #     preds,
            #     labels
            # )

            # recall_metric.update(
            #     preds,
            #     labels
            # )

    acc = acc_metric.compute().item()
    f1 = f1_metric.compute().item()
    # precision = precision_metric.compute().item()
    # recall = recall_metric.compute().item()

    return (
        total_loss / len(loader),
        acc,
        f1
    )