from torchmetrics.classification import (
    MulticlassAccuracy
)


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    num_classes
):

    model.train()

    metric = MulticlassAccuracy(
        num_classes=num_classes
    ).to(device)

    total_loss = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        preds = outputs.argmax(dim=1)
        metric.update(preds, labels)

    acc = metric.compute().item()

    return total_loss / len(loader), acc