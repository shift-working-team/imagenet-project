import torch


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    total_loss = 0

    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)
        input_caption = captions[:, :-1]
        target_caption = captions[:, 1:]
        outputs = model(
            images,
            input_caption
        )

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            target_caption.reshape(-1)
        )

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)