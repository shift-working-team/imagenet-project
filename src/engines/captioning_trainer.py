import torch


def train_one_epoch(
    encoder,
    decoder,
    loader,
    criterion,
    optimizer,
    device,
    scheduler=None
    ):

    encoder.train()
    decoder.train()

    total_loss = 0
    for images, captions in loader:
        images = images.to(device)
        captions = captions.to(device)

        feature = encoder(images, return_features=True)

        input_caption = captions[:, :-1]
        target_caption = captions[:, 1:]

        outputs = decoder(feature, input_caption)

        loss = criterion(
            outputs.reshape(-1, outputs.shape[-1]),
            target_caption.reshape(-1)
        )

        if scheduler is not None:
            scheduler.step()
            
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)