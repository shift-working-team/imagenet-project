import torch


def validation_one_epoch(
    encoder,
    decoder,
    loader,
    criterion,
    device,
    w2i
):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_loss = 0
        for images, captions, references_caption in loader:
            images = images.to(device)
            captions = captions.to(device)
            references_caption = list(zip(*references_caption))

            feature = encoder(images)

            input_caption = captions[:, :-1]
            target_caption = captions[:, 1:]

            outputs = decoder(feature, input_caption)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                target_caption.reshape(-1)
            )

            total_loss += loss.item()

    return total_loss / len(loader), feature[-1].unsqueeze(0), references_caption[-1]