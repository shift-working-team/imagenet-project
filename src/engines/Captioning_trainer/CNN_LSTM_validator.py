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
        for images, captions in loader:
            images = images.to(device)
            captions = captions.to(device)

            _, feature = encoder(images)

            input_caption = captions[:, :-1]
            target_caption = captions[:, 1:]

            outputs = decoder(feature, input_caption)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                target_caption.reshape(-1)
            )

            total_loss += loss.item()

        generated_inx = decoder.generate(
            feature[-1].unsqueeze(0),
            torch.tensor([w2i["<sos>"]]),
            torch.tensor([w2i["<eos>"]])
            )

    return total_loss / len(loader), generated_inx