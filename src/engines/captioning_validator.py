import torch


def validation_one_epoch(
    encoder,
    decoder,
    loader,
    criterion,
    device,
):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_loss = 0
        for images, captions, _, __ in loader:

            images = images.to(device) # B, 3, 224, 224
            captions = captions.to(device) # B, seq_len

            feature = encoder(images, return_features=True) # B, 49, 512
            
            input_caption = captions[:, :-1] # B, seq_len-1
            target_caption = captions[:, 1:] # B, seq_len-1

            outputs = decoder(feature, input_caption) # B, seq_len-1, voca_size

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), # B*(seq_len-1), voca_size
                target_caption.reshape(-1) # B*seq_len-1
            )

            total_loss += loss.item()

    return total_loss / len(loader)