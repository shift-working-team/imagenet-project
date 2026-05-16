import torch


def validation_one_epoch(
    encoder,
    decoder,
    loader,
    criterion,
    device,
    epoch
):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_loss = 0
        if epoch >= 5 and epoch % 5 == 0:
            all_references = []
            all_feature = []
        else:
            all_references = None
            all_feature = None

        for images, captions, batch_references in loader:
            images = images.to(device)
            captions = captions.to(device)

            feature = encoder(images, return_features=True)

            if epoch >= 5 and epoch % 5 == 0:
                all_references.extend(list(zip(*batch_references)))
                all_feature.append(feature.cpu())
            
            input_caption = captions[:, :-1]
            target_caption = captions[:, 1:]

            outputs = decoder(feature, input_caption)

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]),
                target_caption.reshape(-1)
            )

            total_loss += loss.item()

    if epoch >= 5 and epoch % 5 == 0:
        return total_loss / len(loader), torch.cat(all_feature, dim=0), all_references
    else:
        return total_loss / len(loader), all_feature, all_references