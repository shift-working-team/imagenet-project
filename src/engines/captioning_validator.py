import torch


def validation_one_epoch(
    encoder,
    decoder,
    loader,
    criterion,
    device,
    epoch,
    epochs
):

    encoder.eval()
    decoder.eval()

    with torch.no_grad():
        total_loss = 0
        if (epoch+1) >= 5 and ((epoch+1) % 5 == 0 or (epoch+1) == epochs):
            all_references = []
            all_feature = []
        else:
            all_references = None
            all_feature = None

        for images, captions, batch_references in loader:
            images = images.to(device) # B, 3, 224, 224
            captions = captions.to(device) # B, seq_len

            feature = encoder(images, return_features=True) # B, 49, 512

            if (epoch+1) >= 5 and ((epoch+1) % 5 == 0 or (epoch+1) == epochs):
                all_references.extend(list(zip(*batch_references))) # 모든 배치의 정답 캡션
                all_feature.append(feature.cpu()) # 모든 배치의 이미지 특성 -> 나중에 generate로 모든 배치에 대해 문장 생성
            
            input_caption = captions[:, :-1] # B, seq_len-1
            target_caption = captions[:, 1:] # B, seq_len-1

            outputs = decoder(feature, input_caption) # B, seq_len-1, voca_size

            loss = criterion(
                outputs.reshape(-1, outputs.shape[-1]), # B*(seq_len-1), voca_size
                target_caption.reshape(-1) # B*seq_len-1
            )

            total_loss += loss.item()

    if (epoch+1) >= 5 and ((epoch+1) % 5 == 0 or (epoch+1) == epochs):
        return total_loss / len(loader), torch.cat(all_feature, dim=0), all_references, images[0]
    else:
        return total_loss / len(loader), all_feature, all_references, None