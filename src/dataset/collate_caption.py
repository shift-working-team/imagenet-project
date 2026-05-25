import torch

def collate_caption(batch):
    images = []
    tokens = []

    for image, token in batch:
        images.append(image)
        tokens.append(token)

    images = torch.cat(images, dim=0)
    tokens = torch.cat(tokens, dim=0)

    return images, tokens