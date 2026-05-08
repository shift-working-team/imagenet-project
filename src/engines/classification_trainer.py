import torch

def train_one_epoch(model, loader, criterion, optimizer, device):

    model.train()

    total_loss = 0

    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        loss = criterion(outputs, labels)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)