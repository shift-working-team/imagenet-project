import os
import torch

def save_checkpoint(
    path,
    encoder,
    decoder,
    optimizer,
    epoch,
    train_loss,
    val_loss
):
    torch.save({
        "epoch": epoch,
        "encoder_state_dict": encoder.state_dict(),
        "decoder_state_dict": decoder.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "train_loss": train_loss,
        "val_loss": val_loss
    }, path)


def load_checkpoint(
        best_path,
        encoder,
        decoder,
        optimizer,
        device
):

    print(f"Loading checkpoint: {best_path}")

    checkpoint = torch.load(
        best_path,
        map_location=device
    )

    encoder.load_state_dict(checkpoint["encoder_state_dict"])

    decoder.load_state_dict(checkpoint["decoder_state_dict"])

    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    start_epoch = checkpoint["epoch"]

    best_val_loss = checkpoint["val_loss"]

    print(
        f"Resume from Epoch {start_epoch} | "
        f"Best Val Loss: {best_val_loss:.4f}"
    )

    return start_epoch, best_val_loss