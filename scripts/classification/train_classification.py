import sys
sys.path.append("/workspace/src")
import os
import yaml
import wandb
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.classification_dataset import (ClassificationDataset)
from transforms.image_transform import (
    get_classification_train_transform,
    get_classification_valid_transform
)
from models.resnet18 import get_resnet18
from engines.Classification_trainer.classification_trainer import train_one_epoch
from engines.Classification_trainer.classification_validator import validation_one_epoch
from models.efficientnet import get_efficientnet_b0
from models.convnext import get_convnext_tiny
from models.mobilenet import get_mobilenet_v3_small

# params
with open(
    "/workspace/params.yaml",
    "r",
    encoding="utf-8"
) as f:

    params = yaml.safe_load(f)



# device
device = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)



# class mapping
classes = sorted(

    [
        cls for cls in os.listdir(
            params["data"]["raw_dir"]
        )

        if os.path.isdir(
            os.path.join(
                params["data"]["raw_dir"],
                cls
            )
        )
    ]
)

class_to_idx = {

    cls: idx
    for idx, cls in enumerate(classes)
}

num_classes = len(classes)



# transform
train_transform = (
    get_classification_train_transform()
)

valid_transform = (
    get_classification_valid_transform()
)




# dataset
train_dataset = ClassificationDataset(
    root_dir=params["data"]["raw_dir"],
    class_to_idx=class_to_idx,
    split="train",
    transform=train_transform
)

val_dataset = ClassificationDataset(
    root_dir=params["data"]["raw_dir"],
    class_to_idx=class_to_idx,
    split="val",
    transform=valid_transform
)



# loader
train_loader = DataLoader(
    train_dataset,
    batch_size=params["train"]["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=params["train"]["batch_size"],
    shuffle=False
)



# model
model_name = params["model"]["name"]


if model_name == "resnet18":
    model = get_resnet18(
        num_classes=num_classes
    ).to(device)

elif model_name == "efficientnet_b0":
    model = get_efficientnet_b0(
        num_classes=num_classes
    ).to(device)

elif model_name == "convnext_tiny":
    model = get_convnext_tiny(
        num_classes=num_classes
    ).to(device)

elif model_name == "mobilenet_v3_small":
    model = get_mobilenet_v3_small(
        num_classes=num_classes
    ).to(device)

else:
    raise ValueError(
        f"Unsupported model: {model_name}"
    )



# loss
criterion = nn.CrossEntropyLoss()



# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=params["train"]["lr"]
)



# wandb
wandb.init(
    project="imagenet-project",
    entity="super-shift-working",
    config=params,
    name=f"{model_name}-baseline"
)



# train
for epoch in range(
    params["train"]["epochs"]
):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_classes
    )

    (
        val_loss,
        val_acc,
        val_f1,
        val_precision,
        val_recall

    ) = validation_one_epoch(
        model,
        val_loader,
        criterion,
        device,
        num_classes
    )

    wandb.log({
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "val/f1": val_f1,
        "val/precision": val_precision,
        "val/recall": val_recall
    })

    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val F1: {val_f1:.4f}"
    )

wandb.finish()