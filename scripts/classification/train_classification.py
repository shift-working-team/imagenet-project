import sys
sys.path.append("/workspace/src")

import os
import yaml
import wandb
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from dataset.classification_dataset import ClassificationDataset
from transforms.image_transform import (
    get_classification_train_transform,
    get_classification_valid_transform,
    get_classification_aug_transform
)
from models.resnet18 import EncoderResnet18
from models.efficientnet import get_efficientnet_b0
from models.convnext import get_convnext_tiny
from models.mobilenet import get_mobilenet_v3_small
from engines.Classification_trainer.classification_trainer import train_one_epoch
from engines.Classification_trainer.classification_validator import validation_one_epoch




# params
with open(
    "/workspace/params.yaml",
    "r",
    encoding="utf-8"
) as f:

    params = yaml.safe_load(f)




# seed
SEED = params["train"]["seed"]
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False




# device
device = torch.device(
    params["train"]["device"]
    if torch.cuda.is_available()
    else "cpu"
)

print(f"device: {device}")




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

print(f"num_classes: {num_classes}")




# augmentation
augmentation_type = (

    params["classification"]
    ["augmentation"]["type"]
)

if augmentation_type == "none":
    augmentation_type = None




# transform
if params["classification"]["augmentation"]["use_aug"]:

    train_transform = (
        get_classification_aug_transform()
    )

else:

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




# dataloader
train_loader = DataLoader(
    train_dataset,
    batch_size=params["train"]["batch_size"],
    shuffle=True,
    num_workers=params["train"]["num_workers"],
    pin_memory=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=params["train"]["batch_size"],
    shuffle=False,
    num_workers=params["train"]["num_workers"],
    pin_memory=True
)




# model
model_name = (
    params["classification"]["model_name"]
)


if model_name == "resnet18":
    model = EncoderResnet18(
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
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=params["classification"]["learning_rate"]
)




# checkpoint
checkpoint_dir = (
    params["classification"]
    ["checkpoint"]["save_dir"]
)

os.makedirs(
    checkpoint_dir,
    exist_ok=True
)




# wandb name
wandb_name = (
    f"{model_name}-baseline"
    if augmentation_type is None
    else f"{model_name}-{augmentation_type}"
)




# 1. 설정값 정의
my_config = {
    "model_name": model_name,
    "learning_rate": params["classification"]["learning_rate"],
    "batch_size": params["train"]["batch_size"],
    "image_size": params["preprocess"]["image_size"],
    "seed": params["train"]["seed"],
    "epochs": params["classification"]["epochs"],
    "augmentation": augmentation_type,
    "optimizer": params["train"]["optimizer"],
    "device": device.type,
    "num_classes":num_classes
}




# 2. W&B 초기화
wandb.init(
    project=params["logging"]["project_name"],
    entity="super-shift-working",
    config=my_config,
    name=wandb_name
)




# train
best_f1 = 0.0

for epoch in range(
    params["classification"]["epochs"]
):

    train_loss, train_acc = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device,
        num_classes,
        augmentation=augmentation_type
    )

    (
        val_loss,
        val_acc,
        val_f1,
        # val_precision,
        # val_recall

    ) = validation_one_epoch(
        model,
        val_loader,
        criterion,
        device,
        num_classes
    )




    # checkpoint save
    if val_f1 > best_f1:
        best_f1 = val_f1

        save_path = os.path.join(
            checkpoint_dir,
            f"{model_name}_best.pth"
        )

        torch.save(
            model.state_dict(),
            save_path
        )

        print(
            f"Best model saved "
            f"(F1: {best_f1:.4f})"
        )




    # wandb log
    wandb.log({
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "val/macro_f1": val_f1,
        # "val/precision": val_precision,
        # "val/recall": val_recall
    })




    # print log
    print(
        f"Epoch {epoch+1} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f} | "
        f"Val Macro F1: {val_f1:.4f}"
    )




# finish
wandb.finish()