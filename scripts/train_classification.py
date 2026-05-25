import argparse
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
from models.efficientnet import EncoderEfficientNetB0
from models.convnext import EncoderConvNextTiny
from models.mobilenet import EncoderMobileNetV3Small
from models.vit import EncoderViTB16
from models.swin import EncoderSwinTiny
from models.deit import EncoderDeiTTiny
from engines.classification_trainer import train_one_epoch
from engines.classification_validator import validation_one_epoch




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


# stage cmd
parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, required=True)
parser.add_argument("--augmentation", type=str, default="none")
args = parser.parse_args()



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



# model
model_name = args.model


# learning rate
transformer_models = [
    "vit_b_16",
    "swin_t",
    "deit_tiny_patch16_224"
]

if model_name in transformer_models:
    learning_rate = (
        params["classification"]
        ["learning_rate"]["transformer"]
    )

else:
    learning_rate = (
        params["classification"]
        ["learning_rate"]["cnn"]
    )




# augmentation
augmentation_type = args.augmentation
if augmentation_type == "none":
    augmentation_type = None




# transform
if augmentation_type is not None:
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
if model_name == "resnet18":
    model = EncoderResnet18(
        num_classes=num_classes
    ).to(device)

elif model_name == "efficientnet_b0":
    model = EncoderEfficientNetB0(
        num_classes=num_classes
    ).to(device)

elif model_name == "convnext_tiny":
    model = EncoderConvNextTiny(
        num_classes=num_classes
    ).to(device)

elif model_name == "mobilenet_v3_small":
    model = EncoderMobileNetV3Small(
        num_classes=num_classes
    ).to(device)

elif model_name == "vit_b_16":
    model = EncoderViTB16(
        num_classes=num_classes
    ).to(device)

elif model_name == "swin_t":
    model = EncoderSwinTiny(
        num_classes=num_classes
    ).to(device)

elif model_name == "deit_tiny_patch16_224":
    model = EncoderDeiTTiny(
        num_classes=num_classes
    ).to(device)

else:
    raise ValueError(
        f"Unsupported model: {model_name}"
    )




# loss
criterion = nn.CrossEntropyLoss()




# optimizer
optimizer_name = (
    params["classification"]["optimizer"]
)

if optimizer_name == "adam":
    optimizer = torch.optim.Adam(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=learning_rate
    )

elif optimizer_name == "sgd":
    optimizer = torch.optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=learning_rate,
        momentum=0.9
    )

elif optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(
        filter(
            lambda p: p.requires_grad,
            model.parameters()
        ),
        lr=learning_rate
    )

else:
    raise ValueError(
        f"Unsupported optimizer: {optimizer_name}"
    )




# scheduler
scheduler = None
if params["classification"]["scheduler"]["use"]:
    scheduler_name = (
        params["classification"]
        ["scheduler"]["name"]
    )

    if scheduler_name == "cosineannealinglr":
        scheduler = (
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=params["classification"]["epochs"]
            )
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
model_tag = model_name.replace("_","-")

augmentation_tag = (
    "base"
    if augmentation_type is None
    else augmentation_type
)

dataset_version_tag = (
    params["data"]["dataset_version"]
)

wandb_name_parts = [
    "cls",
    model_tag,
    augmentation_tag,
    dataset_version_tag
]




# step 4 tuning
optimizer_name = (params["classification"]["optimizer"])
batch_size = (params["train"]["batch_size"])




# learning rate
if model_name in transformer_models:
    baseline_lr = (
        params["classification"]
        ["learning_rate"]["transformer"]
    )

else:
    baseline_lr = (
        params["classification"]
        ["learning_rate"]["cnn"]
    )

if learning_rate != baseline_lr:
    lr_tag = str(learning_rate).replace(
        "0.",
        ""
    )

    wandb_name_parts.append(
        f"lr-{lr_tag}"
    )




# batch size
if batch_size != 32:
    wandb_name_parts.append(
        f"bs-{batch_size}"
    )




# optimizer
if optimizer_name != "adam":
    wandb_name_parts.append(
        optimizer_name
    )




# scheduler
if (
    params["classification"]
    ["scheduler"]["use"]
):

    scheduler_name = (
        params["classification"]
        ["scheduler"]["name"]
    )

    if scheduler_name == "cosineannealinglr":
        wandb_name_parts.append(
            "cosine"
        )


wandb_name = "_".join(
    wandb_name_parts
)




# config
my_config = {
    "model_name": model_name,
    "learning_rate": learning_rate,
    "batch_size": params["train"]["batch_size"],
    "image_size": params["preprocess"]["image_size"],
    "seed": params["train"]["seed"],
    "epochs": params["classification"]["epochs"],
    "augmentation": augmentation_type,
    "optimizer": optimizer_name,
    "scheduler": (
        params["classification"]["scheduler"]["name"]
        if params["classification"]["scheduler"]["use"]
        else None
    ),
    "device": device.type,
    "num_classes": num_classes,
    "dataset_version": (
        params["data"]["dataset_version"]
    )
}




# wandb
wandb.init(
    project=params["logging"]["project_name"],
    entity="super-shift-working",
    config=my_config,
    name=wandb_name,
    tags=["classification"]
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




    # scheduler step
    if scheduler is not None:
        scheduler.step()




    # checkpoint save
    if val_f1 > best_f1:
        best_f1 = val_f1

        dataset_version = (
            params["data"]["dataset_version"]
        )

        save_path = os.path.join(
            checkpoint_dir,
            f"{model_name}_{dataset_version}_best.pth"
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
    log_dict = {
        "train/loss": train_loss,
        "train/accuracy": train_acc,
        "val/loss": val_loss,
        "val/accuracy": val_acc,
        "val/macro_f1": val_f1,
        # "val/precision": val_precision,
        # "val/recall": val_recall
    }

    if scheduler is not None:
        log_dict["learning_rate"] = (
            optimizer.param_groups[0]["lr"]
        )
    wandb.log(log_dict)




    # print log
    print(
        f"Epoch {epoch+1}"
        f"Train Loss: {train_loss:.4f}"
        f"Train Acc: {train_acc:.4f}"
        f"Val Loss: {val_loss:.4f}"
        f"Val Acc: {val_acc:.4f}"
        f"Val Macro F1: {val_f1:.4f}"
    )




# finish
wandb.finish()