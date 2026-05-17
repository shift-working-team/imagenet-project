import sys
sys.path.append("/workspace/src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
import wandb
import subprocess
import os
from datetime import datetime

from dataset.build_vocab import build_vocab, tokenizer
from dataset.captioning_dataset import CaptionDataset
from transforms.image_transform import get_caption_transform

from models.resnet18 import EncoderResnet18
from models.lstm import DecoderLSTM
from models.gru import DecoderGRU
from models.transformer import DecoderTransformer

from engines.resnet18_decoder_trainer import train_one_epoch
from engines.resnet18_decoder_validator import validation_one_epoch

from metrics.evaluate_caption import evaluate_caption


# params
with open("/workspace/params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


model_name = (
    f'{params["captioning"]["encoder"]}-'
    f'{params["captioning"]["decoder"]}'
)
version = params["captioning"]["version"]
date = datetime.now().strftime("%Y%m%d")


# device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# vocab
w2i, i2w, voca_size = build_vocab(
    params["data"]["captions_file"],
    min_freq=params["tokenizer"]["min_freq"],
    max_size=params["tokenizer"]["max_vocab_size"]
)


# transform
transform = get_caption_transform()


# train dataset
train_dataset = CaptionDataset(
    json_path=params["data"]["captions_file"],
    image_dir=params["data"]["raw_dir"],
    w2i=w2i,
    tokenizer=tokenizer,
    split="train",
    transform=transform,
    max_len=params["captioning"]["max_caption_length"]
)

# validation dataset
val_dataset = CaptionDataset(
    json_path=params["data"]["captions_file"],
    image_dir=params["data"]["raw_dir"],
    w2i=w2i,
    tokenizer=tokenizer,
    split="val",
    transform=transform,
    max_len=params["captioning"]["max_caption_length"]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=params["captioning"]["batch_size"],
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=params["captioning"]["batch_size"],
    shuffle=False
)


# model
encoder = EncoderResnet18().to(device)
if params["captioning"]["decoder"] == "transformer":
    decoder = DecoderTransformer(
        d_model=params["captioning"]["transformer"]["d_model"],
        nhead=params["captioning"]["transformer"]["nhead"],
        num_layers=params["captioning"]["transformer"]["num_layers"],
        voca_size=voca_size,
        max_len=params["captioning"]["max_caption_length"]
        ).to(device)
elif params["captioning"]["decoder"] == "lstm":
    decoder = DecoderLSTM(
        voca_size=voca_size,
        emd_size=params["captioning"]["lstm"]["embed_dim"],
        hidden_size=params["captioning"]["lstm"]["hidden_dim"],
        max_len=params["captioning"]["max_caption_length"]
        ).to(device)
elif params["captioning"]["decoder"] == "gru":
    decoder = DecoderGRU(
        voca_size=voca_size,
        emd_size=params["captioning"]["gru"]["embed_dim"],
        hidden_size=params["captioning"]["gru"]["hidden_dim"],
        max_len=params["captioning"]["max_caption_length"]
        ).to(device)


# optimizer
optimizer = torch.optim.Adam(
    list(encoder.projector.parameters()) +
    list(decoder.parameters()),
    lr=params["captioning"]["learning_rate"]
)


# loss
criterion = nn.CrossEntropyLoss(
    ignore_index=w2i["<pad>"]
)


# 1. 설정값 정의 (yaml 파일에서 읽어오는 것을 추천)
my_config = {
    "model_name": f'resnet18-{params["captioning"]["decoder"]}',
    "learning_rate": params["captioning"]["learning_rate"],
    "batch_size": params["captioning"]["batch_size"],
    "image_size": params["preprocess"]["image_size"],
    "seed": params["train"]["seed"],
    "epochs" : params["captioning"]["epochs"],
    "dataset_version": params["data"]["dataset_version"],
    "optimizer": params["captioning"]["optimizer"],
    "dvice": device.type,
}


# 2. W&B 초기화
wandb.init(
    project="imagenet-project",
    entity="super-shift-working", # 팀 계정이 있다면 작성
    config=my_config,
    name=f'{model_name}-{date}-{version}'
)


# checkpoint setting
save_dir = params["captioning"]["checkpoint"]["save_dir"]
os.makedirs(save_dir, exist_ok=True)

save_prefix = f"{model_name}_{date}_{version}"

best_val_loss = float("inf")

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


# train
for epoch in range(params["captioning"]["epochs"]):
    train_loss = train_one_epoch(
        encoder,
        decoder,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, all_feature, all_reference = validation_one_epoch(
        encoder,
        decoder,
        val_loader,
        criterion,
        device,
        epoch,
        params["captioning"]["epochs"]
    )

    print(f"Epoch {epoch+1} Train_Loss: {train_loss:.4f} Val_Loss: {val_loss:.4f}")

    log_dict = {
        "train/loss": train_loss,
        "val/loss": val_loss
    }

    if (epoch+1) >= 5 and ((epoch+1) % 5 == 0 or (epoch+1) == params["captioning"]["epochs"]):
        metric_result = evaluate_caption(
            decoder,
            all_feature,
            all_reference,
            w2i,
            i2w
        )

        log_dict.update(metric_result)

        sample_idx = 0

        print("-" * 60)
        print(f' Generated Sentence: {metric_result["generated"][sample_idx]}')
        print("-" * 60)

        for i, reference in enumerate(metric_result["references"][sample_idx], start=1):
            print(f'Reference {i}: {reference}')
        print("-" * 60)

        print(f'BLEU-1: {metric_result["bleu1"]:.4f}')
        print(f'BLEU-2: {metric_result["bleu2"]:.4f}')
        print(f'BLEU-3: {metric_result["bleu3"]:.4f}')
        print(f'BLEU-4: {metric_result["bleu4"]:.4f}')
        print(f'CIDEr: {metric_result["cider"]:.4f}')

    print('='*60)

    epoch_path = os.path.join(save_dir, f"{save_prefix}_epoch_{epoch+1}.pt")
    save_checkpoint(
        epoch_path,
        encoder,
        decoder,
        optimizer,
        epoch+1,
        train_loss,
        val_loss
    )

    latest_path = os.path.join(save_dir, f"{save_prefix}_epoch_latest.pt")
    save_checkpoint(
        latest_path,
        encoder,
        decoder,
        optimizer,
        epoch+1,
        train_loss,
        val_loss
    )

    if val_loss < best_val_loss:
        best_val_loss = val_loss

        best_path = os.path.join(save_dir, f"{save_prefix}_best.pt")
        save_checkpoint(
            best_path,
            encoder,
            decoder,
            optimizer,
            epoch+1,
            train_loss,
            val_loss
        )


    # 4. 지표 기록
    wandb.log(log_dict)

print(f"Best model val loss: {best_val_loss:.4f}")

wandb.finish()