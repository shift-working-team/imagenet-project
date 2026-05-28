import argparse
import sys
sys.path.append("/workspace/src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import random
import yaml
import wandb
import os
from datetime import datetime

from dataset.build_vocab import build_vocab, tokenizer
from dataset.captioning_dataset import CaptionDataset
from dataset.collate_caption import collate_caption

from transforms.image_transform import get_caption_transform

from models.resnet18 import EncoderResnet18
from models.swin import EncoderSwinTiny
from models.vit import EncoderViTB16
from models.lstm import DecoderLSTM
from models.gru import DecoderGRU
from models.transformer import DecoderTransformer

from engines.captioning_trainer import train_one_epoch
from engines.captioning_validator import validation_one_epoch

from metrics.evaluate_caption import evaluate_caption
from metrics.make_show_all_caption import make_show_all_caption

from utils.checkpoint_manager import save_checkpoint, load_checkpoint


# params
with open("/workspace/params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


# seed
SEED = params["train"]["seed"]
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

encoder_name = params["captioning"]["encoder"]
decoder_name = params["captioning"]["decoder"]

model_name = (f'{encoder_name}-{decoder_name}')
version = params["captioning"]["version"]
date = datetime.now().strftime("%Y%m%d")


# device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# vocab
w2i, i2w, voca_size = build_vocab(
    params["captioning"]["data"]["train_caption"],
    min_freq=params["captioning"]["tokenizer"]["min_freq"],
    max_size=params["captioning"]["tokenizer"]["max_vocab_size"],
    use_subword=params["captioning"]["tokenizer"]["use_subword"],
    sp_model_path=params["captioning"]["tokenizer"]["sp_model_path"]
)


# transform
transform = get_caption_transform()


# train dataset
train_dataset = CaptionDataset(
    json_path=params["captioning"]["data"]["train_caption"],
    image_dir=params["captioning"]["data"]["train_img"],
    w2i=w2i,
    tokenizer=tokenizer,
    split="train",
    transform=transform,
    max_len=params["captioning"]["max_caption_length"],
    train_num_caption=params["captioning"]["train_num_caption"],
    debug=params["captioning"]["debug"],
    use_subword=params["captioning"]["tokenizer"]["use_subword"],
    sp_model_path=params["captioning"]["tokenizer"]["sp_model_path"]
)

# validation dataset
val_dataset = CaptionDataset(
    json_path=params["captioning"]["data"]["val_caption"],
    image_dir=params["captioning"]["data"]["val_img"],
    w2i=w2i,
    tokenizer=tokenizer,
    split="val",
    transform=transform,
    max_len=params["captioning"]["max_caption_length"],
    debug=params["captioning"]["debug"],
    use_subword=params["captioning"]["tokenizer"]["use_subword"],
    sp_model_path=params["captioning"]["tokenizer"]["sp_model_path"]
)


train_loader = DataLoader(
    train_dataset,
    batch_size=params["captioning"]["batch_size"],
    shuffle=True,
    collate_fn=collate_caption
)

val_loader = DataLoader(
    val_dataset,
    batch_size=params["captioning"]["batch_size"],
    shuffle=False
)


# model
if encoder_name == "resnet18":
    encoder = EncoderResnet18(embed_size=params["captioning"]["transformer"]["d_model"]).to(device)
elif encoder_name == "swin":
    encoder = EncoderSwinTiny(embed_size=params["captioning"]["transformer"]["d_model"]).to(device)
elif encoder_name == "vit":
    encoder = EncoderViTB16(embed_size=params["captioning"]["transformer"]["d_model"]).to(device)

if decoder_name == "transformer":
    decoder = DecoderTransformer(
        n_layers=params["captioning"]["transformer"]["n_layers"],
        nhead=params["captioning"]["transformer"]["nhead"],
        d_model=params["captioning"]["transformer"]["d_model"],
        d_ff=params["captioning"]["transformer"]["d_model"]*4,
        voca_size=voca_size,
        max_len=params["captioning"]["max_caption_length"],
        drop_p=params["captioning"]["transformer"]["drop_p"]
    ).to(device)
elif decoder_name == "lstm":
    decoder = DecoderLSTM(
        voca_size=voca_size,
        emd_size=params["captioning"]["lstm"]["embed_dim"],
        hidden_size=params["captioning"]["lstm"]["hidden_dim"],
        max_len=params["captioning"]["max_caption_length"]
        ).to(device)
elif decoder_name == "gru":
    decoder = DecoderGRU(
        voca_size=voca_size,
        emd_size=params["captioning"]["gru"]["embed_dim"],
        hidden_size=params["captioning"]["gru"]["hidden_dim"],
        max_len=params["captioning"]["max_caption_length"]
        ).to(device)


# optimizer
optimizer_name = params["captioning"]["optimizer"].lower()
if optimizer_name == "adam":
    optimizer = torch.optim.Adam(
        list(encoder.projector.parameters()) +
        list(decoder.parameters()),
        lr=params["captioning"]["learning_rate"],
    )
elif optimizer_name == "adamw":
    optimizer = torch.optim.AdamW(
        list(encoder.projector.parameters()) +
        list(decoder.parameters()),
        lr=params["captioning"]["learning_rate"],
        weight_decay=params["captioning"]["transformer"]["weight_decay"]
    )


# loss
criterion = nn.CrossEntropyLoss(
    ignore_index=w2i["<pad>"],
    label_smoothing=params["captioning"]["transformer"]["label_smoothing"]
)

start_epoch = 0

# 1. 설정값 정의 (yaml 파일에서 읽어오는 것을 추천)
my_config = {
    "model_name": model_name,
    "learning_rate": params["captioning"]["learning_rate"],
    "batch_size": params["captioning"]["batch_size"],
    "image_size": params["preprocess"]["image_size"],
    "seed": params["train"]["seed"],
    "epochs" : params["captioning"]["epochs"],
    "dataset_version": params["captioning"]["data"]["dataset_version"],
    "optimizer": params["captioning"]["optimizer"],
    "device": device.type,
}


# 2. W&B 초기화
wandb.init(
    project=params["logging"]["project_name"],
    entity="super-shift-working", # 팀 계정이 있다면 작성
    config=my_config,
    name=f'cap_{model_name}-{version}',
    tags=["captioning"]
)


# checkpoint setting
save_dir = params["captioning"]["checkpoint"]["save_dir"]
save_prefix = f"{model_name}_{version}"
best_path = os.path.join(save_dir, f"{save_prefix}_best.pt")
os.makedirs(save_dir, exist_ok=True)

start_epoch = 0
best_val_loss = float("inf")
if params["captioning"]["checkpoint"]["resume"]:
    start_epoch, best_val_loss = load_checkpoint(
            best_path,
            encoder,
            decoder,
            optimizer,
            device
            )

# train
log_dict = {}
for epoch in range(start_epoch, params["captioning"]["epochs"]):

    train_loss = train_one_epoch(
        encoder,
        decoder,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss = validation_one_epoch(
        encoder,
        decoder,
        val_loader,
        criterion,
        device,
    )

    print(f"Epoch {epoch+1} Train_Loss: {train_loss:.4f} Val_Loss: {val_loss:.4f}")

    log_dict.update({
        "train/loss": train_loss,
        "val/loss": val_loss
    })

    print('='*60)

    if val_loss < best_val_loss:
        best_val_loss = val_loss
        save_checkpoint(
            best_path,
            encoder,
            decoder,
            optimizer,
            epoch,
            train_loss,
            val_loss
        )

    # 지표 기록
    if epoch+1 < params["captioning"]["epochs"]:
        wandb.log(log_dict)

dec_atten_dir = os.path.join(params["captioning"]["heatmap"]["dec_atten_dir"], f"{model_name}_{version}_dec_atten.jpg")
enc_dec_atten_dir = os.path.join(params["captioning"]["heatmap"]["enc_dec_atten_dir"], f"{model_name}_{version}_cross_atten.jpg")
all_generated_sentence, all_references = make_show_all_caption(
        val_loader,
        encoder,
        decoder,
        optimizer,
        w2i,
        i2w,
        best_path,
        dec_atten_dir,
        enc_dec_atten_dir,
        params["captioning"]["heatmap"]["n_sample"],
        params["captioning"]["heatmap"]["layer"],
        device
    )

metric_result = evaluate_caption(all_generated_sentence, all_references)
log_dict.update({
            "BLEU_1": metric_result["bleu1"],
            "BLEU_2": metric_result["bleu2"],
            "BLEU_3": metric_result["bleu3"],
            "BLEU_4": metric_result["bleu4"],
            "CIDEr": metric_result["cider"],
        })

wandb.log(log_dict)

wandb.finish()