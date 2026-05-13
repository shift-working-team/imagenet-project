import sys
sys.path.append("/workspace/src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

import yaml
import wandb
import subprocess
from metrics.captioning.bleu import calculate_bleu_n
from metrics.captioning.cider import calculate_cider

from dataset.build_voca import build_voca, tokenizer
from dataset.dataset import CaptionDataset
from transforms.image_transform import get_train_transform
from engines.Captioning_trainer.Resnet18_LSTM_trainer import train_one_epoch
from engines.Captioning_trainer.Resnet18_LSTM_validator import validation_one_epoch
from models.lstm import DecoderLSTM
from models.resnet18_seon import EncoderResnet18


# params
with open("/workspace/params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)


# device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# vocab
w2i, i2w, voca_size = build_voca(
    params["data"]["captions_file"],
    min_freq=params["tokenizer"]["min_freq"],
    max_size=params["tokenizer"]["max_vocab_size"]
)


# transform
transform = get_train_transform()


# train dataset
train_dataset = CaptionDataset(
    json_path=params["data"]["captions_file"],
    image_dir=params["data"]["raw_dir"],
    w2i=w2i,
    tokenizer=tokenizer,
    split="train",
    transform=transform,
    max_len=params["preprocess"]["max_caption_length"]
)

# validation dataset
val_dataset = CaptionDataset(
    json_path=params["data"]["captions_file"],
    image_dir=params["data"]["raw_dir"],
    w2i=w2i,
    tokenizer=tokenizer,
    split="val",
    transform=transform,
    max_len=params["preprocess"]["max_caption_length"]
)


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
encoder = EncoderResnet18().to(device)
decoder = DecoderLSTM(
    voca_size=voca_size,
    emd_size=params["model"]["lstm"]["embed_dim"],
    hidden_size=params["model"]["lstm"]["hidden_dim"],
    max_len=params["preprocess"]["max_caption_length"]
    ).to(device)


# loss
criterion = nn.CrossEntropyLoss(
    ignore_index=w2i["<pad>"]
)


# optimizer
optimizer = torch.optim.Adam(
    list(encoder.projection.parameters()) +
    list(decoder.parameters()),
    lr=params["model"]["lstm"]["learning_rate"]
)


def get_git_revision_hash():
    # 전체 해시 출력
    return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


# 1. 설정값 정의 (yaml 파일에서 읽어오는 것을 추천)
my_config = {
    "model_name": "cnn-lstm",
    "learning_rate": params["model"]["lstm"]["learning_rate"],
    "batch_size": params["train"]["batch_size"],
    "image_size": params["preprocess"]["image_size"],
    "seed": params["train"]["seed"],
    "epochs" : params["train"]["epochs"],
    "dataset_version": "dvc-v1",
    "optimizer": params["train"]["optimizer"],
    "dvice": device.type,
    "commit_hash": get_git_revision_hash()
}


# 2. W&B 초기화
wandb.init(
    project="imagenet-project",
    entity="super-shift-working", # 팀 계정이 있다면 작성
    config=my_config,
    name="Resnet18-lstm-20260513-baseline"
)


# train
for epoch in range(params["train"]["epochs"]):

    train_loss = train_one_epoch(
        encoder,
        decoder,
        train_loader,
        criterion,
        optimizer,
        device
    )

    val_loss, feature, target_inx = validation_one_epoch(
        encoder,
        decoder,
        val_loader,
        criterion,
        device,
        w2i,
    )

    print(f"Epoch {epoch+1} Train_Loss: {train_loss:.4f} Val_Loss: {val_loss:.4f}")

    generated_inx = decoder.generate(
            feature,
            torch.tensor([w2i["<sos>"]]),
            torch.tensor([w2i["<eos>"]])
            )
    
    # <end> 제거
    generated_inx = generated_inx[:-1]
    end_inx = torch.where(target_inx == w2i["<eos>"])[0]
    target_inx = target_inx[:end_inx].tolist()
    
    print(f'생성캡션: {generated_inx}')
    print(f'타겟캡션: {target_inx}')

    generated_sentence = []
    for i in generated_inx:
        generated_sentence.append(i2w[i])
    generated_sentence = " ".join(generated_sentence)

    target_sentence = []
    for i in target_inx:
        target_sentence.append(i2w[i])
    target_sentence = " ".join(target_sentence)

    print(f'생성 문장: {generated_sentence}')
    print(f'타겟 문장: {target_sentence}')

    generated_dict= {epoch:generated_sentence}
    target_dict= {epoch:target_sentence}

    
    # 4. 지표 기록
    wandb.log({
        "train/loss": train_loss,
        "validation/loss": val_loss,
        "bleu":calculate_bleu_n(generated_dict, target_dict)
    })


wandb.finish()