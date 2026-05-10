import sys
sys.path.append("/workspace/src")

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset.build_voca import build_voca, tokenizer
from dataset.dataset import CaptionDataset
from transforms.image_transform import get_train_transform
from models.cnn_lstm import CNN_LSTM
from engines.captioning_trainer import train_one_epoch



# config
json_path = "/workspace/data/annotations/annotation.json"
image_dir = "/workspace/data/raw/"

BATCH_SIZE = 8
EPOCHS = 5
LR = 1e-4



# device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)



# vocab
w2i, i2w, vocab_size = build_voca(
    json_path,
    min_freq=1,
    max_size=10000
)



# transform
transform = get_train_transform()



# dataset
train_dataset = CaptionDataset(
    json_path=json_path,
    image_dir=image_dir,
    w2i=w2i,
    tokenizer=tokenizer,
    split="train",
    transform=transform,
    max_len=30
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)



# model
model = CNN_LSTM(
    vocab_size=vocab_size
).to(device)



# loss
criterion = nn.CrossEntropyLoss(
    ignore_index=w2i["<pad>"]
)



# optimizer
optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LR
)



# train
for epoch in range(EPOCHS):

    loss = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    print(f"Epoch {epoch+1} Loss: {loss:.4f}")