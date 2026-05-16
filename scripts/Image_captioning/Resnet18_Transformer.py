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
from dataset.captioning_dataset import CaptionDataset
from transforms.image_transform import get_caption_transform
from engines.Captioning_trainer.Resnet18_Decoder_trainer import train_one_epoch
from engines.Captioning_trainer.Resnet18_Decoder_validator import validation_one_epoch
from models.transformer import DecoderTransformer
from models.resnet18 import EncoderResnet18

from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider


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
transform = get_caption_transform()


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
decoder = DecoderTransformer(
    d_model=params["model"]["transformer"]["d_model"],
    nhead=params["model"]["transformer"]["nhead"],
    num_layers=params["model"]["transformer"]["num_layers"],
    voca_size=voca_size,
    max_len=params["preprocess"]["max_caption_length"]
    ).to(device)


# optimizer
optimizer = torch.optim.Adam(
    list(encoder.projector.parameters()) +
    list(decoder.parameters()),
    lr=params["model"]["transformer"]["learning_rate"]
)


# loss
criterion = nn.CrossEntropyLoss(
    ignore_index=w2i["<pad>"]
)


# def get_git_revision_hash():
#     # 전체 해시 출력
#     return subprocess.check_output(['git', 'rev-parse', 'HEAD']).decode('ascii').strip()


# # 1. 설정값 정의 (yaml 파일에서 읽어오는 것을 추천)
# my_config = {
#     "model_name": "cnn-transformer",
#     "learning_rate": params["model"]["transformer"]["learning_rate"],
#     "batch_size": params["train"]["batch_size"],
#     "image_size": params["preprocess"]["image_size"],
#     "seed": params["train"]["seed"],
#     "epochs" : params["train"]["epochs"],
#     "dataset_version": "dvc-v1",
#     "optimizer": params["train"]["optimizer"],
#     "dvice": device.type,
#     "commit_hash": get_git_revision_hash()
# }


# # 2. W&B 초기화
# wandb.init(
#     project="imagenet-project",
#     entity="super-shift-working", # 팀 계정이 있다면 작성
#     config=my_config,
#     name="test-docker-20260516-v1"
# )


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

    val_loss, all_feature, all_references = validation_one_epoch(
        encoder,
        decoder,
        val_loader,
        criterion,
        device,
        epoch
    )

    print(f"Epoch {epoch+1} Train_Loss: {train_loss:.4f} Val_Loss: {val_loss:.4f}")

    log_dict = {
        "train/loss": train_loss,
        "val/loss": val_loss
    }

    if epoch >= 5 and epoch % 5 == 0:
        generated_token = decoder.generate(
                all_feature,
                torch.full((all_feature.size(0),), w2i["<sos>"]),
                torch.tensor([w2i["<eos>"]])
                )

        generated_sentence = []
        for sentence in generated_token:
            end_inx = sentence.index(w2i["<eos>"])
            sentence = sentence[:end_inx]

            word = []
            for i in sentence:
                word.append(i2w[i])
            
            generated_sentence.append(' '.join(word))

        generated_dict = {i:[sentence] for i, sentence in enumerate(generated_sentence)}
        references_dict = {i:list(sentences) for i, sentences in enumerate(all_references)}

        bleu_scorer = Bleu(4)
        bleu_score, bleu_scores = bleu_scorer.compute_score(
            references_dict,
            generated_dict
        )

        cider_scorer = Cider()
        cider_score, cider_scores = cider_scorer.compute_score(
            references_dict,
            generated_dict
        )

        sample_index = 0

        print('-'*60)
        print(f'Generated sentence: {generated_dict[sample_index]}')
        print('-'*60)

        print(f'references sentence1: {references_dict[sample_index][0]}')
        print(f'references sentence2: {references_dict[sample_index][1]}')
        print('-'*60)
        
        print("BLEU-1:", bleu_score[0])
        print("BLEU-2:", bleu_score[1])
        print("BLEU-3:", bleu_score[2])
        print("BLEU-4:", bleu_score[3])
        print("CIDEr:", cider_score)

        log_dict = {
            "train/loss": train_loss,
            "val/loss": val_loss,
            "bleu1":bleu_score[0],
            "bleu2":bleu_score[1],
            "bleu3":bleu_score[2],
            "bleu4":bleu_score[3],
            "cider":cider_score
        }

    print('='*60)

#     # 4. 지표 기록
#     wandb.log(log_dict)

# wandb.finish()