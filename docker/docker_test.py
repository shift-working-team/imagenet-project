# =========================================================
# requirements.txt 전체 라이브러리 import / 동작 검증 스크립트
# =========================================================

print("\n[1] Core")
import numpy as np
from typing_extensions import Annotated
print("✅ Core OK")

# =========================================================

print("\n[2] Deep Learning Framework")
import torch
import torchvision
import timm
from accelerate import Accelerator

print("Torch Version:", torch.__version__)
print("Torchvision Version:", torchvision.__version__)
print("CUDA Available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("CUDA Version:", torch.version.cuda)
    print("GPU:", torch.cuda.get_device_name(0))

print("✅ Deep Learning Framework OK")

# =========================================================

print("\n[3] Transformer / NLP / Multimodal")

import transformers
import tokenizers
import huggingface_hub
import safetensors
import sentencepiece
import open_clip
import nltk

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoProcessor,
)

# 실제 tokenizer 동작 테스트
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
tokens = tokenizer("hello world")

print("Tokenizer Test:", tokens["input_ids"])

print("✅ Transformers / NLP OK")

# =========================================================

print("\n[4] Dataset / Data Handling")

import datasets
import pandas as pd

df = pd.DataFrame({"a": [1, 2, 3]})

print(df.head())

print("Datasets Version:", datasets.__version__)
print("Pandas Version:", pd.__version__)

print("✅ Dataset Handling OK")

# =========================================================

print("\n[5] Image Processing / Vision")

import albumentations
import cv2
from PIL import Image
import imagehash

dummy = np.zeros((224, 224, 3), dtype=np.uint8)

transform = albumentations.HorizontalFlip(p=1.0)
augmented = transform(image=dummy)

pil_img = Image.fromarray(dummy)
hash_value = imagehash.average_hash(pil_img)

print("ImageHash:", hash_value)

print("✅ Vision Libraries OK")

# =========================================================

print("\n[6] Evaluation Metrics")

import pycocoevalcap
import rouge_score
import sacrebleu

# pycocoevalcap 실제 동작 검증
from pycocoevalcap.bleu.bleu import Bleu

bleu = Bleu(4)

gts = {0: ["a cat on the mat"]}
res = {0: ["a cat on the mat"]}

score, scores = bleu.compute_score(gts, res)

print("BLEU Score:", score)

# rouge-score 검증
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1"], use_stemmer=True)

scores = scorer.score(
    "the cat is on the mat",
    "the cat sat on the mat"
)

print("ROUGE:", scores)

# sacrebleu 검증
bleu_score = sacrebleu.corpus_bleu(
    ["the cat is on the mat"],
    [["the cat is on the mat"]]
)

print("SacreBLEU:", bleu_score.score)

print("✅ Evaluation Metrics OK")

# =========================================================

print("\n[7] Visualization")

import matplotlib
import matplotlib.pyplot as plt

plt.plot([1, 2, 3], [1, 4, 9])
plt.close()

print("Matplotlib Version:", matplotlib.__version__)

print("✅ Visualization OK")

# =========================================================

print("\n[8] Experiment / Logging")

import wandb

print("WandB Version:", wandb.__version__)

print("✅ WandB OK")

# =========================================================

print("\n[9] Utils / System")

import yaml
from tqdm import tqdm
import requests

sample_yaml = """
name: test
version: 1
"""

parsed = yaml.safe_load(sample_yaml)

print(parsed)

print("✅ Utils OK")

# =========================================================

print("\n[10] DVC")

import networkx
import dvc
import boto3

print("NetworkX Version:", networkx.__version__)

# boto3 client 생성 테스트
s3 = boto3.client("s3")

print("Boto3 Client Created")

print("✅ DVC / AWS OK")

# =========================================================

print("\n[11] Sentence Transformers")

import sentence_transformers
from sentence_transformers import SentenceTransformer

print("Sentence Transformers Version:", sentence_transformers.__version__)

print("✅ Sentence Transformers OK")

# =========================================================
import sklearn

print("sklearn Version:", sklearn.__version__)

print("✅ sklearn OK")

# =========================================================
import torchmetrics

print("torchmetrics Version:", torchmetrics.__version__)

print("✅ torchmetrics OK")

# =========================================================
import ipykernel

print("ipykernel Version:", ipykernel.__version__)

print("✅ ipykernel OK")

# =========================================================
import jupyterlab

print("jupyterlab Version:", jupyterlab.__version__)

print("✅ jupyterlab OK")

# =========================================================
print("\n[12] Notebook")

import notebook

print("Notebook Version:", notebook.__version__)

print("✅ Notebook OK")

# =========================================================

print("\n[13] Einops")

from einops import rearrange

x = torch.randn(2, 3, 4)
y = rearrange(x, "b c h -> b h c")

print("Einops Output Shape:", y.shape)

print("✅ Einops OK")

# =========================================================

print("\n[14] UMAP")

import umap

reducer = umap.UMAP(
    n_neighbors=5,
    n_components=2,
    random_state=42
)

dummy = np.random.rand(20, 8)

embedding = reducer.fit_transform(dummy)

print("UMAP Output Shape:", embedding.shape)

print("✅ UMAP OK")

# =========================================================

print("\n[15] Grad-CAM")

from pytorch_grad_cam import GradCAM

print("GradCAM Class:", GradCAM)

print("✅ Grad-CAM OK")

# =========================================================

print("\n[16] TTACH")

import ttach as tta

transforms = tta.aliases.flip_transform()

print("TTACH Transform:", transforms)

print("✅ TTACH OK")

# =========================================================

print("\n[17] Gradio")

import gradio as gr

print("Gradio Version:", gr.__version__)

print("✅ Gradio OK")

# =========================================================

print("\n[18] FastAPI")

import fastapi

app = fastapi.FastAPI()

print("FastAPI Version:", fastapi.__version__)

print("✅ FastAPI OK")

# =========================================================

print("\n[19] Starlette")

import starlette

print("Starlette Version:", starlette.__version__)

print("✅ Starlette OK")

# =========================================================
print("\n" + "=" * 60)
print("🎉 ALL IMPORTS & BASIC FUNCTION TESTS PASSED")
print("=" * 60)