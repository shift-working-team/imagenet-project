# Mini-ImageNet+ : 이미지 분류 및 이미지 캡셔닝 프로젝트

## 1. 프로젝트 개요

본 프로젝트는 이미지 분류(Image Classification)와 이미지 캡셔닝(Image Captioning)을 수행하는 컴퓨터 비전 프로젝트이다.

이미지 분류에서는 다양한 CNN 및 Transformer Backbone을 비교하여 최종 분류 모델을 선정하였고, 이미지 캡셔닝에서는 Encoder-Decoder 기반 구조를 실험하여 최종 캡셔닝 모델을 구축하였다.

또한 데이터 수집, 데이터 정제, 모델 학습, 성능 평가, 실험 관리(DVC, W&B), Docker 기반 환경 구축, Gradio 데모 배포까지 전체 머신러닝 파이프라인을 구현하였다.

---

## 2. 최종 성능 결과

### 이미지 분류

| 항목             | 값                   |
| -------------- | ------------------- |
| Dataset           | cls_raw-20260525-v2 |
| 최종 모델          | Swin-T              |
| Test Accuracy  | 89.64%              |
| Test Macro F1  | 89.61%              |
| Test Precision | 90.01%              |
| Test Recall    | 89.61%              |

### 이미지 캡셔닝

| 항목        | 값                 |
| --------- | ----------------- |
| Dataset   | cap_raw-20260524-v1 |
| Encoder   | Swin-T            |
| Decoder   | Transformer       |
| Tokenizer | Word-level        |
| 탐색 방법     | Beam Search (k=3) |
| BLEU-4    | 0.1997        |
| CIDEr     | 0.6054        |

---

## 3. 주요 기능

### 이미지 분류

- 이미지 분류 모델 학습
- Grad-CAM 시각화
- UMAP 기반 Latent Space 분석
- Gradio 추론 데모

### 이미지 캡셔닝

- 이미지 기반 캡션 생성
- BLEU / CIDEr 평가
- Beam Search 기반 문장 생성
- Attention Heatmap 분석
- Gradio 추론 데모

### 데이터 및 실험 관리

* DVC 기반 데이터 버전 관리
* W&B 기반 실험 추적
* Docker 기반 개발 환경 통일

---

## 4. 프로젝트 구조

```text
root/
├── docs/
│   └── dataset_card.md
├── data/
│   ├── raw/
│   └── captioning/
├── src/
│   ├── dataset/
│   ├── models/
│   ├── engines/
│   ├── transforms/
│   └── utils/
├── scripts/
│   ├── train_classification.py
│   ├── train_captioning.py
│   ├── evaluate_classification.py
│   └── app.py
├── outputs/
├── docker/
├── params.yaml
├── dvc.yaml
└── README.md
```


---

## 5. 데이터셋

데이터셋에 대한 상세 내용은 아래 문서를 참고한다.

[데이터셋 카드](docs/dataset_card.md)

포함 내용

* 분류 데이터셋
* 캡셔닝 데이터셋
* 데이터 수집 기준
* 데이터 정제 과정
* 버전 관리 이력
* 데이터셋 한계

---

## 6. 실험 과정

### 이미지 분류

1. ResNet18 Baseline 구축

2. CNN Backbone 비교
  - ResNet18
  - EfficientNet-B0
  - ConvNeXt-Tiny
  - MobileNetV3-Small

3. Transformer Backbone 비교
  - ViT-B/16
  - Swin-T
  - DeiT-Tiny

4. Best Backbone 선정
  - CNN Best vs Transformer Best
  - Seed(42, 7, 21) 반복 실험 수행

5. 데이터셋 정제 전/후 성능 비교
  - raw-20260509-v1
  - cls_raw-20260525-v2

6. Augmentation 실험
  - MixUp
  - CutMix

7. Hyperparameter Tuning
  - Learning Rate
  - Batch Size
  - Scheduler
  - Weight Decay
  - Label Smoothing

8. 최종 모델 평가
  - Accuracy
  - Macro F1
  - Precision / Recall
  - Confusion Matrix
  - Grad-CAM
  - UMAP

### 이미지 캡셔닝

1. Decoder Baseline 선정
  - LSTM
  - GRU
  - Transformer

2. Encoder 비교
  - ResNet18
  - ViT
  - Swin-T

3. Tokenizer 비교
  - Word-level
  - Subword-level

4. Transformer 구조 실험
  - Layer 수
  - Hidden Dimension
  - Multi-Head Attention
  - ReLU, GELU 비교
  - Positional 비교 (sinusoidal, Learnable)

5. 정규화 및 일반화 성능 개선
  - Dropout
  - Weight Decay
  - Label Smoothing

6. 최적화 실험
  - Learning Rate
  - Learning Rate Scheduler
  - Batch Size

7. Decoding 비교
  - Greedy Search
  - Beam Search

8. 최종 평가
  - 생성 캡션 정성 평가
  - BLEU 및 CIDEr 평가
  - Attention Heatmap
  - Gradio 기반 캡션 생성 데모

---

## 7. 실행 방법

### 저장소 복제

```bash
git clone https://github.com/Mini-imagenet-project/imagenet-project.git
cd imagenet-project
```

### docker 이미지 pull

```bash
docker pull j1seon/supercoding:v5
```

### docker 컨테이너 실행

```bash
docker run --shm-size=8g -it --gpus all --name 컨테이너이름 -v "본인절대경로:/workspace" j1seon/supercoding:v5 /bin/bash
```

### 라이브러리 설치

```bash
pip install -r docker/requirements.txt
```

### 데이터 다운로드

```bash
dvc pull
```

### 이미지 분류 학습

```bash
# 직접 학습 실행
python scripts/train_classification.py

# DVC 파이프라인으로 실행
dvc repro train_classification
```

### 이미지 캡셔닝 학습

```bash
# 직접 학습 실행
python scripts/train_captioning.py

# DVC 파이프라인으로 실행
dvc repro train_captioning
```

### 이미지 분류 & 캡셔닝 추론 데모 

```bash
python scripts/app.py
```

---

## 8. MLOps 및 실험 관리

## 개발 환경

| 항목 | 버전 |
| --- | --- |
| Python | 3.10 |
| PyTorch | 2.11.0+cu128 |
| CUDA | 12.9 |

### Docker

* 개발 환경 통일
* 실행 환경 재현

### DVC

* 데이터 버전 관리
* 실험 파이프라인 관리

### Weights & Biases

* 실험 로그 관리
* 성능 지표 시각화
* 체크포인트 추적

### GitHub

* 코드 버전 관리
* Pull Request 기반 협업

---

### 9. 참고 자료
1. 실험 로그(wandb)
   - URL : https://wandb.ai/super-shift-working/imagenet-project
2. 최종 가중치 및 시연 영상(Dagshub)
   - URL : https://dagshub.com/Shift-working/imagenet-project
   - 분류 모델 가중치 경로 : root/outputs/classification/cls_swin-t_base_cls_raw-20260525-v2_lr-0005_bs-32_adamw_none_wdc-0.05_ls-0.0_best.pth
   - 캡션 모델 가중치 경로 : root/outputs/captioning/swin-transformer_final_best.pt
   - 시연 영상 경로 : root/outputs/슈퍼코딩 2차 프로젝트 시연 영상.mp4

---

## 10. 팀 구성

| 역할            | 담당 |
| ------------- | --         |
| PM            | 김민주      |
| Model Lead    | 안가인      |
| Seq & MML     | 선종일      |
| MLOps         | 팀원 전체   |

