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
| 최종 모델          | Swin-T              |
| 데이터셋           | cls_raw-20260525-v2 |
| Test Accuracy  | 89.64%              |
| Test Macro F1  | 89.61%              |
| Test Precision | 90.01%              |
| Test Recall    | 89.61%              |

### 이미지 캡셔닝

| 항목        | 값                 |
| --------- | ----------------- |
| Encoder   | Swin-T            |
| Decoder   | Transformer       |
| Tokenizer | Word-level        |
| 탐색 방법     | Beam Search (k=5) |
| BLEU-1    | (최종 결과 입력)        |
| BLEU-4    | (최종 결과 입력)        |
| CIDEr     | (최종 결과 입력)        |

---

## 3. 주요 기능

### 이미지 분류

- CNN Backbone 비교
  - ResNet18
  - EfficientNet-B0
  - ConvNeXt-Tiny
  - MobileNetV3-Small

- Transformer Backbone 비교
  - ViT-B/16
  - Swin-T
  - DeiT-Tiny

- 데이터셋 정제 전/후 성능 비교
- Augmentation 실험 (MixUp, CutMix)
- Hyperparameter Tuning
  - Learning Rate
  - Weight Decay
  - Scheduler
  - Label Smoothing

- Confusion Matrix 생성
- Grad-CAM 시각화
- UMAP 기반 Latent Space 분석
- Gradio 기반 추론 데모

### 이미지 캡셔닝

- Decoder 비교
  - LSTM
  - GRU
  - Transformer

- Encoder 비교
  - ResNet18
  - ViT
  - Swin-T

- Tokenizer 비교
  - Word-level
  - Subword-level

- Transformer 구조 실험
  - Layer 수
  - Hidden Dimension
  - Multi-Head Attention

- 정규화 및 최적화 실험
  - Dropout
  - Weight Decay
  - Learning Rate
  - Scheduler

- Greedy Search / Beam Search 비교
- BLEU 및 CIDEr 평가
- Gradio 기반 캡션 생성 데모

### 데이터 및 실험 관리

* 데이터 수집 및 정제
* DVC 기반 데이터 버전 관리
* W&B 기반 실험 추적
* Docker 기반 개발 환경 통일

---

## 4. 프로젝트 구조

## 프로젝트 구조

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
│   ├── gradio_classification_demo.py
│   └── gradio_captioning_demo.py
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
3. Transformer Backbone 비교
4. 최종 Backbone 선정
5. 데이터셋 정제 후 재검증
6. Augmentation 비교
7. Hyperparameter Tuning
8. 최종 평가

### 이미지 캡셔닝

1. Decoder 비교 (LSTM / GRU / Transformer)
2. Encoder 비교
3. Tokenizer 비교
4. 모델 구조 개선
5. Optimization 실험
6. Beam Search 평가
7. 최종 평가

---

## 7. 실행 방법

### 저장소 복제

```bash
git clone https://github.com/Mini-imagenet-project/imagenet-project.git
cd imagenet-project
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
python scripts/train_classification.py
```

### 이미지 캡셔닝 학습

```bash
python scripts/train_captioning.py
```

### 이미지 분류 데모 실행

```bash
python scripts/gradio_classification_demo.py
```

### 이미지 캡셔닝 데모 실행

```bash
python scripts/gradio_captioning_demo.py
```

---

## 8. MLOps 및 실험 관리

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

## 9. 팀 구성

| 역할            | 담당 |
| ------------- | --         |
| PM            | 김민주      |
| Model Lead    | 안가인      |
| Seq & MML     | 선종일      |
| MLOps         | 팀원 전체   |

