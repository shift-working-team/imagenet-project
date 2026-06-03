# Image Classification & Captioning Project

## 1. 프로젝트 개요

본 프로젝트는 다양한 이미지 데이터를 입력받아 올바른 클래스로 분류하는 **이미지 분류 모델**을 구축하고, 이미지에 대한 자연어 설명을 생성하는 **이미지 캡셔닝 모델**까지 함께 실험한 컴퓨터 비전 프로젝트이다.

이미지 분류에서는 여러 CNN 계열 및 Transformer 계열 backbone을 비교하고, 데이터셋 정제, augmentation, optimizer, scheduler, weight decay 등의 조건을 단계적으로 실험하여 최종 분류 모델을 선정하였다.

이미지 캡셔닝에서는 이미지와 캡션 JSON 데이터를 기반으로 Encoder-Decoder 구조의 모델을 학습하고, 생성된 캡션의 품질을 BLEU, CIDEr 등의 지표로 평가하였다.

본 프로젝트의 핵심 목표는 단순히 모델을 학습하는 것이 아니라, **데이터 수집 → 데이터 정제 → 모델 학습 → 성능 평가 → 실험 관리 → 시각화 및 데모**까지 이어지는 전체 머신러닝 파이프라인을 구현하는 것이다.

---

## 2. 주요 기능

### 이미지 분류 모델 학습

* ResNet18, EfficientNet-B0, ConvNeXt-Tiny, MobileNetV3-Small, ViT-B/16, Swin-T, DeiT-Tiny 등 다양한 backbone 지원
* train / validation split 기반 학습
* Adam, SGD, AdamW optimizer 지원
* label smoothing, weight decay, cosine scheduler 옵션 제공
* checkpoint 저장 및 재학습을 위한 resume 기능 지원

### 이미지 분류 데이터 증강

* 기본 이미지 전처리 및 정규화 지원
* 학습용 / 검증용 transform 분리
* MixUp, CutMix 기반 augmentation 지원
* augmentation 적용 여부에 따른 validation 성능 비교 가능

### 분류 모델 평가

* 최종 test dataset 기반 평가 수행
* accuracy, macro F1, precision, recall 계산
* classification report 저장
* confusion matrix 저장
* prediction 결과 CSV 저장
* 정답 / 오답 예시 이미지 저장

### Grad-CAM 시각화

* Swin-T 분류 모델 기준 Grad-CAM 생성
* 정답 / 오답 샘플에 대한 heatmap 저장
* 모델이 이미지의 어느 영역을 보고 예측했는지 시각적으로 확인 가능
* Gradio 데모에서도 Grad-CAM overlay 결과 표시 가능

### 이미지 분류 Gradio 데모

* 사용자가 이미지를 업로드하면 최종 checkpoint 기반으로 클래스 예측
* Top-K 클래스 확률 출력
* 예측 결과 요약 제공
* Grad-CAM 시각화 이미지 제공

### 이미지 캡셔닝 모델 학습

* Encoder: ResNet18, Swin, ViT 지원
* Decoder: Transformer, LSTM, GRU 지원
* 이미지와 캡션 JSON을 이용한 학습 / 검증 파이프라인 구성
* checkpoint 저장 및 resume 기능 지원

### 캡션 생성 및 평가

* greedy search / beam search 기반 캡션 생성 옵션 제공
* BLEU-1~4, CIDEr 등 캡셔닝 평가 지표 계산
* 생성 캡션과 reference caption 비교 결과 생성
* 캡션 품질 검토를 위한 결과 파일 저장

### 캡셔닝 어텐션 시각화

* decoder attention, encoder-decoder attention heatmap 저장
* 특정 샘플과 layer를 지정해 attention 시각화 가능
* 모델이 캡션을 생성할 때 이미지의 어느 부분에 집중했는지 확인 가능

### 토크나이저 및 어휘 구축

* 캡션 데이터 기반 vocabulary 생성
* word-level tokenizer 지원
* SentencePiece 기반 subword tokenizer 학습 및 사용 옵션 제공
* 여러 vocab size의 tokenizer 모델 포함

### 데이터 수집 및 정제

* Hugging Face, Kaggle, Unsplash 등에서 이미지 데이터 수집 스크립트 제공
* 클래스별 이미지 개수 확인
* 저해상도 이미지 확인 및 필터링
* 중복 이미지 해시 검사
* 깨진 이미지 제거
* label ambiguity가 큰 이미지 제거
* 클래스별 샘플 선별 기능 제공

### 자동 캡션 데이터 생성

* BLIP, Florence2, GIT, ViT-GPT2 기반 이미지 캡션 생성 스크립트 제공
* 생성 캡션 정제 및 중복 제거
* train / validation / test split 할당 기능 포함
* CLIP score 기반 캡션 품질 확인 기능 제공

### 잠재 공간 분석

* 학습된 분류 모델의 feature 추출
* UMAP 기반 latent space 시각화
* 클래스별 feature 분포 확인
* metadata 및 scatter plot 저장
* W&B를 통한 시각화 결과 기록 가능

### 실험 관리

* `params.yaml` 기반 하이퍼파라미터 관리
* DVC 파이프라인으로 학습, 평가, latent space 분석 단계 정의
* Weights & Biases를 통한 실험 로그, metric, artifact 기록
* Git / GitHub 기반 코드 버전 관리

### Docker 실행 환경

* Dockerfile과 requirements 제공
* 컨테이너 기반 학습 / 평가 실행 지원
* `/workspace` 기준 경로 구조 사용
* 팀원 간 동일한 실행 환경을 구성할 수 있도록 설계

---

## 3. 프로젝트 구조

```text
root/
├── data/
│   ├── raw/
│   │   ├── airplane/
│   │   ├── apple/
│   │   ├── ...
│   │   └── waffle/
│   ├── captioning/
│   │   ├── raw/
│   │   └── annotations/
│   ├── annotations/
│   │   └── annotation.json
│   ├── raw.dvc
│   └── captioning.dvc
├── scripts/
│   ├── train_classification.py
│   ├── train_captioning.py
│   ├── evaluate_classification.py
│   ├── analyze_latent_space.py
│   └── gradio_demo.py
├── src/
│   ├── caption/
│   ├── collection/
│   ├── dataset/
│   ├── debug/
│   ├── engines/
│   ├── metrics/
│   ├── models/
│   ├── transforms/
│   ├── utils/
│   └── visualization/
├── outputs/
│   ├── classification/
│   │   ├── correct_examples/
│   │   └── incorrect_examples/
│   ├── captioning/
│   │   └── heatmap/
│   └── latent_space/
├── checkpoints/
│   ├── classification/
│   └── captioning/
├── docker/
│   ├── dockerfile
│   ├── requirements.txt
│   └── docker_test.py
├── wandb/
├── params.yaml
├── dvc.yaml
├── dvc.lock
├── outputs.dvc
└── README.md
```

## 폴더별 역할과 기능

| 폴더/파일                                | 역할                                                                                       |
| ------------------------------------ | ---------------------------------------------------------------------------------------- |
| `data/`                              | 학습과 평가에 사용하는 데이터 저장 공간                                                                   |
| `data/raw/`                          | 이미지 분류용 원본 데이터셋. 클래스별 폴더 구조로 구성                                                          |
| `data/captioning/`                   | 이미지 캡셔닝용 데이터셋 저장 공간                                                                      |
| `data/captioning/raw/`               | 캡셔닝 모델에 입력되는 이미지 데이터                                                                     |
| `data/captioning/annotations/`       | 캡션 학습 / 검증 / 테스트용 JSON annotation 파일                                                     |
| `data/annotations/`                  | 별도 annotation JSON 저장 공간                                                                 |
| `scripts/`                           | 프로젝트 실행용 메인 스크립트 모음                                                                      |
| `scripts/train_classification.py`    | 이미지 분류 모델 학습 실행                                                                          |
| `scripts/train_captioning.py`        | 이미지 캡셔닝 모델 학습 실행                                                                         |
| `scripts/evaluate_classification.py` | 분류 모델 최종 평가, 지표 저장, 정답 / 오답 예시 저장                                                        |
| `scripts/analyze_latent_space.py`    | 모델 feature를 추출해 latent space 시각화                                                         |
| `scripts/gradio_demo.py`             | 이미지 업로드 기반 분류 예측 Gradio 데모 실행                                                            |
| `src/`                               | 실제 모델, 데이터셋, 학습 로직, 유틸리티 코드가 들어 있는 핵심 소스 폴더                                              |
| `src/models/`                        | ResNet, EfficientNet, ConvNeXt, MobileNet, ViT, Swin, DeiT, LSTM, GRU, Transformer 모델 정의 |
| `src/dataset/`                       | 분류 / 캡셔닝 Dataset, vocabulary 생성, tokenizer, collate 함수 관리                                |
| `src/engines/`                       | train / validation loop 구현                                                               |
| `src/transforms/`                    | 이미지 전처리, augmentation, MixUp, CutMix 관련 코드                                               |
| `src/metrics/`                       | 평가 지표 계산 및 결과 정리                                                                         |
| `src/visualization/`                 | Grad-CAM, attention heatmap 등 시각화 코드                                                     |
| `src/caption/`                       | BLIP, Florence2, GIT, ViT-GPT2 기반 자동 캡션 생성 및 CLIP score 확인                               |
| `src/collection/`                    | 데이터 다운로드, 클래스 개수 확인, 이미지 필터링, 샘플 선택 등 데이터 수집 / 정제 코드                                     |
| `src/utils/`                         | checkpoint 저장 / 로드, scheduler 등 공통 유틸리티                                                  |
| `src/debug/`                         | 모델 forward 테스트 등 디버깅용 코드                                                                 |
| `outputs/`                           | 학습 / 평가 결과 저장 공간                                                                         |
| `outputs/classification/`            | 분류 모델 평가 결과, 정답 / 오답 예시 저장                                                               |
| `outputs/captioning/`                | 캡셔닝 모델 결과와 attention heatmap 저장                                                          |
| `outputs/latent_space/`              | UMAP scatter plot, metadata 등 latent space 분석 결과 저장                                      |
| `checkpoints/`                       | 모델 checkpoint 저장용 폴더                                                                     |
| `docker/`                            | Docker 실행 환경 정의                                                                          |
| `wandb/`                             | Weights & Biases 실험 로그 저장                                                                |
| `params.yaml`                        | 데이터 경로, 모델, 학습 설정, 하이퍼파라미터 관리                                                            |
| `dvc.yaml`                           | DVC 파이프라인 단계 정의                                                                          |
| `dvc.lock`                           | DVC 파이프라인 실행 상태 및 의존성 lock 파일                                                            |
| `*.dvc`                              | DVC로 관리되는 데이터 / 출력 추적 파일                                                                 |
| `README.md`                          | 프로젝트 설명 문서                                                                               |

---

## 4. 데이터셋

## 이미지 분류 데이터셋

본 프로젝트의 이미지 분류 데이터셋은 최종적으로 **50개 클래스**, 총 **10,370장** 규모로 구성되었다.

최종 실험 기준 데이터셋 버전은 `cls_raw-20260525-v2`이며, train / validation / test split은 **70 / 15 / 15** 비율로 구성하였다.

| 항목         | 내용                                    |
| ---------- | ------------------------------------- |
| 데이터셋 버전    | `cls_raw-20260525-v2`                 |
| 클래스 수      | 50개                                   |
| 총 이미지 수    | 10,370장                               |
| 클래스당 이미지 수 | 약 180~220장                            |
| Split      | train 70% / validation 15% / test 15% |
| 입력 이미지 크기  | 224 x 224                             |
| 주요 전처리     | Resize, ToTensor, Normalize           |
| 관리 방식      | DVC 기반 데이터 버전 관리                      |

### 데이터 수집 방식

이미지는 공개 데이터셋과 웹 기반 수집을 함께 활용하였다.

주요 수집 출처는 다음과 같다.

* Hugging Face
* Kaggle
* Unsplash
* 웹 크롤링

초기 수집 단계에서는 클래스당 약 200장을 목표로 하였으며, 다양한 출처에서 이미지를 수집하여 특정 출처나 이미지 스타일에 편향되지 않도록 구성하였다.

### 데이터 정제 기준

모델이 잘못된 이미지를 학습하지 않도록 데이터셋 정제를 수행하였다.

주요 정제 항목은 다음과 같다.

| 정제 항목                  | 목적                          |
| ---------------------- | --------------------------- |
| 깨진 이미지 제거              | 학습 중 오류 및 잘못된 feature 학습 방지 |
| 중복 이미지 제거              | 특정 이미지에 과적합되는 현상 완화         |
| 저해상도 이미지 제거            | 품질이 낮은 이미지로 인한 성능 저하 방지     |
| label ambiguity 이미지 제거 | 여러 클래스가 섞여 있는 이미지로 인한 혼동 감소 |
| 클래스별 이미지 수 보완          | 클래스 간 데이터 수 불균형 완화          |

데이터셋을 `raw-20260509-v1`에서 `cls_raw-20260525-v2`로 정제한 후, validation loss가 감소하고 validation accuracy 및 macro F1이 향상되었다. 이를 통해 모델 구조뿐 아니라 **데이터 품질이 모델 성능에 매우 큰 영향을 준다**는 것을 확인하였다.

---

## 5. 모델 실험 및 최종 결과

## 이미지 분류 실험 흐름

이미지 분류 실험은 다음 순서로 진행하였다.

```text
Step 1. Baseline 실험: ResNet18
Step 2. Backbone 비교: CNN 계열 vs Transformer 계열
Step 3. Best Backbone 선정
Step 4. 데이터셋 정제 후 재검증
Step 5. Augmentation 비교: None / MixUp / CutMix
Step 6. Hyperparameter Tuning
Step 7. Test Dataset 최종 평가
```

### 실험 모델

| 구분          | 모델                | 역할                       |
| ----------- | ----------------- | ------------------------ |
| CNN         | ResNet18          | baseline 모델              |
| CNN         | EfficientNet-B0   | CNN 계열 주요 비교 모델          |
| CNN         | ConvNeXt-Tiny     | CNN 계열 비교 모델             |
| CNN         | MobileNetV3-Small | 경량 CNN 비교 모델             |
| Transformer | ViT-B/16          | Vision Transformer 비교 모델 |
| Transformer | Swin-T            | 최종 선정 backbone           |
| Transformer | DeiT-Tiny         | Transformer 계열 비교 모델     |

### 최종 선정 모델

최종 이미지 분류 backbone으로는 **Swin-T**를 선정하였다.

Swin-T는 다른 후보 모델 대비 validation macro F1, validation loss 안정성, 일반화 성능 측면에서 가장 안정적인 결과를 보였다.

| 항목                 | 최종 설정                                       |
| ------------------ | ------------------------------------------- |
| Backbone           | Swin-T                                      |
| Dataset            | `cls_raw-20260525-v2`                       |
| Epoch              | 50                                          |
| Batch Size         | 32                                          |
| Learning Rate      | 0.0005                                      |
| Optimizer          | Adam / AdamW 실험                             |
| Loss Function      | CrossEntropyLoss                            |
| Input Size         | 224 x 224                                   |
| Evaluation Metrics | Accuracy, Macro F1, Precision, Recall, Loss |

### 데이터셋 정제 후 성능 변화

정제 전 데이터셋인 `raw-20260509-v1` 기준 Swin-T 실험에서는 seed에 따라 validation macro F1 편차가 존재하였다.

정제 후 데이터셋인 `cls_raw-20260525-v2`를 사용한 실험에서는 validation macro F1이 크게 향상되었고, validation loss와 accuracy도 함께 개선되었다.

| Dataset Version       | Best Val Macro F1 |
| --------------------- | ----------------: |
| `raw-20260509-v1`     |           0.88008 |
| `cls_raw-20260525-v2` |           0.92209 |

성능 차이는 다음과 같다.

```text
0.92209 - 0.88008 = +0.04201
```

이를 통해 깨진 이미지, 중복 이미지, label ambiguity가 큰 이미지가 모델의 일반화 성능을 떨어뜨릴 수 있음을 확인하였다.

### Swin-T Refined Dataset 실험 결과

| Metric   |  Train | Validation |
| -------- | -----: | ---------: |
| Loss     | 0.0201 |     0.2949 |
| Accuracy | 0.9971 |     0.9197 |
| Macro F1 |      - |    0.92209 |

| 항목                | 값                     |
| ----------------- | --------------------- |
| Best Epoch        | 43                    |
| Best Val Macro F1 | 0.92209               |
| 최종 데이터셋           | `cls_raw-20260525-v2` |
| 최종 Backbone       | Swin-T                |

### 결과 해석

* Swin-T는 최종 이미지 분류 모델로 가장 안정적인 성능을 보였다.
* 데이터셋 정제 후 validation macro F1이 크게 향상되었다.
* validation loss가 감소하고 validation accuracy가 상승하였다.
* 데이터 품질 개선이 모델의 일반화 성능 향상에 직접적으로 기여하였다.
* 이후 모델 개선에서는 단순히 모델 구조를 변경하기보다 데이터 품질, 클래스별 오분류, augmentation 효과를 함께 확인해야 한다.

---

## 6. 실행 방법

## 6-1. 프로젝트 클론

```bash
git clone https://github.com/Mini-imagenet-project/imagenet-project.git
cd imagenet-project
```

## 6-2. Docker 환경 실행

이미지
Name : j1seon/supercoding
Tag : v4
size : 60.11 GB

## 6-3. Python 패키지 설치

Docker를 사용하지 않는 경우 다음 명령어로 필요한 패키지를 설치한다.

```bash
pip install -r docker/requirements.txt
```

## 6-4. DVC 데이터 다운로드

DVC로 관리되는 데이터셋과 결과 파일을 내려받는다.

```bash
dvc pull
```

특정 데이터만 내려받고 싶은 경우 다음과 같이 실행할 수 있다.

```bash
dvc pull data/raw.dvc
```

```bash
dvc pull data/captioning.dvc
```

## 6-5. 이미지 분류 모델 학습

```bash
python scripts/train_classification.py
```

DVC 파이프라인에 정의된 stage를 기준으로 실행하는 경우 다음 명령어를 사용할 수 있다.

```bash
dvc repro train_classification
```

설정값을 바꿔서 실행할 경우 params.yaml 파일에서 설정값을 바꾼 후 실행한다.

## 6-6. 이미지 분류 모델 평가

```bash
python scripts/evaluate_classification.py
```

평가 결과는 기본적으로 다음 경로에 저장된다.

```text
outputs/classification/
├── classification_report.json
├── confusion_matrix.png
├── predictions.csv
├── correct_examples/
└── incorrect_examples/
```

## 6-7. Latent Space 분석

```bash
python scripts/analyze_latent_space.py
```

분석 결과는 다음 경로에 저장된다.

```text
outputs/latent_space/
```

## 6-8. Gradio 데모 실행

```bash
python scripts/gradio_demo.py
```

실행 후 브라우저에서 제공되는 Gradio URL로 접속하여 이미지를 업로드하면 분류 예측 결과를 확인할 수 있다.

## 6-9. 이미지 캡셔닝 모델 학습

```bash
python scripts/train_captioning.py
```

캡셔닝 학습에는 이미지 데이터와 caption annotation JSON 파일이 필요하다.

---

## 7. 실험 관리 도구

## DVC

본 프로젝트에서는 데이터셋과 실험 파이프라인을 관리하기 위해 DVC를 사용하였다.

DVC의 주요 사용 목적은 다음과 같다.

* 대용량 데이터셋 버전 관리
* 데이터 변경 이력 추적
* 실험 파이프라인 재현
* 학습 / 평가 / 분석 stage 관리
* 팀원 간 동일한 데이터셋 공유

주요 DVC 파일은 다음과 같다.

| 파일            | 역할                                   |
| ------------- | ------------------------------------ |
| `dvc.yaml`    | 학습, 평가, 분석 stage 정의                  |
| `dvc.lock`    | 실행된 stage의 dependency 및 output 상태 기록 |
| `*.dvc`       | DVC로 관리되는 데이터 또는 출력 추적               |
| `params.yaml` | 실험 파라미터 및 하이퍼파라미터 관리                 |

기본 실행 명령어는 다음과 같다.

```bash
dvc pull
```

```bash
dvc repro
```

```bash
dvc status
```

## Weights & Biases

Weights & Biases는 실험 로그와 결과를 추적하기 위해 사용하였다.

W&B의 주요 사용 목적은 다음과 같다.

* train loss 기록
* train accuracy 기록
* validation accuracy 기록
* validation macro F1 기록
* learning rate 기록
* 실험별 hyperparameter 기록
* best checkpoint artifact 관리
* confusion matrix, latent space 결과 등 시각화 결과 기록

W&B를 통해 여러 실험을 비교하면서 최종 backbone과 학습 조건을 선정하였다.

## Git / GitHub

Git과 GitHub는 코드 버전 관리를 위해 사용하였다.

주요 사용 목적은 다음과 같다.

* 기능 단위 branch 관리
* 코드 변경 이력 추적
* issue 기반 작업 관리
* pull request 기반 코드 병합
* 팀원 간 협업 관리

