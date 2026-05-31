import csv
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append("/workspace/src")

import umap
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
import numpy as np
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader
from torchvision import models

from dataset.classification_dataset import ClassificationDataset
from transforms.image_transform import get_classification_valid_transform


# params
with open("/workspace/params.yaml", "r", encoding="utf-8") as f:
    params = yaml.safe_load(f)



# 분석용 Swin-T 모델 정의
class AnalysisSwinTiny(nn.Module):
    def __init__(self, num_classes=50, embed_size=512):
        super().__init__()

        model = models.swin_t(weights=None)
        in_features = model.head.in_features
        model.head = nn.Identity()

        self.backbone = model
        self.classifier = nn.Linear(in_features, num_classes)
        self.cap_backbone = model.features
        self.projector = nn.Linear(in_features, embed_size)


    # classifier 입력으로 들어가는 전역 feature를 추출
    def extract_classification_features(self, images):
        features = self.backbone(images)
        features = features.view(features.size(0), -1)
        return features


    # 입력 이미지에 대해 분류 logits 반환
    def forward(self, images):
        features = self.extract_classification_features(images)
        return self.classifier(features)
    


# 폴더명을 클래스 목록으로 사용
def get_classes(data_dir):
    return sorted(
        class_name
        for class_name in os.listdir(data_dir)
        if os.path.isdir(os.path.join(data_dir, class_name))
    )

# split 설정에 맞는 classification dataset을 생성
# split이 all이면 train, val, test를 모두 합쳐 latent 분석 대상으로 사용
def build_dataset(data_dir, class_to_idx, split, transform):
    if split != "all":
        return ClassificationDataset(
            root_dir=data_dir,
            class_to_idx=class_to_idx,
            split=split,
            transform=transform
        )

    datasets = [
        ClassificationDataset(
            root_dir=data_dir,
            class_to_idx=class_to_idx,
            split=part,
            transform=transform
        )
        for part in ["train", "val", "test"]
    ]

    dataset = datasets[0]
    dataset.samples = [
        sample
        for part_dataset in datasets
        for sample in part_dataset.samples
    ]

    return dataset


# 분석용 Swin-T 모델을 생성하고 학습된 checkpoint weight를 Load
def load_model(checkpoint_path, num_classes, device):
    model = AnalysisSwinTiny(num_classes=num_classes).to(device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


# classifier 직전 latent feature, 실제 label, 예측 label, 이미지 경로를 추출
def extract_features(model, loader, samples, device):
    features = [] # 추출된 특징 벡터 list
    labels = [] # 이미지의 실제 정답(index) list
    preds = [] # 모델이 예측한 정답 list
    paths = [] # 각 이미지의 파일 경로 list
    offset = 0 # 파일 경로를 매칭할 때 쓸 현재 데이터의 위치 표시용

    with torch.no_grad():
        for images, batch_labels in loader:
            batch_size = images.size(0)
            images = images.to(device)

            batch_features = model.extract_classification_features(images)
            logits = model.classifier(batch_features)

            features.append(batch_features.cpu().numpy())
            labels.extend(batch_labels.numpy().tolist())
            preds.extend(logits.argmax(dim=1).cpu().numpy().tolist())
            paths.extend(
                path
                for path, _ in samples[offset:offset + batch_size]
            )
            offset += batch_size

    return (
        np.concatenate(features, axis=0),
        np.array(labels),
        np.array(preds),
        paths
    )


# 이미지별 경로, 실제 클래스, 예측 클래스, 정답 여부를 csv로 저장
def save_metadata(output_dir, paths, labels, preds, classes):
    metadata_path = output_dir / f'{params["latent_space"]["output_meta_csv"]}.csv'

    with metadata_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "image_path",
                "label_idx",
                "label_name",
                "pred_idx",
                "pred_name",
                "correct"
            ]
        )

        for path, label, pred in zip(paths, labels, preds):
            writer.writerow(
                [
                    path,
                    int(label),
                    classes[int(label)],
                    int(pred),
                    classes[int(pred)],
                    int(label == pred)
                ]
            )


# UMAP으로 축소한 2D 좌표를 클래스 label 색상 기준 scatter plot으로 저장
def save_scatter(points, labels, classes, output_path, title):
    # 커스텀 컬러맵 제작
    num_classes = len(classes)

    base_cmap = plt.colormaps["turbo"]
    class_colors = base_cmap(
        np.linspace(0, 1, num_classes)
    )
    cmap = ListedColormap(class_colors)

    # 색상 경계선 구분
    norm = BoundaryNorm(
        boundaries=np.arange(num_classes + 1) - 0.5,
        ncolors=num_classes
    )

    # 도화지 세팅 및 산점도 입력
    plt.figure(figsize=(14, 10))

    scatter = plt.scatter(
        points[:, 0],
        points[:, 1],
        c=labels,
        cmap=cmap,
        norm=norm,
        s=14,
        alpha=0.75
    )

    plt.title(title)
    plt.xlabel("component 1")
    plt.ylabel("component 2")

    # 우측 색상 막대
    cbar = plt.colorbar(
        scatter,
        ticks=np.arange(num_classes)
    )
    cbar.ax.set_yticklabels(classes)

    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


# 추출된 latent feature를 UMAP으로 2차원 좌표로 축소
def run_umap(features, params):
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=params["latent_space"]["umap"]["n_neighbors"],
        min_dist=params["latent_space"]["umap"]["min_dist"],
        metric=params["latent_space"]["umap"]["metric"],
        random_state=params["latent_space"]["seed"]
    )

    return reducer.fit_transform(features)


def main():
    # device
    device = torch.device(
        params["latent_space"]["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    # 1. 출력 위치 지정
    output_dir = Path(params["latent_space"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    # 2. 클래스 추출
    classes = get_classes(params["latent_space"]["data_dir"])
    class_to_idx = {
        class_name: idx
        for idx, class_name in enumerate(classes)
    }

    # 3. 전처리 데이터
    transform = get_classification_valid_transform()
    dataset = build_dataset(
        params["latent_space"]["data_dir"],
        class_to_idx,
        params["latent_space"]["split"],
        transform
    )

    # 4. 데이터 Load
    loader = DataLoader(
        dataset,
        batch_size=params["latent_space"]["batch_size"],
        shuffle=False,
        num_workers=params["latent_space"]["num_workers"],
        pin_memory= device == "cuda"
    )

    # 5. 모델 선언(swin-t)
    model = load_model(
        params["latent_space"]["checkpoint"],
        num_classes=len(classes),
        device=device
    )

    # 6. 이미지 특징 추출
    features, labels, preds, paths = extract_features(
        model,
        loader,
        dataset.samples,
        device
    )

    # 7. 분석용 meta 데이터 저장
    # np.save(output_dir / "features.npy", features)
    # np.save(output_dir / "labels.npy", labels)
    # np.save(output_dir / "preds.npy", preds)
    if params["latent_space"]["save_meta"]:
        save_metadata(output_dir, paths, labels, preds, classes)

    # 8. umap 실행 및 저장
    umap_points = run_umap(features, params)
    np.save(output_dir / f'{params["latent_space"]["output_umap_npy"]}.npy', umap_points)
    save_scatter(
        umap_points,
        labels,
        classes,
        output_dir / f'{params["latent_space"]["output_umap_png"]}.png',
        "Swin-T Latent Space UMAP"
    )

    print(f"saved latent outputs to: {output_dir}")
    print(f'data_dir: {params["latent_space"]["data_dir"]}')
    print(f'checkpoint: {params["latent_space"]["checkpoint"]}')
    print(f'split: {params["latent_space"]["split"]}')
    print(f'samples: {len(labels)}')
    print(f'feature_dim: {features.shape[1]}')
    print(
        "umap: "
        f'n_neighbors={params["latent_space"]["umap"]["n_neighbors"]}, '
        f'min_dist={params["latent_space"]["umap"]["min_dist"]}, '
        f'metric={params["latent_space"]["umap"]["metric"]}'
    )


if __name__ == "__main__":
    main()
