import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor
import os
from dotenv import load_dotenv

# ============================================================
# 설정값
# ============================================================

load_dotenv()

# .env 안의 HF_TOKEN 읽기
hf_token = os.getenv("HF_TOKEN")

# 전체 클래스를 검수하려면 True
# 특정 클래스만 검수하려면 False
CHECK_ALL_CLASSES = True

# 전체 클래스 검수 시 기준이 되는 raw 데이터 루트
DATA_RAW_ROOT_DIR = Path("data/raw")

# 특정 클래스만 검수할 때 사용할 클래스 폴더 경로
# CHECK_ALL_CLASSES = False 일 때만 사용됨
TARGET_CLASS_DIR = Path("data/raw")

# 입력 JSON 파일
INPUT_JSON_PATH = Path("data/annotations/captions_flo_all.json")

# 출력 JSON 파일
OUTPUT_JSON_PATH = Path("data/annotations/clip_checked_flo_all.json")

# 사용할 CLIP 모델
MODEL_NAME = "openai/clip-vit-base-patch32"

# 한 번에 처리할 이미지-캡션 쌍 개수
BATCH_SIZE = 32

# 하위 몇 %를 fail / review로 볼지
FAIL_BOTTOM_PERCENT = 10
REVIEW_BOTTOM_PERCENT = 20

print("경로 : " , INPUT_JSON_PATH)

# ============================================================
# JSON 입출력
# ============================================================

def load_json(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, list):
        raise ValueError("입력 JSON은 반드시 배열 형태여야 합니다.")

    return data


def save_json(data: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)


# ============================================================
# 클래스 / 경로 처리
# ============================================================

def get_target_class_name() -> str:
    """
    TARGET_CLASS_DIR = data/raw/airplane 이면 airplane 반환
    """
    return TARGET_CLASS_DIR.name


def get_class_name_from_image_value(image_value: str) -> str:
    """
    JSON의 image 값이 airplane/hf_airplane_001.jpg 라면 airplane 반환
    """
    image_value = image_value.replace("\\", "/")
    image_path = Path(image_value)

    if len(image_path.parts) < 2:
        return ""

    return image_path.parts[0]


def is_target_item(item: dict[str, Any]) -> bool:
    """
    CHECK_ALL_CLASSES = True:
        모든 item 처리

    CHECK_ALL_CLASSES = False:
        TARGET_CLASS_DIR.name과 JSON image의 첫 번째 폴더명이 같은 item만 처리
    """
    if CHECK_ALL_CLASSES:
        return True

    image_value = str(item.get("image", ""))
    image_class_name = get_class_name_from_image_value(image_value)

    return image_class_name == get_target_class_name()


def resolve_image_path(image_value: str) -> Path:
    """
    JSON:
        "image": "airplane/hf_airplane_001.jpg"

    전체 클래스 검수:
        DATA_RAW_ROOT_DIR / image
        → data/raw/airplane/hf_airplane_001.jpg

    특정 클래스 검수:
        TARGET_CLASS_DIR / 파일명
        → data/raw/airplane/hf_airplane_001.jpg
    """
    image_value = image_value.replace("\\", "/")
    image_path = Path(image_value)

    if CHECK_ALL_CLASSES:
        return DATA_RAW_ROOT_DIR / image_path

    return TARGET_CLASS_DIR / image_path.name


def load_image(image_path: Path) -> Image.Image | None:
    try:
        with Image.open(image_path) as img:
            return img.convert("RGB").copy()
    except Exception:
        return None


# ============================================================
# 캡션 펼치기
# ============================================================

def flatten_caption_items(data: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """
    이미지 1장에 caption 3개가 있으면
    이미지-캡션 쌍 3개로 펼친다.
    """
    target_data = []
    flat_items = []

    for item in data:
        if not is_target_item(item):
            continue

        target_item_index = len(target_data)
        target_data.append(item)

        image_value = str(item.get("image", ""))
        captions = item.get("captions", [])

        if not isinstance(captions, list):
            captions = []

        for caption_index, caption in enumerate(captions):
            flat_items.append({
                "item_index": target_item_index,
                "caption_index": caption_index,
                "image": image_value,
                "class": item.get("class", ""),
                "split": item.get("split", ""),
                "caption": str(caption).strip()
            })

    return target_data, flat_items


# ============================================================
# CLIP Score 계산
# ============================================================

@torch.no_grad()
def compute_clip_scores(
    flat_items: list[dict[str, Any]],
    model: CLIPModel,
    processor: CLIPProcessor,
    device: torch.device
) -> list[dict[str, Any]]:

    results = []

    for start in tqdm(range(0, len(flat_items), BATCH_SIZE), desc="computing CLIP scores"):
        batch_items = flat_items[start:start + BATCH_SIZE]

        valid_items = []
        images = []
        texts = []

        for item in batch_items:
            image_path = resolve_image_path(item["image"])
            image = load_image(image_path)

            if image is None:
                results.append({
                    **item,
                    "resolved_image_path": str(image_path).replace("\\", "/"),
                    "clip_cosine": None,
                    "clip_score": None,
                    "clip_status": "missing_image",
                    "clip_reason": f"image file could not be opened: {image_path}"
                })
                continue

            caption = item["caption"]

            if not caption:
                results.append({
                    **item,
                    "resolved_image_path": str(image_path).replace("\\", "/"),
                    "clip_cosine": None,
                    "clip_score": None,
                    "clip_status": "empty_caption",
                    "clip_reason": "caption is empty"
                })
                continue

            valid_items.append({
                **item,
                "resolved_image_path": str(image_path).replace("\\", "/")
            })
            images.append(image)
            texts.append(caption)

        if not valid_items:
            continue

        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        inputs = {
            key: value.to(device)
            for key, value in inputs.items()
        }

        outputs = model(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            pixel_values=inputs["pixel_values"]
        )

        image_features = outputs.image_embeds
        text_features = outputs.text_embeds

        image_features = F.normalize(image_features, p=2, dim=1)
        text_features = F.normalize(text_features, p=2, dim=1)

        cosine_scores = (image_features * text_features).sum(dim=1)

        for item, cosine in zip(valid_items, cosine_scores):
            cosine_value = float(cosine.detach().cpu().item())
            clip_score = 2.5 * max(cosine_value, 0.0)

            results.append({
                **item,
                "clip_cosine": round(cosine_value, 6),
                "clip_score": round(clip_score, 6),
                "clip_status": "pending",
                "clip_reason": ""
            })

    return results


# ============================================================
# pass / review / fail 판정
# ============================================================

def assign_clip_status(results: list[dict[str, Any]]) -> None:
    valid_scores = [
        result["clip_score"]
        for result in results
        if isinstance(result.get("clip_score"), float)
    ]

    if not valid_scores:
        return

    fail_threshold = np.percentile(valid_scores, FAIL_BOTTOM_PERCENT)
    review_threshold = np.percentile(valid_scores, REVIEW_BOTTOM_PERCENT)

    for result in results:
        clip_score = result.get("clip_score")

        if clip_score is None:
            continue

        if clip_score <= fail_threshold:
            result["clip_status"] = "fail"
            result["clip_reason"] = f"clip score is in the bottom {FAIL_BOTTOM_PERCENT}%"
        elif clip_score <= review_threshold:
            result["clip_status"] = "review"
            result["clip_reason"] = f"clip score is in the bottom {REVIEW_BOTTOM_PERCENT}%"
        else:
            result["clip_status"] = "pass"
            result["clip_reason"] = "clip score is acceptable"


# ============================================================
# 결과를 원래 JSON 구조에 붙이기
# ============================================================

def attach_results_to_data(
    target_data: list[dict[str, Any]],
    results: list[dict[str, Any]]
) -> list[dict[str, Any]]:

    for item in target_data:
        item["caption_checks"] = []

    results = sorted(
        results,
        key=lambda x: (x["item_index"], x["caption_index"])
    )

    for result in results:
        item_index = result["item_index"]

        check = {
            "caption_index": result["caption_index"],
            "caption": result["caption"],
            "resolved_image_path": result.get("resolved_image_path"),
            "clip_cosine": result.get("clip_cosine"),
            "clip_score": result.get("clip_score"),
            "clip_status": result.get("clip_status"),
            "clip_reason": result.get("clip_reason", "")
        }

        target_data[item_index]["caption_checks"].append(check)

    return target_data


# ============================================================
# 요약 출력
# ============================================================

def print_summary(
    target_data: list[dict[str, Any]],
    flat_items: list[dict[str, Any]],
    results: list[dict[str, Any]]
) -> None:

    status_count = {}
    valid_scores = []

    for result in results:
        status = result.get("clip_status", "unknown")
        status_count[status] = status_count.get(status, 0) + 1

        if isinstance(result.get("clip_score"), float):
            valid_scores.append(result["clip_score"])

    print("\n===== CLIP Score Summary =====")
    print(f"check all classes: {CHECK_ALL_CLASSES}")

    if CHECK_ALL_CLASSES:
        print(f"data raw root dir: {DATA_RAW_ROOT_DIR}")
    else:
        print(f"target class dir: {TARGET_CLASS_DIR}")
        print(f"target class name: {get_target_class_name()}")

    print(f"target images: {len(target_data)}")
    print(f"target image-caption pairs: {len(flat_items)}")
    print(f"status count: {status_count}")

    if valid_scores:
        print(f"min score: {min(valid_scores):.4f}")
        print(f"max score: {max(valid_scores):.4f}")
        print(f"mean score: {np.mean(valid_scores):.4f}")
        print(f"bottom {FAIL_BOTTOM_PERCENT}% threshold: {np.percentile(valid_scores, FAIL_BOTTOM_PERCENT):.4f}")
        print(f"bottom {REVIEW_BOTTOM_PERCENT}% threshold: {np.percentile(valid_scores, REVIEW_BOTTOM_PERCENT):.4f}")


# ============================================================
# 실행
# ============================================================

def main():
    if not INPUT_JSON_PATH.exists():
        raise FileNotFoundError(f"input file not found: {INPUT_JSON_PATH}")

    if CHECK_ALL_CLASSES:
        if not DATA_RAW_ROOT_DIR.exists():
            raise FileNotFoundError(f"data raw root directory not found: {DATA_RAW_ROOT_DIR}")
    else:
        if not TARGET_CLASS_DIR.exists():
            raise FileNotFoundError(f"target class directory not found: {TARGET_CLASS_DIR}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    print(f"loading model: {MODEL_NAME}")
    model = CLIPModel.from_pretrained(MODEL_NAME, token=hf_token).to(device)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME, token=hf_token)
    model.eval()

    data = load_json(INPUT_JSON_PATH)
    target_data, flat_items = flatten_caption_items(data)

    if not target_data:
        raise ValueError("검수 대상 데이터가 없습니다. CHECK_ALL_CLASSES 또는 TARGET_CLASS_DIR 설정을 확인하세요.")

    results = compute_clip_scores(
        flat_items=flat_items,
        model=model,
        processor=processor,
        device=device
    )

    assign_clip_status(results)

    checked_data = attach_results_to_data(target_data, results)

    save_json(checked_data, OUTPUT_JSON_PATH)

    print_summary(target_data, flat_items, results)
    print(f"\nsaved: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()