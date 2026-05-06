import json
import math
import random
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import os
import torch
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoProcessor


# ============================================================
# 1. 설정값
# ============================================================

load_dotenv()

# .env 안의 HF_TOKEN 읽기
hf_token = os.getenv("HF_TOKEN")

# 이미지 원본 루트 경로
# 예:
# - 전체 클래스 캡셔닝: "data/raw"
# - 특정 클래스만 캡셔닝: "data/raw/apple"
INPUT_IMAGE_PATH = "data/raw/airplane"

# image 필드를 만들 때 기준이 되는 root
# JSON에는 "pizza/hf_pizza_001.jpg" 형태로 저장됨
DATA_RAW_ROOT = "data/raw"

# 결과 저장 경로
OUTPUT_JSON_PATH = "data/annotations/captions_git.json"

# 에러 이미지 목록 저장 경로
ERROR_JSON_PATH = "data/annotations/caption_git_errors.json"

# GIT 모델
# 기본 추천: microsoft/git-base-coco
# 더 큰 모델을 쓰고 싶으면: microsoft/git-large-coco
MODEL_NAME = "microsoft/git-large-coco"

# 이미지당 생성할 캡션 개수
CAPTIONS_PER_IMAGE = 3

# split 비율
# 기본 7 : 1.5 : 1.5
SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

# split 재현성을 위한 seed
RANDOM_SEED = 42

# 추론 배치 크기
# GPU 메모리가 부족하면 8 -> 4 -> 2 -> 1 순서로 줄이기
BATCH_SIZE = 8

# 장치 설정
# "auto": CUDA 가능하면 GPU, 아니면 CPU
# 직접 지정 가능: "cuda", "cpu"
DEVICE = "auto"

# dtype 설정
# "auto": CUDA면 float16, CPU면 float32
# 직접 지정 가능: "float32", "float16", "bfloat16"
TORCH_DTYPE = "auto"

# 중간 저장 간격
# 이미지가 많을 때 중간에 오류가 나도 일부 결과를 보존하기 위한 설정
SAVE_EVERY_N_IMAGES = 100

# 기존 OUTPUT_JSON_PATH가 있으면 이미 캡셔닝된 이미지는 건너뛸지 여부
RESUME_FROM_EXISTING_JSON = True

# 지원 이미지 확장자
SUPPORTED_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".webp", ".bmp"
}

# 캡션 생성 설정
# num_beams >= num_return_sequences 여야 함
# num_beam_groups를 사용해서 서로 조금 다른 caption을 생성하도록 함
GENERATION_CONFIG = {
    "max_length": 40,
    "num_beams": 5,
    "num_return_sequences": CAPTIONS_PER_IMAGE,
    "early_stopping": True,
    "no_repeat_ngram_size": 2,
}

# beam search 결과가 너무 중복될 때 추가 샘플링으로 보완할지 여부
ENABLE_SAMPLING_FALLBACK = True

SAMPLING_FALLBACK_CONFIG = {
    "max_length": 40,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "num_return_sequences": CAPTIONS_PER_IMAGE,
    "no_repeat_ngram_size": 2,
}


# ============================================================
# 2. 데이터 구조
# ============================================================

@dataclass
class ImageItem:
    path: Path
    image_field: str
    class_name: str
    split: str = ""


# ============================================================
# 3. 유틸 함수
# ============================================================

def resolve_device() -> torch.device:
    if DEVICE == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(DEVICE)


def resolve_dtype(device: torch.device) -> torch.dtype:
    if TORCH_DTYPE == "auto":
        return torch.float16 if device.type == "cuda" else torch.float32

    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }

    if TORCH_DTYPE not in dtype_map:
        raise ValueError(f"지원하지 않는 TORCH_DTYPE입니다: {TORCH_DTYPE}")

    if device.type == "cpu" and TORCH_DTYPE in {"float16", "bfloat16"}:
        print("[WARN] CPU에서는 float16/bfloat16이 불안정할 수 있어 float32로 변경합니다.")
        return torch.float32

    return dtype_map[TORCH_DTYPE]


def normalize_caption(text: str) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    return text


def deduplicate_captions(captions: List[str]) -> List[str]:
    result = []
    seen = set()

    for caption in captions:
        caption = normalize_caption(caption)

        if not caption:
            continue

        key = caption.lower()

        if key in seen:
            continue

        seen.add(key)
        result.append(caption)

    return result


def ensure_caption_count(captions: List[str], target_count: int) -> List[str]:
    captions = deduplicate_captions(captions)

    if len(captions) >= target_count:
        return captions[:target_count]

    if len(captions) == 0:
        return [""] * target_count

    while len(captions) < target_count:
        captions.append(captions[-1])

    return captions


def save_json(path: Path, data: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_json(path: Path) -> Dict[str, dict]:
    if not path.exists():
        return {}

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}

    for item in data:
        image_key = item.get("image")

        if image_key:
            result[image_key] = item

    return result


# ============================================================
# 4. 이미지 수집
# ============================================================

def collect_images(input_path: Path, data_raw_root: Path) -> List[ImageItem]:
    if not input_path.exists():
        raise FileNotFoundError(f"입력 경로가 존재하지 않습니다: {input_path}")

    image_paths = sorted([
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    ])

    if not image_paths:
        raise RuntimeError(f"이미지를 찾지 못했습니다: {input_path}")

    items = []

    for image_path in image_paths:
        try:
            relative_path = image_path.relative_to(data_raw_root)
        except ValueError:
            raise ValueError(
                f"이미지 경로가 DATA_RAW_ROOT 하위에 있어야 합니다.\n"
                f"image_path={image_path}\n"
                f"DATA_RAW_ROOT={data_raw_root}"
            )

        if len(relative_path.parts) < 2:
            raise ValueError(
                f"이미지는 클래스 폴더 하위에 있어야 합니다: {image_path}\n"
                f"예: data/raw/pizza/hf_pizza_001.jpg"
            )

        class_name = relative_path.parts[0]
        image_field = relative_path.as_posix()

        items.append(
            ImageItem(
                path=image_path,
                image_field=image_field,
                class_name=class_name,
            )
        )

    return items


# ============================================================
# 5. split 분리
# ============================================================

def calculate_split_counts(total_count: int) -> Dict[str, int]:
    ratio_sum = sum(SPLIT_RATIO.values())

    raw_counts = {
        split_name: total_count * ratio / ratio_sum
        for split_name, ratio in SPLIT_RATIO.items()
    }

    counts = {
        split_name: int(math.floor(count))
        for split_name, count in raw_counts.items()
    }

    remaining = total_count - sum(counts.values())

    # 소수점이 큰 split부터 남은 개수 배분
    sorted_splits = sorted(
        raw_counts.keys(),
        key=lambda split_name: raw_counts[split_name] - counts[split_name],
        reverse=True,
    )

    for split_name in sorted_splits[:remaining]:
        counts[split_name] += 1

    return counts


def assign_splits(items: List[ImageItem]) -> List[ImageItem]:
    rng = random.Random(RANDOM_SEED)

    class_map = defaultdict(list)

    for item in items:
        class_map[item.class_name].append(item)

    for class_name, class_items in class_map.items():
        rng.shuffle(class_items)

        counts = calculate_split_counts(len(class_items))

        start = 0

        for split_name in ["train", "val", "test"]:
            end = start + counts.get(split_name, 0)

            for item in class_items[start:end]:
                item.split = split_name

            start = end

    return items


# ============================================================
# 6. 모델 로드
# ============================================================

def load_model():
    device = resolve_device()
    torch_dtype = resolve_dtype(device)

    print(f"[INFO] device={device}")
    print(f"[INFO] dtype={torch_dtype}")
    print(f"[INFO] model={MODEL_NAME}")

    processor = AutoProcessor.from_pretrained(MODEL_NAME, token=hf_token)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch_dtype,
        token=hf_token
    )
    model.to(device)
    model.eval()

    return model, processor, device, torch_dtype


# ============================================================
# 7. 캡셔닝
# ============================================================

def load_batch_images(batch_items: List[ImageItem]) -> Tuple[List[Image.Image], List[ImageItem], List[dict]]:
    images = []
    valid_items = []
    errors = []

    for item in batch_items:
        try:
            with Image.open(item.path) as img:
                images.append(img.convert("RGB"))
                valid_items.append(item)
        except Exception as e:
            errors.append({
                "image": item.image_field,
                "class": item.class_name,
                "split": item.split,
                "error": str(e),
            })

    return images, valid_items, errors


@torch.inference_mode()
def generate_batch_captions(
    model,
    processor,
    device: torch.device,
    torch_dtype: torch.dtype,
    images: List[Image.Image],
) -> List[List[str]]:
    inputs = processor(images=images, return_tensors="pt")

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch_dtype)

    generated_ids = model.generate(
        **inputs,
        **GENERATION_CONFIG,
    )

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    grouped_captions = []

    for i in range(len(images)):
        start = i * CAPTIONS_PER_IMAGE
        end = start + CAPTIONS_PER_IMAGE

        captions = decoded[start:end]
        captions = deduplicate_captions(captions)

        grouped_captions.append(captions)

    return grouped_captions


@torch.inference_mode()
def generate_sampling_fallback_captions(
    model,
    processor,
    device: torch.device,
    torch_dtype: torch.dtype,
    image: Image.Image,
) -> List[str]:
    inputs = processor(images=[image], return_tensors="pt")

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    if "pixel_values" in inputs:
        inputs["pixel_values"] = inputs["pixel_values"].to(dtype=torch_dtype)

    generated_ids = model.generate(
        **inputs,
        **SAMPLING_FALLBACK_CONFIG,
    )

    decoded = processor.batch_decode(
        generated_ids,
        skip_special_tokens=True,
    )

    return deduplicate_captions(decoded)


def make_result_item(item: ImageItem, captions: List[str]) -> dict:
    return {
        "image": item.image_field,
        "class": item.class_name,
        "captions": captions,
        "split": item.split,
    }


def caption_images(
    model,
    processor,
    device: torch.device,
    torch_dtype: torch.dtype,
    items: List[ImageItem],
    existing_result_map: Dict[str, dict],
) -> Tuple[Dict[str, dict], List[dict]]:
    result_map = dict(existing_result_map)
    error_list = []

    target_items = [
        item
        for item in items
        if item.image_field not in result_map
    ]

    print(f"[INFO] 전체 이미지 수: {len(items)}")
    print(f"[INFO] 기존 결과 수: {len(existing_result_map)}")
    print(f"[INFO] 새로 캡셔닝할 이미지 수: {len(target_items)}")

    processed_count = 0

    for batch_start in tqdm(range(0, len(target_items), BATCH_SIZE), desc="Captioning"):
        batch_items = target_items[batch_start:batch_start + BATCH_SIZE]

        images, valid_items, errors = load_batch_images(batch_items)
        error_list.extend(errors)

        if not images:
            continue

        try:
            batch_captions = generate_batch_captions(
                model=model,
                processor=processor,
                device=device,
                torch_dtype=torch_dtype,
                images=images,
            )

            for image, item, captions in zip(images, valid_items, batch_captions):
                if ENABLE_SAMPLING_FALLBACK and len(captions) < CAPTIONS_PER_IMAGE:
                    fallback_captions = generate_sampling_fallback_captions(
                        model=model,
                        processor=processor,
                        device=device,
                        torch_dtype=torch_dtype,
                        image=image,
                    )

                    captions = deduplicate_captions(captions + fallback_captions)

                captions = ensure_caption_count(
                    captions=captions,
                    target_count=CAPTIONS_PER_IMAGE,
                )

                result_map[item.image_field] = make_result_item(
                    item=item,
                    captions=captions,
                )

                processed_count += 1

        except Exception as e:
            print("[ERROR] 배치 캡셔닝 실패")
            print(f"[ERROR] {type(e).__name__}: {e}")

            for item in valid_items:
                error_list.append({
                    "image": item.image_field,
                    "class": item.class_name,
                    "split": item.split,
                    "error": str(e),
                })

        if SAVE_EVERY_N_IMAGES > 0 and processed_count > 0:
            if processed_count % SAVE_EVERY_N_IMAGES == 0:
                current_results = [
                    result_map[item.image_field]
                    for item in items
                    if item.image_field in result_map
                ]
                save_json(Path(OUTPUT_JSON_PATH), current_results)
                save_json(Path(ERROR_JSON_PATH), error_list)

    return result_map, error_list


# ============================================================
# 8. main
# ============================================================

def main():
    input_path = Path(INPUT_IMAGE_PATH).resolve()
    data_raw_root = Path(DATA_RAW_ROOT).resolve()
    output_json_path = Path(OUTPUT_JSON_PATH)
    error_json_path = Path(ERROR_JSON_PATH)

    items = collect_images(
        input_path=input_path,
        data_raw_root=data_raw_root,
    )

    items = assign_splits(items)

    existing_result_map = {}

    if RESUME_FROM_EXISTING_JSON:
        existing_result_map = load_existing_json(output_json_path)

    model, processor, device, torch_dtype = load_model()

    result_map, error_list = caption_images(
        model=model,
        processor=processor,
        device=device,
        torch_dtype=torch_dtype,
        items=items,
        existing_result_map=existing_result_map,
    )

    final_results = [
        result_map[item.image_field]
        for item in items
        if item.image_field in result_map
    ]

    save_json(output_json_path, final_results)
    save_json(error_json_path, error_list)

    print("[DONE] 캡셔닝 완료")
    print(f"[DONE] 결과 저장: {output_json_path}")
    print(f"[DONE] 에러 저장: {error_json_path}")
    print(f"[DONE] 정상 결과 수: {len(final_results)}")
    print(f"[DONE] 에러 수: {len(error_list)}")


if __name__ == "__main__":
    main()