import json
import random
from pathlib import Path
from collections import defaultdict

import torch
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm
from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer


# =========================================================
# 1. 설정값
# =========================================================

# 전체 클래스 캡셔닝:
# INPUT_IMAGE_DIR = "/workspace/data/raw"
#
# 특정 클래스만 캡셔닝:
# INPUT_IMAGE_DIR = "/workspace/data/raw/apple"
INPUT_IMAGE_DIR = "/workspace/data/raw/airplane"

OUTPUT_JSON_PATH = "/workspace/data/annotations/annotation.json"

MODEL_NAME = "nlpconnect/vit-gpt2-image-captioning"

CAPTIONS_PER_IMAGE = 3

SPLIT_RATIO = {
    "train": 0.7,
    "val": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42

BATCH_SIZE = 8

IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

# "auto": data/raw 입력 시 전체 클래스, data/raw/apple 입력 시 apple 클래스만 자동 판단
# "raw": INPUT_IMAGE_DIR 아래를 전체 raw 폴더로 간주
# "class": INPUT_IMAGE_DIR 자체를 하나의 클래스 폴더로 간주
INPUT_MODE = "auto"

# 캡션 문장 끝의 마침표 제거 여부
REMOVE_TRAILING_PERIOD = True

# beam search 설정
GENERATION_CONFIG = {
    "max_new_tokens": 32,
    "num_beams": 8,
    "num_return_sequences": CAPTIONS_PER_IMAGE,
    "early_stopping": True,
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.1,
    "length_penalty": 0.8,
}

# beam search 결과가 중복될 때 샘플링으로 보충
ENABLE_SAMPLING_FALLBACK = True

SAMPLING_FALLBACK_CONFIG = {
    "max_new_tokens": 32,
    "do_sample": True,
    "top_p": 0.9,
    "temperature": 0.8,
    "num_return_sequences": CAPTIONS_PER_IMAGE * 2,
    "no_repeat_ngram_size": 2,
    "repetition_penalty": 1.1,
}

MAX_FALLBACK_ROUNDS = 3

# 그래도 3개를 못 채우면 중복을 허용해서라도 3개를 맞출지 여부
FILL_WITH_DUPLICATES_IF_NEEDED = True


# =========================================================
# 2. 기본 유틸 함수
# =========================================================

def validate_config():
    total_ratio = sum(SPLIT_RATIO.values())

    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError(f"SPLIT_RATIO의 합은 1이어야 합니다. 현재 합: {total_ratio}")

    if GENERATION_CONFIG["num_beams"] < CAPTIONS_PER_IMAGE:
        raise ValueError("num_beams는 CAPTIONS_PER_IMAGE보다 크거나 같아야 합니다.")

    if GENERATION_CONFIG["num_return_sequences"] != CAPTIONS_PER_IMAGE:
        raise ValueError("GENERATION_CONFIG의 num_return_sequences는 CAPTIONS_PER_IMAGE와 같아야 합니다.")


def is_image_file(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def clean_caption(text: str) -> str:
    caption = " ".join(text.strip().split())

    if REMOVE_TRAILING_PERIOD:
        caption = caption.rstrip(".")

    return caption


def unique_captions(captions):
    result = []
    seen = set()

    for caption in captions:
        caption = clean_caption(caption)
        key = caption.lower()

        if caption and key not in seen:
            result.append(caption)
            seen.add(key)

    return result


def load_image(image_path: Path):
    try:
        return Image.open(image_path).convert("RGB")
    except (UnidentifiedImageError, OSError) as e:
        print(f"[SKIP] 이미지를 열 수 없습니다: {image_path} / error: {e}")
        return None


# =========================================================
# 3. 이미지 목록 수집
# =========================================================

def has_direct_images(input_dir: Path) -> bool:
    for child in input_dir.iterdir():
        if child.is_file() and is_image_file(child):
            return True
    return False


def get_relative_base_dir(input_dir: Path) -> Path:
    """
    JSON의 image 값을 '클래스폴더/이미지명' 형태로 만들기 위한 기준 경로를 정한다.

    예시 1)
    INPUT_IMAGE_DIR = /workspace/data/raw
    image file      = /workspace/data/raw/pizza/hf_pizza_001.jpg
    relative base   = /workspace/data/raw
    result          = pizza/hf_pizza_001.jpg

    예시 2)
    INPUT_IMAGE_DIR = /workspace/data/raw/apple
    image file      = /workspace/data/raw/apple/hf_apple_001.jpg
    relative base   = /workspace/data/raw
    result          = apple/hf_apple_001.jpg
    """

    if INPUT_MODE == "raw":
        return input_dir

    if INPUT_MODE == "class":
        return input_dir.parent

    if INPUT_MODE == "auto":
        if has_direct_images(input_dir):
            return input_dir.parent
        return input_dir

    raise ValueError("INPUT_MODE은 'auto', 'raw', 'class' 중 하나여야 합니다.")


def collect_image_records(input_dir: str):
    input_path = Path(input_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"이미지 경로가 존재하지 않습니다: {input_path}")

    relative_base_dir = get_relative_base_dir(input_path)

    records = []

    for image_path in sorted(input_path.rglob("*")):
        if not image_path.is_file():
            continue

        if not is_image_file(image_path):
            continue

        relative_path = image_path.relative_to(relative_base_dir)
        relative_path_str = relative_path.as_posix()

        # image 값이 apple/xxx.jpg 라면 class는 apple
        class_name = relative_path.parts[0]

        records.append({
            "path": image_path,
            "image": relative_path_str,
            "class": class_name,
        })

    if not records:
        raise ValueError(f"캡셔닝할 이미지가 없습니다: {input_path}")

    return records


# =========================================================
# 4. train / val / test split 배정
# =========================================================

def assign_split(records):
    random.seed(RANDOM_SEED)

    class_map = defaultdict(list)

    for record in records:
        class_map[record["class"]].append(record)

    result = []

    for class_name, items in class_map.items():
        random.shuffle(items)

        total = len(items)
        train_count = int(total * SPLIT_RATIO["train"])
        val_count = int(total * SPLIT_RATIO["val"])

        for idx, item in enumerate(items):
            if idx < train_count:
                item["split"] = "train"
            elif idx < train_count + val_count:
                item["split"] = "val"
            else:
                item["split"] = "test"

            result.append(item)

    result.sort(key=lambda x: x["image"])

    return result


# =========================================================
# 5. 모델 로드
# =========================================================

def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"[INFO] device: {device}")
    print(f"[INFO] model: {MODEL_NAME}")

    model = VisionEncoderDecoderModel.from_pretrained(MODEL_NAME)
    processor = ViTImageProcessor.from_pretrained(MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model.config.pad_token_id = tokenizer.pad_token_id
    model.to(device)
    model.eval()

    return model, processor, tokenizer, device


# =========================================================
# 6. 캡션 생성
# =========================================================

def decode_output_ids(output_ids, tokenizer):
    captions = tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return [clean_caption(caption) for caption in captions]


@torch.no_grad()
def generate_by_beam_search(images, model, processor, tokenizer, device):
    pixel_values = processor(
        images=images,
        return_tensors="pt"
    ).pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        **GENERATION_CONFIG
    )

    captions = decode_output_ids(output_ids, tokenizer)

    grouped = []
    start = 0

    for _ in images:
        end = start + CAPTIONS_PER_IMAGE
        grouped.append(captions[start:end])
        start = end

    return grouped


@torch.no_grad()
def generate_by_sampling(image, model, processor, tokenizer, device):
    pixel_values = processor(
        images=[image],
        return_tensors="pt"
    ).pixel_values.to(device)

    output_ids = model.generate(
        pixel_values,
        **SAMPLING_FALLBACK_CONFIG
    )

    return decode_output_ids(output_ids, tokenizer)


def complete_caption_count(captions, original_candidates):
    """
    기본 목표:
    - 최대한 중복 없는 캡션 3개를 만든다.

    단, 모델이 비슷한 문장만 계속 만들면 3개를 못 채울 수 있다.
    이때 FILL_WITH_DUPLICATES_IF_NEEDED=True이면 중복을 허용해서 3개를 맞춘다.
    """

    captions = unique_captions(captions)

    if len(captions) >= CAPTIONS_PER_IMAGE:
        return captions[:CAPTIONS_PER_IMAGE]

    if not FILL_WITH_DUPLICATES_IF_NEEDED:
        return captions

    for caption in original_candidates:
        caption = clean_caption(caption)

        if caption:
            captions.append(caption)

        if len(captions) >= CAPTIONS_PER_IMAGE:
            break

    return captions[:CAPTIONS_PER_IMAGE]


def generate_captions_for_batch(batch_records, model, processor, tokenizer, device):
    images = []
    valid_records = []

    for record in batch_records:
        image = load_image(record["path"])

        if image is None:
            continue

        images.append(image)
        valid_records.append(record)

    if not images:
        return []

    beam_caption_groups = generate_by_beam_search(
        images=images,
        model=model,
        processor=processor,
        tokenizer=tokenizer,
        device=device
    )

    results = []

    for record, image, beam_captions in zip(valid_records, images, beam_caption_groups):
        all_candidates = list(beam_captions)
        captions = unique_captions(beam_captions)

        if ENABLE_SAMPLING_FALLBACK:
            fallback_round = 0

            while len(captions) < CAPTIONS_PER_IMAGE and fallback_round < MAX_FALLBACK_ROUNDS:
                sampled_captions = generate_by_sampling(
                    image=image,
                    model=model,
                    processor=processor,
                    tokenizer=tokenizer,
                    device=device
                )

                all_candidates.extend(sampled_captions)
                captions = unique_captions(captions + sampled_captions)
                fallback_round += 1

        captions = complete_caption_count(
            captions=captions,
            original_candidates=all_candidates
        )

        results.append({
            "image": record["image"],
            "class": record["class"],
            "captions": captions,
            "split": record["split"],
        })

    return results


# =========================================================
# 7. JSON 저장
# =========================================================

def save_json(data, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=4)

    print(f"[DONE] JSON 저장 완료: {output_path}")
    print(f"[DONE] 총 이미지 수: {len(data)}")


# =========================================================
# 8. 실행
# =========================================================

def main():
    validate_config()

    records = collect_image_records(INPUT_IMAGE_DIR)
    records = assign_split(records)

    print(f"[INFO] 캡셔닝 대상 이미지 수: {len(records)}")

    model, processor, tokenizer, device = load_model()

    results = []

    for start in tqdm(range(0, len(records), BATCH_SIZE), desc="captioning"):
        end = start + BATCH_SIZE
        batch_records = records[start:end]

        batch_results = generate_captions_for_batch(
            batch_records=batch_records,
            model=model,
            processor=processor,
            tokenizer=tokenizer,
            device=device
        )

        results.extend(batch_results)

    save_json(results, OUTPUT_JSON_PATH)


if __name__ == "__main__":
    main()