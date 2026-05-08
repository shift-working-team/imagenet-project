import os
import json
import random
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from transformers import AutoProcessor, Florence2ForConditionalGeneration


# =========================================================
# 1. 설정값
# =========================================================

# 전체 클래스 캡셔닝: "data/raw"
# 특정 클래스만 캡셔닝: "data/raw/apple"
INPUT_IMAGE_DIR = "data/raw"

# image 값을 "pizza/hf_pizza_001.jpg" 형태로 만들기 위한 기준 경로
DATA_RAW_ROOT = "data/raw"

# 결과 JSON 저장 경로
OUTPUT_JSON_PATH = "data/annotations/captions_flo_all.json"

# transformers 5.7.0에서는 florence-community 모델 사용 권장
# base-ft: 가볍고 다운스트림 task에 fine-tuning된 모델
# large-ft: 더 무겁지만 품질이 더 좋을 수 있음
MODEL_ID = "florence-community/Florence-2-base-ft"
# MODEL_ID = "florence-community/Florence-2-large-ft"

# .env 파일에서 읽을 Hugging Face 토큰 이름
# 공개 모델이면 없어도 동작할 수 있지만, 토큰을 넣어두는 편이 안정적입니다.
HF_TOKEN_ENV_NAME = "HF_TOKEN"

# split 비율: 기본 7 : 1.5 : 1.5
TRAIN_RATIO = 0.7
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# split 재현을 위한 seed
RANDOM_SEED = 42

# 이미지당 캡션 3개 생성
# Florence-2 문서에서 지원하는 caption task입니다.
CAPTION_TASKS = [
    "<CAPTION>",
    "<DETAILED_CAPTION>",
    "<MORE_DETAILED_CAPTION>",
]

# 생성 옵션
NUM_BEAMS = 3
MAX_NEW_TOKENS = 64

# 몇 장마다 중간 저장할지
SAVE_EVERY = 220

# 이미 JSON에 있는 이미지는 건너뛸지 여부
SKIP_ALREADY_DONE = True

# 허용 이미지 확장자
IMAGE_EXTENSIONS = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]


# =========================================================
# 2. 이미지 목록 가져오기
# =========================================================

def get_image_list():
    input_dir = Path(INPUT_IMAGE_DIR).resolve()
    raw_root = Path(DATA_RAW_ROOT).resolve()

    if not input_dir.exists():
        raise FileNotFoundError(f"입력 경로가 없습니다: {input_dir}")

    image_list = []

    for image_path in sorted(input_dir.rglob("*")):
        if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
            continue

        # 예:
        # /workspace/data/raw/pizza/hf_pizza_001.jpg
        # -> pizza/hf_pizza_001.jpg
        relative_image_path = image_path.resolve().relative_to(raw_root).as_posix()

        # 예:
        # pizza/hf_pizza_001.jpg
        # -> pizza
        class_name = relative_image_path.split("/")[0]

        image_list.append({
            "path": image_path,
            "image": relative_image_path,
            "class": class_name,
        })

    return image_list


# =========================================================
# 3. train / val / test 나누기
# =========================================================

def add_split(image_list):
    random.seed(RANDOM_SEED)

    total_ratio = TRAIN_RATIO + VAL_RATIO + TEST_RATIO

    result = []

    # 클래스별로 이미지 모으기
    class_map = {}

    for item in image_list:
        class_name = item["class"]

        if class_name not in class_map:
            class_map[class_name] = []

        class_map[class_name].append(item)

    # 클래스별로 train / val / test 나누기
    for class_name, items in class_map.items():
        random.shuffle(items)

        total_count = len(items)

        train_count = round(total_count * TRAIN_RATIO / total_ratio)
        val_count = round(total_count * VAL_RATIO / total_ratio)

        for index, item in enumerate(items):
            if index < train_count:
                split = "train"
            elif index < train_count + val_count:
                split = "val"
            else:
                split = "test"

            item["split"] = split
            result.append(item)

    return result


# =========================================================
# 4. Florence-2 모델 준비
# =========================================================

def load_model():
    load_dotenv()

    hf_token = os.getenv(HF_TOKEN_ENV_NAME)

    if torch.cuda.is_available():
        device = "cuda"

        # GPU가 bfloat16을 지원하면 bfloat16 사용
        # 아니면 float16 사용
        if torch.cuda.is_bf16_supported():
            torch_dtype = torch.bfloat16
        else:
            torch_dtype = torch.float16
    else:
        device = "cpu"
        torch_dtype = torch.float32

    print(f"device: {device}")
    print(f"dtype: {torch_dtype}")
    print(f"model: {MODEL_ID}")

    processor = AutoProcessor.from_pretrained(
        MODEL_ID,
        token=hf_token,
    )

    model = Florence2ForConditionalGeneration.from_pretrained(
        MODEL_ID,
        dtype=torch_dtype,
        token=hf_token,
    ).to(device)

    model.eval()

    return model, processor, device, torch_dtype


# =========================================================
# 5. 이미지 1장 캡셔닝
# =========================================================

def make_caption(image, task, model, processor, device, torch_dtype):
    inputs = processor(
        text=task,
        images=image,
        return_tensors="pt",
    )

    inputs = inputs.to(device, torch_dtype)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            num_beams=NUM_BEAMS,
            do_sample=False,
        )

    generated_text = processor.batch_decode(
        generated_ids,
        skip_special_tokens=False,
    )[0]

    parsed_result = processor.post_process_generation(
        generated_text,
        task=task,
        image_size=image.size,
    )

    caption = parsed_result.get(task, "")

    if not isinstance(caption, str):
        caption = str(caption)

    return caption.strip()


def make_three_captions(image_path, model, processor, device, torch_dtype):
    image = Image.open(image_path).convert("RGB")

    captions = []

    for task in CAPTION_TASKS:
        caption = make_caption(
            image=image,
            task=task,
            model=model,
            processor=processor,
            device=device,
            torch_dtype=torch_dtype,
        )

        captions.append(caption)

    return captions


# =========================================================
# 6. 기존 JSON 읽기 / 저장하기
# =========================================================

def load_existing_result():
    output_path = Path(OUTPUT_JSON_PATH)

    if not output_path.exists():
        return {}

    with output_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    result = {}

    for item in data:
        result[item["image"]] = item

    return result


def save_result(result_map):
    output_path = Path(OUTPUT_JSON_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    result_list = list(result_map.values())
    result_list.sort(key=lambda x: x["image"])

    with output_path.open("w", encoding="utf-8") as f:
        json.dump(result_list, f, ensure_ascii=False, indent=4)


# =========================================================
# 7. 실행
# =========================================================

def main():
    print("이미지 목록을 읽는 중입니다.")

    image_list = get_image_list()
    image_list = add_split(image_list)

    print(f"총 이미지 수: {len(image_list)}")

    result_map = load_existing_result()

    model, processor, device, torch_dtype = load_model()

    new_count = 0
    skip_count = 0
    fail_count = 0

    for item in tqdm(image_list):
        image_key = item["image"]

        if SKIP_ALREADY_DONE and image_key in result_map:
            skip_count += 1
            continue

        try:
            captions = make_three_captions(
                image_path=item["path"],
                model=model,
                processor=processor,
                device=device,
                torch_dtype=torch_dtype,
            )

            result_map[image_key] = {
                "image": item["image"],
                "class": item["class"],
                "captions": captions,
                "split": item["split"],
            }

            new_count += 1

            if new_count % SAVE_EVERY == 0:
                save_result(result_map)

        except Exception as e:
            fail_count += 1
            print(f"\n실패한 이미지: {item['path']}")
            print(f"에러 내용: {e}")

    save_result(result_map)

    print("\n캡셔닝 완료")
    print(f"새로 처리한 이미지 수: {new_count}")
    print(f"건너뛴 이미지 수: {skip_count}")
    print(f"실패한 이미지 수: {fail_count}")
    print(f"저장 위치: {OUTPUT_JSON_PATH}")


if __name__ == "__main__":
    main()