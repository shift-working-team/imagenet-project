import os                      # 파일/폴더 탐색
import json                    # JSON 저장
import random                  # 데이터 섞기
import torch                   # GPU 사용
import re                      # 정규식 (문장 필터링)
from collections import defaultdict  # 클래스별 그룹화
from PIL import Image          # 이미지 로드
from transformers import BlipProcessor, BlipForConditionalGeneration  # BLIP
from sentence_transformers import SentenceTransformer, util           # SBERT

# ----------------------
# 1. 설정
# ----------------------
ROOT_DIR = "data/raw"                         # 이미지 루트 폴더 (raw/클래스/이미지)
OUTPUT_JSON = "annotation.json"         # 결과 JSON 파일 이름

TARGET_CAPTIONS = 3                        # 이미지당 캡션 개수 (3 또는 5 추천)
SIM_THRESHOLD = 0.85                       # 문장 유사도 기준 (높을수록 엄격)
MIN_WORDS = 3                              # 최소 단어 수 (짧은 문장 제거)
MAX_ATTEMPTS = 10                          # 캡션 생성 최대 반복 횟수

TRAIN_RATIO = 0.7                          # train 비율
VAL_RATIO = 0.15                           # val 비율
TEST_RATIO = 0.15                          # test 비율

device = "cuda" if torch.cuda.is_available() else "cpu"  # GPU 사용 여부
print("device : ", device)
# ----------------------
# 2. 모델 로드
# ----------------------
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")  
# 이미지 → 토큰 변환

blip_model = BlipForConditionalGeneration.from_pretrained(
    "Salesforce/blip-image-captioning-base"
).to(device)  
# 캡션 생성 모델

embedder = SentenceTransformer("all-MiniLM-L6-v2", device=device)  
# 문장 → 벡터 (유사도 계산용)

# ----------------------
# 3. 캡션 생성 함수
# ----------------------
def generate_captions(image, n):
    inputs = processor(images=image, return_tensors="pt").to(device)  # 이미지 전처리

    outputs = blip_model.generate(
        **inputs,
        do_sample=True,        # 다양성 확보 (샘플링)
        top_k=50,
        top_p=0.95,
        temperature=0.9,
        num_return_sequences=n,  # n개 생성
        max_length=30
    )

    # 토큰 → 문자열 변환
    return [
        processor.decode(o, skip_special_tokens=True).strip().lower()
        for o in outputs
    ]

# ----------------------
# 4. 기본 품질 필터
# ----------------------
def basic_filter(captions):
    filtered = []

    for c in captions:
        words = c.split()

        if len(words) < MIN_WORDS:          # 너무 짧은 문장 제거
            continue

        if len(set(words)) < len(words) * 0.6:  # 반복 단어 많은 문장 제거
            continue

        if re.search(r"[^a-z0-9\s]", c):    # 이상한 문자 제거
            continue

        filtered.append(c)

    return filtered

# ----------------------
# 5. 키워드 추출
# ----------------------
def extract_keywords(caption):
    stopwords = {"a","the","on","in","at","with","and","of","to","is","are"}  # 불용어
    return set([w for w in caption.split() if w not in stopwords])  # 핵심 단어만 추출

# ----------------------
# 6. 유사도 + 키워드 필터
# ----------------------
def advanced_filter(captions):
    if not captions:
        return []

    embeddings = embedder.encode(captions, convert_to_tensor=True)  # 문장 → 벡터

    selected = []
    selected_idx = []

    for i, cap in enumerate(captions):
        keep = True
        kw_i = extract_keywords(cap)

        for j in selected_idx:
            sim = util.cos_sim(embeddings[i], embeddings[j]).item()  # cosine similarity

            if sim > SIM_THRESHOLD:   # 의미가 너무 비슷하면 제거
                keep = False
                break

            kw_j = extract_keywords(captions[j])
            overlap = len(kw_i & kw_j) / max(len(kw_i), 1)

            if overlap > 0.7:         # 키워드 많이 겹치면 제거
                keep = False
                break

        if keep:
            selected.append(cap)
            selected_idx.append(i)

    return selected

# ----------------------
# 7. 캡션 생성 루프
# ----------------------
def get_captions(image):
    final_caps = []
    attempts = 0

    while len(final_caps) < TARGET_CAPTIONS and attempts < MAX_ATTEMPTS:
        needed = TARGET_CAPTIONS - len(final_caps)

        new_caps = generate_captions(image, needed * 3)  # 부족분보다 넉넉히 생성
        new_caps = basic_filter(new_caps)                # 1차 필터

        combined = list(set(final_caps + new_caps))      # 중복 제거
        filtered = advanced_filter(combined)             # 유사도 필터

        final_caps = filtered[:TARGET_CAPTIONS]          # 목표 개수 맞춤
        attempts += 1

    return final_caps

# ----------------------
# 8. 데이터 수집
# ----------------------
dataset = []

for class_name in os.listdir(ROOT_DIR):                  # 클래스 폴더 순회
    class_path = os.path.join(ROOT_DIR, class_name)

    if not os.path.isdir(class_path):
        continue

    for filename in os.listdir(class_path):              # 이미지 순회
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(class_path, filename)
        image = Image.open(path).convert("RGB")          # 이미지 로드

        captions = get_captions(image)                   # 캡션 생성

        dataset.append({
            "image": f"{class_name}/{filename}",         # 상대 경로 저장
            "class": class_name,                         # 클래스 라벨
            "captions": captions                         # 캡션 리스트
        })

        print(f"\n{class_name}/{filename}")
        for i, c in enumerate(captions):
            print(f"{i+1}. {c}")

# ----------------------
# 9. Stratified Split
# ----------------------
class_groups = defaultdict(list)

for item in dataset:
    class_groups[item["class"]].append(item)             # 클래스별 묶기

train_set, val_set, test_set = [], [], []

for class_name, items in class_groups.items():
    random.shuffle(items)                                # 클래스 내부 shuffle

    total = len(items)

    train_end = max(1, int(total * TRAIN_RATIO))         # 최소 1개 보장
    val_end = train_end + max(1, int(total * VAL_RATIO))

    train_set += items[:train_end]
    val_set += items[train_end:val_end]
    test_set += items[val_end:]

# split 라벨 부여
for item in train_set:
    item["split"] = "train"

for item in val_set:
    item["split"] = "val"

for item in test_set:
    item["split"] = "test"

dataset = train_set + val_set + test_set                 # 다시 하나로 합침

# ----------------------
# 10. JSON 저장
# ----------------------
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=4, ensure_ascii=False)

print(f"\n완료: {OUTPUT_JSON} 생성됨")