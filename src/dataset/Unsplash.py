import os
import requests
import urllib.request
from PIL import Image

# =========================
# 1. 설정
# =========================
ACCESS_KEY = "epIj8a7EvUfAyR05Jr7iaaItTtfEVzpjyuK4BD_ldJA"

CLASSES = [
    "pizza", "hamburger", "sushi", "pasta", "steak", "sandwich", "fried_chicken", "ramen", "apple", "banana", "strawberry", "mushroom", "cake", "croissant", "salad",
    "dog", "cat", "lion", "elephant", "panda", "giraffe", "penguin", "dolphin", "owl", "butterfly",
    "rose", "sunflower", "cactus", "pine_tree", "maple_leaf", "tulip", "bamboo", "grass",
    "laptop", "smartphone", "digital_camera", "wristwatch", "backpack", "chair", "umbrella",
    "car", "bicycle", "motorcycle", "airplane", "bus",
    "mountain", "beach", "forest", "desert", "glacier"
]

MAX_NUM = 60

BASE_DIR = "us_images"
os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# 2. Unsplash 검색
# =========================
def unsplash_search(query, per_page=30):
    url = "https://api.unsplash.com/search/photos"

    headers = {
        "Authorization": f"Client-ID {ACCESS_KEY}"
    }

    params = {
        "query": query.replace("_", " "),  # 검색은 공백으로
        "per_page": per_page
    }

    res = requests.get(url, headers=headers, params=params)

    if res.status_code != 200:
        print("API ERROR:", res.text)
        return []

    data = res.json()

    return [item["urls"]["regular"] for item in data.get("results", [])]


# =========================
# 3. 다운로드
# =========================
def download_image(url, path):
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except:
        return False


# =========================
# 4. 검증
# =========================
def is_valid_image(path):
    try:
        img = Image.open(path)
        img.verify()
        return True
    except:
        return False


# =========================
# 5. 전체 파이프라인
# =========================
for cls in CLASSES:
    print(f"\n[START] {cls}")

    class_dir = os.path.join(BASE_DIR, cls)
    os.makedirs(class_dir, exist_ok=True)

    count = 0

    while count < MAX_NUM:
        urls = unsplash_search(cls, per_page=30)

        for url in urls:
            if count >= MAX_NUM:
                break

            file_name = f"us_{cls}_{count+1:03d}.jpg"
            file_path = os.path.join(class_dir, file_name)

            success = download_image(url, file_path)

            if not success:
                continue

            if not is_valid_image(file_path):
                os.remove(file_path)
                continue

            print(f"Saved: {file_path}")
            count += 1

    print(f"[DONE] {cls} -> {count} images")


# 이미지 개수, 해상도 검사

# =========================
# 설정
# =========================
BASE_DIR = "us_images"
EXPECTED_COUNT = 60
MIN_WIDTH = 256
MIN_HEIGHT = 256

# =========================
# 검사 시작
# =========================
print("\n[데이터셋 검사 시작]\n")

total_classes = 0
total_images = 0

for cls in os.listdir(BASE_DIR):
    class_dir = os.path.join(BASE_DIR, cls)

    if not os.path.isdir(class_dir):
        continue

    total_classes += 1

    files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith(".jpg")
    ]

    print(f"\n[{cls}]")

    # 1. 개수 체크
    count = len(files)
    total_images += count

    if count != EXPECTED_COUNT:
        print(f"❌ 이미지 개수 문제: {count}개 (기대값: {EXPECTED_COUNT})")
    else:
        print(f"✔ 이미지 개수 정상: {count}개")

    # 2. 해상도 체크
    small_images = []
    broken_images = []

    for file in files:
        path = os.path.join(class_dir, file)

        try:
            with Image.open(path) as img:
                width, height = img.size

                if width < MIN_WIDTH or height < MIN_HEIGHT:
                    small_images.append((file, width, height))

        except:
            broken_images.append(file)

    # 결과 출력
    if small_images:
        print(f"❌ 해상도 부족 이미지 ({len(small_images)}개)")
        for f, w, h in small_images[:5]:  # 너무 많으면 5개만 출력
            print(f"   - {f} ({w}x{h})")
    else:
        print("✔ 해상도 조건 만족")

    if broken_images:
        print(f"❌ 깨진 이미지 ({len(broken_images)}개)")
        for f in broken_images[:5]:
            print(f"   - {f}")
    else:
        print("✔ 깨진 이미지 없음")

# =========================
# 전체 요약
# =========================
print("\n======================")
print("전체 요약")
print("======================")
print(f"클래스 수: {total_classes}")
print(f"총 이미지 수: {total_images}")
print("검사 완료\n")