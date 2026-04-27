import os
import requests
import urllib.request
from PIL import Image

# =========================
# 1. 설정
# =========================
ACCESS_KEY = "epIj8a7EvUfAyR05Jr7iaaItTtfEVzpjyuK4BD_ldJA"

CLASSES = ["pizza", "hamburger", "sushi", "pasta", "steak", "sandwich", "fried_chicken", "ramen", "apple", "banana", "strawberry", "mushroom", "cake", "croissant", "salad",
    "dog", "cat", "lion", "elephant", "panda", "giraffe", "penguin", "dolphin", "owl", "butterfly",
    "rose", "sunflower", "cactus", "pine_tree", "maple_leaf", "tulip", "bamboo", "grass",
    "laptop", "smartphone", "digital_camera", "wristwatch", "backpack", "chair", "umbrella",
    "car", "bicycle", "motorcycle", "airplane", "bus",
    "mountain", "beach", "forest", "desert", "glacier"]
MAX_NUM = 60

BASE_DIR = "images"
os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# 2. Unsplash 검색 함수
# =========================
def unsplash_search(query, per_page=30):
    url = "https://api.unsplash.com/search/photos"

    headers = {
        "Authorization": f"Client-ID {ACCESS_KEY}"
    }

    params = {
        "query": query,
        "per_page": per_page
    }

    res = requests.get(url, headers=headers, params=params)

    if res.status_code != 200:
        print("API ERROR:", res.text)
        return []

    data = res.json()

    urls = []
    for item in data.get("results", []):
        img_url = item["urls"]["regular"]
        urls.append(img_url)

    return urls


# =========================
# 3. 이미지 다운로드
# =========================
def download_image(url, path):
    try:
        urllib.request.urlretrieve(url, path)
        return True
    except:
        return False


# =========================
# 4. 이미지 검증
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

    urls = unsplash_search(cls, per_page=30)

    count = 0

    # 여러 페이지 보완 (60개 맞추기)
    page = 1

    while count < MAX_NUM:
        urls = unsplash_search(cls, per_page=30)

        for url in urls:
            if count >= MAX_NUM:
                break

            file_name = f"{cls}_{count}.jpg"
            file_path = os.path.join(class_dir, file_name)

            success = download_image(url, file_path)

            if not success:
                continue

            if not is_valid_image(file_path):
                os.remove(file_path)
                continue

            print(f"Saved: {file_path}")
            count += 1

        page += 1

    print(f"[DONE] {cls} -> {count} images")