import os
import re
import requests
from PIL import Image
from io import BytesIO
import time

# =========================
# 1. 설정 (여기만 수정하면 됨)
# =========================
ACCESS_KEY = "epIj8a7EvUfAyR05Jr7iaaItTtfEVzpjyuK4BD_ldJA"

TARGET_COUNT = 100
MIN_WIDTH = 256
MIN_HEIGHT = 256
SLEEP_TIME = 2

BASE_DIR = "un_images"

MASTER_CLASSES = [
    "pizza","hamburger","sushi","pasta","salad","steak","cake","sandwich","fried_chicken","bread",
    "apple","banana","strawberry","orange","carrot",
    "golden_retriever","bulldog","siamese_cat","persian_cat","eagle","owl","lion","elephant","zebra","giraffe",
    "rose","sunflower","daisy","tulip","palm_tree","pine_tree","maple_tree","bamboo",
    "laptop","watch","camera","chair","clock","microwave","refrigerator",
    "car","bicycle","motorcycle","airplane","bus",
    "backpack","sneakers","umbrella","glasses","hat"
]

SIMPLE_CLASSES = [
    "pizza","burger","sushi","pasta","salad","steak","cake","sandwich","fried_chicken","bread",
    "apple","banana","strawberry","orange","carrot",
    "golden_retriever","bulldog","siamese_cat","persian_cat","eagle","owl","lion","elephant","zebra","giraffe",
    "rose","sunflower","daisy","tulip","palm_tree","pine_tree","maple_tree","bamboo",
    "laptop","wristwatch","camera","chair","wall_clock","microwave","refrigerator",
    "car","bicycle","motorcycle","airplane","bus",
    "backpack","sneakers","umbrella","glasses","hat"
]

os.makedirs(BASE_DIR, exist_ok=True)

# =========================
# 2. 유틸
# =========================
def format_name(name):
    return name.replace("_", "-")

def get_start_index(folder, simple_cls):
    pattern = re.compile(rf"un_{simple_cls}_(\d+)\.jpg")
    max_idx = 0

    for f in os.listdir(folder):
        match = pattern.match(f)
        if match:
            num = int(match.group(1))
            max_idx = max(max_idx, num)

    return max_idx + 1

# =========================
# 3. API
# =========================
def search_images(query, page):
    url = "https://api.unsplash.com/search/photos"

    headers = {
        "Authorization": f"Client-ID {ACCESS_KEY}"
    }

    params = {
        "query": query.replace("_", " "),
        "per_page": 30,
        "page": page
    }

    res = requests.get(url, headers=headers, params=params)

    if res.status_code == 429:
        print("⏳ Rate limit → 60초 대기")
        time.sleep(60)
        return []

    if res.status_code != 200:
        print("API ERROR:", res.text)
        return []

    return [item["urls"]["regular"] for item in res.json().get("results", [])]

def download_image(url):
    try:
        res = requests.get(url, timeout=10)
        if res.status_code != 200:
            return None

        img = Image.open(BytesIO(res.content))
        w, h = img.size

        if w < MIN_WIDTH or h < MIN_HEIGHT:
            return None

        return res.content
    except:
        return None

# =========================
# 4. 메인
# =========================
for idx in range(len(MASTER_CLASSES)):

    master_cls = MASTER_CLASSES[idx]
    simple_cls = format_name(SIMPLE_CLASSES[idx])

    print(f"\n[START] {master_cls}")

    class_dir = os.path.join(BASE_DIR, master_cls)
    os.makedirs(class_dir, exist_ok=True)

    start_idx = get_start_index(class_dir, simple_cls)
    count = start_idx - 1

    page = 1
    seen = set()

    while count < TARGET_COUNT:
        urls = search_images(simple_cls, page)

        if not urls:
            print("이미지 없음")
            break

        for url in urls:
            if count >= TARGET_COUNT:
                break

            if url in seen:
                continue
            seen.add(url)

            img_data = download_image(url)
            if img_data is None:
                continue

            count += 1

            file_name = f"un_{simple_cls}_{count:03d}.jpg"
            path = os.path.join(class_dir, file_name)

            with open(path, "wb") as f:
                f.write(img_data)

            print(f"Saved: {path}")

        page += 1
        time.sleep(SLEEP_TIME)

    print(f"[DONE] {master_cls} -> {count}/{TARGET_COUNT}")

# =========================
# 5. 검증 + 부족분 자동 보충
# =========================
print("\n[검증 시작]\n")

for idx in range(len(MASTER_CLASSES)):

    master_cls = MASTER_CLASSES[idx]
    simple_cls = format_name(SIMPLE_CLASSES[idx])

    class_dir = os.path.join(BASE_DIR, master_cls)
    files = [f for f in os.listdir(class_dir) if f.endswith(".jpg")]

    valid_files = []

    for f in files:
        path = os.path.join(class_dir, f)
        try:
            img = Image.open(path)
            w, h = img.size

            if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                valid_files.append(f)
            else:
                os.remove(path)
        except:
            os.remove(path)

    count = len(valid_files)

    print(f"{master_cls}: {count}/{TARGET_COUNT}")

    if count < TARGET_COUNT:
        print(f"→ 부족분 재수집 시작")

        page = 1
        seen = set()
        start_idx = get_start_index(class_dir, simple_cls)

        while count < TARGET_COUNT:
            urls = search_images(simple_cls, page)

            if not urls:
                break

            for url in urls:
                if count >= TARGET_COUNT:
                    break

                if url in seen:
                    continue
                seen.add(url)

                img_data = download_image(url)
                if img_data is None:
                    continue

                file_name = f"un_{simple_cls}_{start_idx:03d}.jpg"
                path = os.path.join(class_dir, file_name)

                with open(path, "wb") as f:
                    f.write(img_data)

                print(f"ReSaved: {path}")

                start_idx += 1
                count += 1

            page += 1
            time.sleep(SLEEP_TIME)

print("\n[완료]")