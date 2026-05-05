import os
from PIL import Image
import hashlib

# ================================
# 1. 클래스 + 유사어 매핑 (최종)
# ================================
CLASS_MAP = {
    # 음식
    "pizza": ["pizza"],
    "hamburger": ["hamburger"],
    "sushi": ["sushi"],
    "pasta": ["pasta", "spaghetti"],
    "salad": ["salad"],
    "steak": ["steak"],
    "cup_cake": ["cup_cake", "cup cake"],
    "sandwich": ["sandwich"],
    "waffle": ["waffle"],
    "dumpling": ["dumpling"],

    # 동물
    "golden-retriever": ["golden retriever"],
    "bulldog": ["bulldog"],
    "siamese-cat": ["siamese"],
    "persian-cat": ["persian"],
    "elephant": ["elephant"],
    "sheep": ["sheep"],
    "horse": ["horse"],
    "penguin": ["penguin"],
    "butterfly": ["butterfly"],
    "squirrel": ["squirrel"],

    # 꽃
    "rose": ["rose"],
    "sunflower": ["sunflower"],
    "daisy": ["daisy"],
    "tulip": ["tulip"],
    "dandelion": ["dandelion"],
    "lily": ["lily"],
    "lavender": ["lavender"],
    "orchid": ["orchid"],
    "iris": ["iris"],
    "marigold": ["marigold"],
    "aster": ["aster"],

    # 과일
    "apple": ["apple"],
    "banana": ["banana"],
    "strawberry": ["strawberry"],
    "orange": ["orange"],
    "carrot": ["carrot"],
    "tomato": ["tomato"],
    "cucumber": ["cucumber"],

    # 탈것
    "car": ["car"],
    "bicycle": ["bicycle"],
    "motorcycle": ["motorcycle"],
    "airplane": ["airplane"],
    "bus": ["bus"],

    # 패션 및 잡화
    "t-shirt": ["t shirt", "t-shirt"],
    "sneakers": ["sneakers"],
    "earrings": ["earring", "earrings"],
    "glasses": ["glasses"],
    "pants": ["pants"],
    "bracelet": ["bracelet"],
    "necklace": ["necklace"]
}

# ================================
# 2. 경로
# ================================
HOME = os.path.expanduser("~")

SRC_ROOT = os.path.join(HOME, "Desktop", "raw_full_kg", "extracted")
DST_ROOT = os.path.join(HOME, "Desktop", "raw_kg")

os.makedirs(DST_ROOT, exist_ok=True)

# ================================
# 3. 해상도 필터
# ================================
def is_valid_image(path, min_size=128):
    try:
        with Image.open(path) as img:
            w, h = img.size
            return w >= min_size and h >= min_size
    except:
        return False

# ================================
# 4. 중복 제거
# ================================
def get_hash(path):
    try:
        with open(path, 'rb') as f:
            return hashlib.md5(f.read()).hexdigest()
    except:
        return None

seen_hashes = set()

# ================================
# 5. 클래스 매칭
# ================================
def match_class(folder_name):
    name = folder_name.lower().replace("-", " ").replace("_", " ")
    words = name.split()

    for target, keywords in CLASS_MAP.items():
        for kw in keywords:
            kw_words = kw.split()
            if all(word in words for word in kw_words):
                return target
    return None

# ================================
# 6. 메인 로직
# ================================
class_counter = {cls: 1 for cls in CLASS_MAP.keys()}

copied = 0
skipped = 0
no_match = 0

for root, dirs, files in os.walk(SRC_ROOT):
    for d in dirs:
        matched_class = match_class(d)

        if matched_class is None:
            no_match += 1
            continue

        src_path = os.path.join(root, d)
        dst_path = os.path.join(DST_ROOT, matched_class)

        for img in os.listdir(src_path):
            src_file = os.path.join(src_path, img)

            if not os.path.isfile(src_file):
                continue

            # 확장자 없는 경우도 허용
            try:
                with Image.open(src_file) as im:
                    im.verify()
            except:
                skipped += 1
                continue

            # 해상도 필터
            if not is_valid_image(src_file, 128):
                skipped += 1
                continue

            # 중복 제거
            img_hash = get_hash(src_file)
            if img_hash is None or img_hash in seen_hashes:
                skipped += 1
                continue
            seen_hashes.add(img_hash)

            if not os.path.exists(dst_path):
                os.makedirs(dst_path, exist_ok=True)

            number = str(class_counter[matched_class]).zfill(3)
            class_name_for_file = matched_class.replace("_", "-")
            new_name = f"kg_{class_name_for_file}_{number}.jpg"
            dst_file = os.path.join(dst_path, new_name)

            try:
                with Image.open(src_file) as im:
                    im.convert("RGB").save(dst_file, "JPEG")

                class_counter[matched_class] += 1
                copied += 1

                if copied % 100 == 0:
                    print(f"{copied}장 처리 중...")

            except:
                skipped += 1

print("\n완료!")
print(f"복사: {copied}")
print(f"스킵: {skipped}")
print(f"매칭 실패 폴더: {no_match}")