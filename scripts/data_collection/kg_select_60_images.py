import os
import random
import shutil

# ================================
# 0. 설정
# ================================
TARGET_COUNT = 60
MIN_RES = 128  # 해상도 128
PREFIX = "kg"
BASE_DIR = "./data/raw"

# ================================
# 1. 경로
# ================================
DATA_DIR = r"C:\Users\qud46\Desktop\raw_kg"

# ================================
# 2. 클래스 목록
# ================================
CLASS_LIST = [
    # 음식 및 식재료
    "pizza","hamburger","sushi","pasta","salad",
    "steak","cup_cake","sandwich","waffle","dumpling",

    # 동물
    "golden-retriever","bulldog","siamese-cat",
    "persian-cat","elephant","sheep","horse",
    "penguin","butterfly","squirrel",

    # 꽃
    "rose","sunflower","daisy","tulip","dandelion",
    "lily","lavender","orchid","iris","marigold","aster",

    # 과일
    "apple","banana","strawberry","orange",
    "carrot","tomato","cucumber",

    # 탈것
    "car","bicycle","motorcycle","airplane","bus",

    # 패션 및 잡화
    "t-shirt","sneakers","earrings","glasses",
    "pants","bracelet","necklace"
]

print("클래스별 이미지 60장 맞추기 시작\n")

# ================================
# 3. 메인 로직
# ================================
for cls in CLASS_LIST:

    cls_path = os.path.join(DATA_DIR, cls)

    if not os.path.exists(cls_path):
        print(f"{cls}: 폴더 없음 (skip)")
        continue

    # 이미지 파일 목록
    images = [
        f for f in os.listdir(cls_path)
        if os.path.isfile(os.path.join(cls_path, f))
    ]

    current_count = len(images)

    print(
        f"{cls}: 현재 {current_count}장 "
        f"→ 목표 {TARGET_COUNT}장"
    )

    # ================================
    # 1) 60장 초과 → 랜덤 삭제
    # ================================
    if current_count > TARGET_COUNT:

        delete_count = current_count - TARGET_COUNT

        to_delete = random.sample(
            images,
            delete_count
        )

        for file in to_delete:

            file_path = os.path.join(cls_path, file)

            try:
                os.remove(file_path)

            except:
                continue

        print(f"   → {delete_count}장 삭제 완료")

    # ================================
    # 2) 60장 미만 → 부족 개수 출력
    # ================================
    elif current_count < TARGET_COUNT:

        need_count = TARGET_COUNT - current_count

        print(
            f"   → {need_count}장 부족 "
            f"(추가 수집 필요)"
        )

    # ================================
    # 3) 정확히 60장
    # ================================
    else:
        print("   → 이미 60장 완료")

print("\n전체 정리 완료!")