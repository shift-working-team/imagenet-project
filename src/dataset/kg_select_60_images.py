import os
import random

# ================================
# 1. 경로
# ================================
DATA_DIR = r"C:\Users\qud46\Desktop\raw_kg"

# ================================
# 2. 클래스 목록 (최종 50개)
# ================================
CLASS_LIST = [
    # 음식 및 식재료
    "pizza","hamburger","sushi","pasta","salad","steak","cup_cake","sandwich","waffle","dumpling",

    # 동물
    "golden-retriever","bulldog","siamese-cat","persian-cat",
    "elephant","sheep","horse","penguin","butterfly","squirrel",

    # 꽃
    "rose","sunflower","daisy","tulip","dandelion","lily","lavender","orchid","iris","marigold","aster",

    # 과일
    "apple","banana","strawberry","orange","carrot","tomato","cucumber",

    # 탈것
    "car","bicycle","motorcycle","airplane","bus",

    # 패션 및 잡화
    "t-shirt","sneakers","earrings","glasses","pants","bracelet","necklace"
]

# ================================
# 3. 기준 개수
# ================================
THRESHOLD = 60  # 원하는 장수로 변경

print("클래스별 이미지 정리 시작\n")

# ================================
# 4. 메인 로직
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

    print(f"{cls}: 현재 {current_count}장 → 목표 {THRESHOLD}장")

    # 초과된 경우 → 랜덤 삭제
    if current_count > THRESHOLD:
        delete_count = current_count - THRESHOLD

        # 랜덤으로 삭제할 파일 선택
        to_delete = random.sample(images, delete_count)

        for file in to_delete:
            file_path = os.path.join(cls_path, file)
            try:
                os.remove(file_path)
            except:
                continue

        print(f"   → {delete_count}장 삭제 완료")

    else:
        print(f"   → 삭제 없음")

print("\n 정리 완료!")