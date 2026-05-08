import os

# ================================
# 1. 경로 (바탕화면 기준)
# ================================
HOME = os.path.expanduser("~")
DATA_DIR = os.path.join(HOME, "Desktop", "raw_kg")

THRESHOLD = 60  # 변경 가능

# ================================
# 2. 클래스 목록 (최종 50개)
# ================================
CLASS_LIST = [
    # 음식 및 식재료
    "pizza","hamburger","sushi","pasta","salad","steak","cup_cake","sandwich","waffle","dumpling",

    # 동물
    "golden_retriever","bulldog","siamese_cat","persian_cat",
    "elephant","sheep","horse","penguin","butterfly","squirrel",

    # 꽃
    "rose","sunflower","daisy","tulip","dandelion","lily","lavender","orchid","iris","marigold","aster",

    # 과일
    "apple","banana","strawberry","orange","carrot","tomato","cucumber",

    # 탈 것
    "car","bicycle","motorcycle","airplane","bus",

    # 패션 및 잡화
    "t_shirt","sneakers","earrings","glasses","pants","bracelet","necklace"
]

print(f"{THRESHOLD}장 이하 클래스 목록 (0장 포함)\n")

low_classes = []

# ================================
# 3. 클래스별 개수 체크
# ================================
for cls in sorted(CLASS_LIST):
    cls_path = os.path.join(DATA_DIR, cls)

    # 폴더 없으면 0장 처리
    if not os.path.exists(cls_path):
        count = 0
    else:
        count = len([
            f for f in os.listdir(cls_path)
            if os.path.isfile(os.path.join(cls_path, f))
        ])

    if count < THRESHOLD:
        print(f"{cls}: {count}장")
        low_classes.append((cls, count))

# ================================
# 4. 요약
# ================================
print("\n요약")
print(f"{THRESHOLD}장 미만 클래스 수: {len(low_classes)}개")