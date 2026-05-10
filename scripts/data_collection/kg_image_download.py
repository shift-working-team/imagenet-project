import os

# ================================
# 0. 설정
# ================================
TARGET_COUNT = 60
MIN_RES = 128  # 해상도 128
PREFIX = "kg"
BASE_DIR = "./data/raw"

# ================================
# 1. 다운로드 경로
# ================================
DOWNLOAD_PATH = "data/raw_full_kg"

os.makedirs(DOWNLOAD_PATH, exist_ok=True)

# ================================
# 2. 사용할 Kaggle 데이터셋 (slug 기준)
# ================================
DATASETS = [
    # 음식 및 식재료
    "kmader/food41",

    # 동물
    "alessiocorrado99/animals10",
    "gpiosenka/100-bird-species",

    # 꽃
    "alxmamaev/flowers-recognition",

    # 과일
    "moltean/fruits",
    "yihfeng/strawberry-maturity",

    # 탈것
    "sshikamaru/car-object-detection",
    "jessicali9530/stanford-cars-dataset",
    "dataclusterlabs/vehicle-detection-image-dataset",
    "meowmeowmeowmeowmeow/vehicle-type-recognition",

    # 패션 및 잡화
    "promptcloudhq/jewelry-text-to-image-dataset",
    "ashwingupta3012/glasses-dataset",
    "agrigorev/clothing-dataset-full",
    "paramaggarwal/fashion-product-images-small"
]

# ================================
# 3. 다운로드 실행
# ================================
for ds in DATASETS:
    print(f"\nDownloading {ds} ...")
    os.system(f"kaggle datasets download -d {ds} -p {DOWNLOAD_PATH}")

print("\n모든 데이터셋 다운로드 완료!")