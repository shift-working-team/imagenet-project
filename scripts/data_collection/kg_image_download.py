import os

# ================================
# 1. 다운로드 경로
# ================================
download_path = "data/raw_full_kg"
os.makedirs(download_path, exist_ok=True)

# ================================
# 2. 사용할 Kaggle 데이터셋 (slug 기준)
# ================================
datasets = [
    # 음식 및 식재료
    "kmader/food41",  # Food-41

    # 동물
    "alessiocorrado99/animals10",  # Animals-10
    "gpiosenka/100-bird-species",  # (90 Animals 대체로 많이 사용됨)

    # 꽃
    "alxmamaev/flowers-recognition",  # Flower Classification (10+ classes)

    # 과일
    "moltean/fruits",  # Fruit dataset (disease dataset 대체로 사용 많음)
    "yihfeng/strawberry-maturity",  # Strawberry maturity

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
for ds in datasets:
    print(f"\n Downloading {ds} ...")
    os.system(f"kaggle datasets download -d {ds} -p {download_path}")

print("\n 모든 데이터셋 다운로드 완료!")