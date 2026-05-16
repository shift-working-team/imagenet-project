import os
import zipfile

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
ZIP_DIR = "data/raw_full_kg"

# 압축 해제 위치
EXTRACT_DIR = os.path.join(ZIP_DIR, "extracted")

os.makedirs(EXTRACT_DIR, exist_ok=True)

# ================================
# 2. zip 파일 목록
# ================================
zip_files = [
    f for f in os.listdir(ZIP_DIR)
    if f.endswith(".zip")
]

# ================================
# 3. 압축 해제
# ================================
for zip_file in zip_files:
    zip_path = os.path.join(ZIP_DIR, zip_file)

    print(f"{zip_file} 압축 해제 중...")

    try:
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)

    except Exception as e:
        print(f"오류 발생: {zip_file} → {e}")

print("모든 압축 해제 완료!")