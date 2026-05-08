import zipfile
import os

# zip 파일 위치
ZIP_DIR = "data/raw_full_kg"

# 압축 해제 위치 (같은 폴더 안에 "extracted" 폴더로 분리)
EXTRACT_DIR = os.path.join(ZIP_DIR, "extracted")

os.makedirs(EXTRACT_DIR, exist_ok=True)

# zip 파일 목록 자동 찾기
zip_files = [f for f in os.listdir(ZIP_DIR) if f.endswith(".zip")]

for zip_file in zip_files:
    zip_path = os.path.join(ZIP_DIR, zip_file)

    print(f"{zip_file} 압축 해제 중...")

    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(EXTRACT_DIR)
    except Exception as e:
        print(f"오류 발생: {zip_file} → {e}")

print("모든 압축 해제 완료!")