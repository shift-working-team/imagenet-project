import os
from PIL import Image

# ================================
# 1. 경로 설정
# ================================
DATA_DIR = "./data/raw"

# ================================
# 2. 최소 해상도 기준
# ================================
MIN_RES = 256

# ================================
# 3. 결과 저장 리스트
# ================================
low_resolution_images = []

# ================================
# 4. 이미지 검사
# ================================
for class_name in os.listdir(DATA_DIR):

    class_path = os.path.join(DATA_DIR, class_name)

    # 폴더만 처리
    if not os.path.isdir(class_path):
        continue

    for file_name in os.listdir(class_path):

        file_path = os.path.join(class_path, file_name)

        # 파일만 처리
        if not os.path.isfile(file_path):
            continue

        try:
            with Image.open(file_path) as img:

                width, height = img.size

                # 256 미만 이미지 찾기
                if width < MIN_RES or height < MIN_RES:

                    low_resolution_images.append({
                        "class": class_name,
                        "file": file_name,
                        "width": width,
                        "height": height
                    })

        except Exception as e:
            print(f"오류 발생: {file_path}")
            print(e)

# ================================
# 5. 결과 출력
# ================================
print(f"\n해상도 {MIN_RES}px 미만 이미지 목록\n")

if len(low_resolution_images) == 0:
    print("모든 이미지가 기준 해상도를 만족합니다.")

else:
    for item in low_resolution_images:

        print(
            f"[{item['class']}] "
            f"{item['file']} "
            f"→ {item['width']} x {item['height']}"
        )

# ================================
# 6. 총 개수 출력
# ================================
print("\n====================")
print(f"총 개수: {len(low_resolution_images)}장")
print("====================")