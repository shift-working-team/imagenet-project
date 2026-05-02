import os
import re
import glob
from dotenv import load_dotenv
from datasets import load_dataset
from PIL import Image

# =====================================================================
# [설정 부분]
# =====================================================================
# 토큰
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

print(f"이거 토큰 : {HF_TOKEN}")

# 수집할 데이터셋
DATASET_NAME = "KrushiJethe/fashion_data" 
# 데이터셋 내의 이미지 데이터가 있는 필드명
IMAGE_FIELD_NAME = "image"
# 데이터셋 내의 라벨 데이터가 있는 필드명
LABEL_FIELD_NAME = "articleType"

# 여러 라벨을 하나의 대표 클래스로 묶는 매핑 딕셔너리
CLASS_MAPPING = {
    "t-shirt": ["Tshirts", "Tops"],
    "sneakers":["Casual Shoes"],
    #"umbrella":["Umbrellas"],
    "glasses":["Sunglasses"],
    "pants":["Jeans"],
}

# 클래스별로 수집할 이미지의 최대 개수
NUM_IMAGES_PER_CLASS = 100
# 저장할 이미지의 해상도 (width, height)
TARGET_RESOLUTION = 256 
# 이미지를 저장할 최상위 디렉토리명
BASE_SAVE_DIR = "./dataset_output" 
# 수집할 데이터셋의 split 이름 (예: "train", "validation", "test")
SPLIT_NAME = "train"

# 컨테이너를 실행한 상태에서는 컨테이너에 캐시 저장됨
# 캐시 확인 -> ls -lah ~/.cache/huggingface
# 캐시 삭제 -> rm -rf ~/.cache/huggingface
USE_STREAMING = False
# =====================================================================

# 클래스 명명 규칙 적용
def format_class_name(class_name: str) -> str:
    """
    클래스명은 소문자로 하고 띄어쓰기가 있을 경우 "-"로 대체
    """
    return str(class_name).lower().replace("_", "-").replace(" ", "-")

# 마지막 이미지의 번호 + 1
def get_next_image_index(save_dir: str, formatted_class_name: str) -> int:
    """
    이미지를 여러 차례 이어서 수집할 수 있도록 마지막 이미지 번호를 탐색
    디렉토리를 스캔하여 가장 높은 번호를 찾은 뒤 +1을 반환
    """
    if not os.path.exists(save_dir):
        return 1
    
    # jpg와 jpeg 확장자 모두 검색
    search_pattern_jpg = os.path.join(save_dir, f"hf_{formatted_class_name}_*.jpg")
    search_pattern_jpeg = os.path.join(save_dir, f"hf_{formatted_class_name}_*.jpeg")
    
    existing_files = glob.glob(search_pattern_jpg) + glob.glob(search_pattern_jpeg)
    
    max_idx = 0
    # 파일명에서 정규표현식을 통해 번호 추출 (예: hf_fried-chicken_001.jpg -> 1)
    regex = re.compile(rf"hf_{formatted_class_name}_(\d+)\.jpe?g$")
    
    for file_path in existing_files:
        basename = os.path.basename(file_path)
        match = regex.match(basename)
        if match:
            idx = int(match.group(1))
            if idx > max_idx:
                max_idx = idx
                
    return max_idx + 1

def collect_hf_images():
    """
    메인 데이터 수집 함수.
    Hugging Face 데이터셋에서 설정을 반영하여 이미지를 수집하고 저장
    """

    label_to_rep_class = {}
    for rep_class, labels in CLASS_MAPPING.items():
        for label in labels:
            label_to_rep_class[label] = rep_class
    

    print(label_to_rep_class)
    # 데이터셋별 낱개로 수집
    # streaming=True 속성을 사용하면 전체 데이터셋을 메모리나 디스크에 한 번에 다운로드하지 않고
    # generator 형태로 하나씩(낱개로) 가져오므로 메모리와 네트워크 효율성이 극대화
    print(f"[{DATASET_NAME}] 데이터셋 스트리밍 로드 시작...")
    dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME, streaming=USE_STREAMING, token=HF_TOKEN)
    
    # 랜덤으로 가져오기
    # random_seed = random.randint(0, 10000)
    # dataset = load_dataset(DATASET_NAME, split=SPLIT_NAME, streaming=USE_STREAMING).shuffle(seed=random_seed, buffer_size=1000)

    # 클래스별로 포맷팅된 폴더명과, 현재까지 수집된 개수, 그리고 저장될 시작 번호를 관리할 딕셔너리
    class_info = {}
    for label in CLASS_MAPPING.keys():
        formatted_name = format_class_name(label)
        save_path = os.path.join(BASE_SAVE_DIR, formatted_name)
        
        # [규칙 1, 4] 클래스를 폴더로 관리하며 폴더명은 변환된 클래스명을 따른다.
        os.makedirs(save_path, exist_ok=True)
        
        # 이어서 수집하기 위한 시작 인덱스 탐색
        start_idx = get_next_image_index(save_path, formatted_name)
        
        class_info[label] = {
            "formatted_name": formatted_name,
            "save_path": save_path,
            "collected_count": 0,
            "current_idx": start_idx
        }

    print("데이터 수집을 시작합니다...")
    
    # 스트리밍 데이터 순회
    for item in dataset:

        print("1. 데이터셋 로드 시작...")
        # 모든 클래스가 목표 수집량을 채웠는지 확인
        if all(info["collected_count"] >= NUM_IMAGES_PER_CLASS for info in class_info.values()):
            print("모든 클래스의 이미지 수집이 완료되었습니다.")
            break

        print("2. 데이터셋 라벨 아이템 꺼내기...")    
        current_label = item.get(LABEL_FIELD_NAME)
        
        print(current_label)
        # 현재 뽑힌 라벨이 정의한 매핑 딕셔너리에 존재하는지 확인
        if current_label in label_to_rep_class:
            rep_class = label_to_rep_class[current_label]
            target_info = class_info[rep_class]
            

            print("4. 이미지 유효성 검사...")
            # 이미 목표 개수를 채운 클래스라면 스킵
            if target_info["collected_count"] >= NUM_IMAGES_PER_CLASS:
                continue
                
            # 이미지 유효성 체크
            image = item.get(IMAGE_FIELD_NAME)
            if image is None:
                continue
            
            print("5. 이미지 변환...")
            try:
                # 이미지를 jpg/jpeg로만 취급하기 위해 RGB 모드로 변환 (알파 채널 등 제거)
                if image.mode != "RGB":
                    image = image.convert("RGB")
                    
                #이미지 해상도가 최소 256px만 수집
                if image.width < TARGET_RESOLUTION or image.height < TARGET_RESOLUTION:
                    continue

                print("6. 클래스 명명 규칙에 따라...")
                # [규칙 3, 4] 이미지 명명 규칙 (hf_[클래스명]_[3자리숫자].jpg)
                # {:03d}를 통해 3자리 숫자로 맞추고 빈자리는 0으로 채움
                file_name = f"hf_{target_info['formatted_name']}_{target_info['current_idx']:03d}.jpg"
                file_path = os.path.join(target_info["save_path"], file_name)
                
                print("7. 이미지 저장...")
                image.save(file_path, "JPEG", quality=95)
                
                # 카운트 및 인덱스 증가
                target_info["collected_count"] += 1
                target_info["current_idx"] += 1
                
                print(f"Saved: {file_path} ({target_info['collected_count']}/{NUM_IMAGES_PER_CLASS})")
                
            except Exception as e:
                # 오류 발생 시 스크립트가 멈추지 않도록 예외 처리
                print(f"이미지 저장 중 오류 발생 (Label: {current_label}): {e}")

if __name__ == "__main__":
    collect_hf_images()