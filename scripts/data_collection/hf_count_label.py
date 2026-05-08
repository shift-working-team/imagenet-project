from datasets import load_dataset
from collections import Counter
from dotenv import load_dotenv
import os

# ============================================================
# [설정 부분]
# ============================================================
load_dotenv()
HF_TOKEN = os.environ.get("HF_TOKEN")

# 확인할 Hugging Face 데이터셋 이름
DATASET_NAME = "jbarat/plant_species"  # 예: "uran66/animals"

# 확인할 split 이름
SPLIT_NAME = "train"

# 라벨 필드명
LABEL_FIELD_NAME = "label"

# streaming 사용 여부
# True  : 전체 데이터셋을 미리 다운로드하지 않고 하나씩 읽으면서 확인
# False : 로컬 캐시에 데이터셋을 다운로드한 뒤 확인
USE_STREAMING = True

# 문자열 라벨 데이터셋일 경우 전체 데이터를 훑어야 정확한 개수를 알 수 있다.
# None이면 전체 확인, 숫자를 넣으면 일부 샘플만 확인한다.
MAX_SCAN_ITEMS = None

# ============================================================


def get_label_name(dataset, label_value):

    label_feature = dataset.features[LABEL_FIELD_NAME]

    # ClassLabel 타입이면 숫자 라벨을 문자열 라벨명으로 변환한다.
    if hasattr(label_feature, "int2str") and isinstance(label_value, int):
        return label_feature.int2str(label_value)

    # 이미 문자열 라벨이면 그대로 문자열로 변환해서 사용한다.
    return str(label_value)


def get_unique_labels_with_counts():

    print(f"[{DATASET_NAME}] 데이터셋 로드 중...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
        streaming=USE_STREAMING,
        token=HF_TOKEN
    )

    # 데이터셋의 feature 정보에서 라벨 필드를 가져온다.
    label_feature = dataset.features[LABEL_FIELD_NAME]

    # 클래스별 이미지 개수를 저장할 Counter
    label_counter = Counter()

    print("\n클래스별 이미지 개수 집계 중...")

    # streaming=True인 경우에도 dataset을 순회하면서 개수를 셀 수 있다.
    for idx, item in enumerate(dataset):
        # MAX_SCAN_ITEMS가 설정되어 있으면 지정한 개수까지만 확인한다.
        if MAX_SCAN_ITEMS is not None and idx >= MAX_SCAN_ITEMS:
            break

        label_value = item.get(LABEL_FIELD_NAME)

        # 라벨 값이 없는 데이터는 건너뛴다.
        if label_value is None:
            continue

        # 숫자 라벨이면 실제 라벨명으로 변환하고,
        # 문자열 라벨이면 그대로 사용한다.
        label_name = get_label_name(dataset, label_value)

        # 해당 라벨의 이미지 개수를 1 증가시킨다.
        label_counter[label_name] += 1

    print("\n라벨 목록 및 클래스별 이미지 개수")
    print("-" * 60)

    # ------------------------------------------------------------
    # 1. Food101처럼 label이 ClassLabel 타입인 경우
    # ------------------------------------------------------------
    # label_feature.names가 있으면 원래 데이터셋의 라벨 순서대로 출력한다.
    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_names = label_feature.names

        for idx, label_name in enumerate(label_names):
            count = label_counter.get(label_name, 0)
            print(f"{idx}: {label_name} - {count} 장")

    # ------------------------------------------------------------
    # 2. label이 문자열로 직접 들어있는 데이터셋인 경우
    # ------------------------------------------------------------
    # Counter에 모인 라벨명을 이름순으로 정렬해서 출력한다.
    else:
        label_names = sorted(label_counter.keys())

        for idx, label_name in enumerate(label_names):
            count = label_counter[label_name]
            print(f"{idx}: {label_name} - {count} 장")

    print("-" * 60)
    print(f"총 라벨 개수: {len(label_counter)}")
    print(f"총 이미지 개수: {sum(label_counter.values())}")

    return label_counter

if __name__ == "__main__":
    get_unique_labels_with_counts()