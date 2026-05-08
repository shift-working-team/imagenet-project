from datasets import load_dataset


# ============================================================
# [설정 부분]
# ============================================================

# 확인할 Hugging Face 데이터셋 이름
DATASET_NAME = "KrushiJethe/fashion_data" #uran66/animals

# 확인할 split 이름
SPLIT_NAME = "train"

# 라벨 필드명
LABEL_FIELD_NAME = "articleType"

# streaming 사용 여부
# 라벨 구조만 확인할 때는 streaming=True로 해도 된다.
USE_STREAMING = True

# 문자열 라벨 데이터셋일 경우 전체 데이터를 훑어야 할 수 있다.
# None이면 전체 확인, 숫자를 넣으면 일부 샘플만 확인한다.
MAX_SCAN_ITEMS = None

# ============================================================


def get_unique_labels():
    """
    Hugging Face 데이터셋에서 라벨 목록을 중복 없이 출력한다.
    """

    print(f"[{DATASET_NAME}] 데이터셋 로드 중...")

    dataset = load_dataset(
        DATASET_NAME,
        split=SPLIT_NAME,
        streaming=USE_STREAMING,
    )

    # 데이터셋의 feature 정보에서 라벨 필드를 가져온다.
    label_feature = dataset.features[LABEL_FIELD_NAME]

    # ------------------------------------------------------------
    # 1. Food101처럼 label이 ClassLabel 타입인 경우
    # ------------------------------------------------------------
    # 이 경우 데이터 전체를 순회하지 않아도
    # dataset.features["label"].names 에서 전체 라벨명을 바로 확인할 수 있다.
    if hasattr(label_feature, "names") and label_feature.names is not None:
        label_names = label_feature.names

        print("\n라벨 목록")
        print("-" * 50)

        for idx, label_name in enumerate(label_names):
            print(f"{idx}: {label_name}")

        print("-" * 50)
        print(f"총 라벨 개수: {len(label_names)}")

        return label_names

    # ------------------------------------------------------------
    # 2. label이 문자열로 직접 들어있는 데이터셋인 경우
    # ------------------------------------------------------------
    # 이 경우에는 데이터를 직접 순회하면서 중복을 제거해야 한다.
    unique_labels = set()

    print("\n라벨 필드가 ClassLabel 타입이 아니므로 데이터를 순회합니다...")

    for idx, item in enumerate(dataset):
        if MAX_SCAN_ITEMS is not None and idx >= MAX_SCAN_ITEMS:
            break

        label_value = item.get(LABEL_FIELD_NAME)

        if label_value is None:
            continue

        unique_labels.add(str(label_value))

    label_names = sorted(unique_labels)

    print("\n라벨 목록")
    print("-" * 50)
    
    
    for idx, label_name in enumerate(label_names):
        print(f"{idx}: {label_name}")

    print("-" * 50)
    print(f"총 라벨 개수: {len(label_names)}")

    return label_names


if __name__ == "__main__":
    get_unique_labels()