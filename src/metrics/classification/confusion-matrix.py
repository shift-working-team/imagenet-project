from typing import Any, Dict, List, Optional, Union

"""
사용 방법

from src.metrics.classification_metrics import (
    calculate_confusion_matrix,
    print_confusion_matrix
)


y_true = ["cat", "dog", "bird", "cat", "dog", "bird", "cat"]
y_pred = ["cat", "dog", "cat", "dog", "dog", "bird", "cat"]

result = calculate_confusion_matrix(
    y_true=y_true,
    y_pred=y_pred,
    labels=["cat", "dog", "bird"]
)

labels = result["labels"]
matrix = result["matrix"]

print_confusion_matrix(labels, matrix)
"""

Label = Union[int, str]

def calculate_confusion_matrix(
    y_true: List[Label],
    y_pred: List[Label],
    labels: Optional[List[Label]] = None
) -> Dict[str, Any]:
    """
    Confusion Matrix를 계산하는 함수.

    Confusion Matrix의 기본 규칙:
    - 행(row): 실제 정답 클래스
    - 열(column): 모델이 예측한 클래스

    Parameters
    ----------
    y_true:
        실제 정답 라벨 리스트
        예: ["cat", "dog", "bird", "cat"]

    y_pred:
        모델이 예측한 라벨 리스트
        예: ["cat", "dog", "cat", "dog"]

    labels:
        클래스 순서를 직접 지정하고 싶을 때 사용
        예: ["cat", "dog", "bird"]

        labels를 지정하지 않으면 y_true와 y_pred에 등장한 라벨을 기준으로 자동 생성한다.

    Returns
    -------
    {
        "labels": 클래스 순서,
        "matrix": confusion matrix
    }
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true와 y_pred의 길이가 같아야 합니다.")

    if len(y_true) == 0:
        raise ValueError("y_true와 y_pred는 비어 있으면 안 됩니다.")

    if labels is None:
        labels = _get_unique_labels(y_true, y_pred)

    label_to_index = {}

    for index, label in enumerate(labels):
        label_to_index[label] = index

    num_classes = len(labels)

    matrix = []

    for _ in range(num_classes):
        row = [0] * num_classes
        matrix.append(row)

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label not in label_to_index:
            raise ValueError(f"labels에 실제 정답 라벨 '{true_label}'이 없습니다.")

        if pred_label not in label_to_index:
            raise ValueError(f"labels에 예측 라벨 '{pred_label}'이 없습니다.")

        true_index = label_to_index[true_label]
        pred_index = label_to_index[pred_label]

        matrix[true_index][pred_index] += 1

    return {
        "labels": labels,
        "matrix": matrix
    }


def _get_unique_labels(
    y_true: List[Label],
    y_pred: List[Label]
) -> List[Label]:
    """
    y_true와 y_pred에 등장한 라벨을 중복 없이 가져온다.

    set을 사용하면 순서가 섞일 수 있으므로,
    등장한 순서를 유지하면서 라벨 목록을 만든다.
    """

    labels = []

    for label in y_true + y_pred:
        if label not in labels:
            labels.append(label)

    return labels


def print_confusion_matrix(
    labels: List[Label],
    matrix: List[List[int]]
) -> None:
    """
    Confusion Matrix를 보기 좋게 출력하는 함수.

    행(row): 실제 정답
    열(column): 모델 예측
    """

    label_names = [str(label) for label in labels]

    print("Confusion Matrix")
    print("-" * 50)

    header = "Actual \\ Pred".ljust(15)

    for label in label_names:
        header += label.rjust(10)

    print(header)

    for label, row in zip(label_names, matrix):
        line = label.ljust(15)

        for value in row:
            line += str(value).rjust(10)

        print(line)