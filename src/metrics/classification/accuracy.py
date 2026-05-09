from typing import List, Union

"""
사용 방법
from src.metrics.classification_metrics import calculate_accuracy


y_true = ["cat", "dog", "bird", "dog"]
y_pred = ["cat", "dog", "cat", "dog"]

accuracy = calculate_accuracy(y_true, y_pred)

print("Accuracy:", accuracy)
"""

def calculate_accuracy(
    y_true: List[Union[int, str]],
    y_pred: List[Union[int, str]]
) -> float:
    """
    Accuracy를 계산하는 함수.

    Parameters
    ----------
    y_true:
        실제 정답 라벨 리스트
        예: [0, 1, 2, 1, 0]

    y_pred:
        모델이 예측한 라벨 리스트
        예: [0, 1, 1, 1, 0]

    Returns
    -------
    accuracy:
        전체 데이터 중 맞춘 비율
        예: 0.8
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true와 y_pred의 길이가 같아야 합니다.")

    if len(y_true) == 0:
        raise ValueError("y_true와 y_pred는 비어 있으면 안 됩니다.")

    correct_count = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == pred_label:
            correct_count += 1

    accuracy = correct_count / len(y_true)

    return accuracy