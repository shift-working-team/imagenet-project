from typing import List, Union, Dict, Any

"""
사용 방법
from src.metrics.classification_metrics import calculate_f1_score


y_true = ["cat", "dog", "bird", "dog", "cat"]
y_pred = ["cat", "dog", "dog", "dog", "cat"]

방법 1 (average : macro, weighted, micro)
f1_macro = calculate_f1_score(y_true, y_pred, average="macro")
print("F1 Macro:", f1_macro)

방법 2
metrics = calculate_f1_metrics(y_true, y_pred)
print(metrics)
"""

Label = Union[int, str]


def calculate_f1_score(
    y_true: List[Label],
    y_pred: List[Label],
    average: str = "macro"
) -> float:
    """
    F1-Score를 계산하는 함수.

    Parameters
    ----------
    y_true:
        실제 정답 라벨 리스트
        예: [0, 1, 2, 1, 0]

    y_pred:
        모델이 예측한 라벨 리스트
        예: [0, 1, 1, 1, 0]

    average:
        F1 평균 계산 방식

        - "macro":
            클래스별 F1을 단순 평균
            클래스 불균형이 있을 때 중요하게 봐야 함

        - "weighted":
            클래스별 데이터 개수를 반영해서 평균
            데이터가 많은 클래스의 영향이 큼

        - "micro":
            전체 TP, FP, FN을 합쳐서 계산
            단일 라벨 다중 분류에서는 Accuracy와 비슷하게 나오는 경우가 많음

    Returns
    -------
    f1_score:
        계산된 F1-Score
    """

    if len(y_true) != len(y_pred):
        raise ValueError("y_true와 y_pred의 길이가 같아야 합니다.")

    if len(y_true) == 0:
        raise ValueError("y_true와 y_pred는 비어 있으면 안 됩니다.")

    if average not in ["macro", "weighted", "micro"]:
        raise ValueError("average는 'macro', 'weighted', 'micro' 중 하나여야 합니다.")

    labels = _get_unique_labels(y_true, y_pred)

    if average == "micro":
        return _calculate_micro_f1(y_true, y_pred, labels)

    class_metrics = []

    for label in labels:
        tp, fp, fn = _calculate_tp_fp_fn(y_true, y_pred, label)

        precision = _safe_divide(tp, tp + fp)
        recall = _safe_divide(tp, tp + fn)
        f1 = _safe_divide(2 * precision * recall, precision + recall)

        support = y_true.count(label)

        class_metrics.append({
            "label": label,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": support
        })

    if average == "macro":
        return sum(item["f1"] for item in class_metrics) / len(class_metrics)

    if average == "weighted":
        total_support = sum(item["support"] for item in class_metrics)

        if total_support == 0:
            return 0.0

        weighted_f1 = 0.0

        for item in class_metrics:
            weighted_f1 += item["f1"] * item["support"]

        return weighted_f1 / total_support

    return 0.0


def calculate_f1_metrics(
    y_true: List[Label],
    y_pred: List[Label]
) -> Dict[str, Any]:
    """
    여러 종류의 F1-Score를 한 번에 계산하는 함수.

    Returns
    -------
    {
        "f1_macro": 클래스별 F1 단순 평균,
        "f1_weighted": 클래스 개수를 반영한 F1 평균,
        "f1_micro": 전체 기준 F1
    }
    """

    return {
        "f1_macro": calculate_f1_score(y_true, y_pred, average="macro"),
        "f1_weighted": calculate_f1_score(y_true, y_pred, average="weighted"),
        "f1_micro": calculate_f1_score(y_true, y_pred, average="micro"),
    }


def _get_unique_labels(
    y_true: List[Label],
    y_pred: List[Label]
) -> List[Label]:
    """
    y_true와 y_pred에 등장한 모든 라벨을 중복 없이 가져온다.

    set을 바로 쓰면 라벨 순서가 섞일 수 있으므로,
    등장한 순서를 유지하면서 라벨 목록을 만든다.
    """

    labels = []

    for label in y_true + y_pred:
        if label not in labels:
            labels.append(label)

    return labels


def _calculate_tp_fp_fn(
    y_true: List[Label],
    y_pred: List[Label],
    target_label: Label
) -> tuple[int, int, int]:
    """
    특정 클래스에 대한 TP, FP, FN을 계산한다.

    TP:
        실제 정답도 target_label이고,
        예측도 target_label인 경우

    FP:
        실제 정답은 target_label이 아닌데,
        예측을 target_label이라고 한 경우

    FN:
        실제 정답은 target_label인데,
        예측을 다른 클래스로 한 경우
    """

    tp = 0
    fp = 0
    fn = 0

    for true_label, pred_label in zip(y_true, y_pred):
        if true_label == target_label and pred_label == target_label:
            tp += 1

        elif true_label != target_label and pred_label == target_label:
            fp += 1

        elif true_label == target_label and pred_label != target_label:
            fn += 1

    return tp, fp, fn


def _calculate_micro_f1(
    y_true: List[Label],
    y_pred: List[Label],
    labels: List[Label]
) -> float:
    """
    micro F1을 계산한다.

    모든 클래스의 TP, FP, FN을 합쳐서 한 번에 F1을 계산한다.
    """

    total_tp = 0
    total_fp = 0
    total_fn = 0

    for label in labels:
        tp, fp, fn = _calculate_tp_fp_fn(y_true, y_pred, label)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    precision = _safe_divide(total_tp, total_tp + total_fp)
    recall = _safe_divide(total_tp, total_tp + total_fn)
    f1 = _safe_divide(2 * precision * recall, precision + recall)

    return f1


def _safe_divide(numerator: float, denominator: float) -> float:
    """
    0으로 나누는 상황을 방지하기 위한 함수.

    예:
    Precision = TP / (TP + FP)

    그런데 TP + FP가 0이면 계산할 수 없으므로 0.0을 반환한다.
    """

    if denominator == 0:
        return 0.0

    return numerator / denominator