import math
import re
from collections import Counter
from typing import Dict, List, Tuple

"""
사용 방법
from src.metrics.captioning_metrics import calculate_bleu_scores


candidates = {
    "img_001": "a dog is running on grass",
    "img_002": "a man is riding a bike"
}

references = {
    "img_001": [
        "a dog runs on the grass",
        "a dog is playing outside on the grass",
        "a brown dog is running in a field"
    ],
    "img_002": [
        "a man rides a bicycle",
        "a person is riding a bike",
        "a man is on a bike"
    ]
}

** 방법 1 **
bleu_scores = calculate_bleu_scores(
    candidates=candidates,
    references=references
)
print(bleu_scores)

** 방법 2 **
bleu_4 = calculate_bleu_n(
    candidates=candidates,
    references=references,
    n=4
)
print("BLEU-4:", bleu_4)
"""

def tokenize(sentence: str) -> List[str]:
    """
    문장을 단어 리스트로 변환한다.

    예:
    "A dog is running on grass."
    -> ["a", "dog", "is", "running", "on", "grass"]
    """

    sentence = sentence.lower()
    tokens = re.findall(r"\b\w+\b", sentence)

    return tokens


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    token 리스트에서 n-gram 개수를 계산한다.

    예:
    tokens = ["a", "dog", "is", "running"]

    n=1:
    ("a",), ("dog",), ("is",), ("running",)

    n=2:
    ("a", "dog"), ("dog", "is"), ("is", "running")
    """

    ngram_counts = Counter()

    if len(tokens) < n:
        return ngram_counts

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        ngram_counts[ngram] += 1

    return ngram_counts


def get_clipped_match_count(
    candidate_ngrams: Counter,
    reference_ngrams_list: List[Counter]
) -> int:
    """
    BLEU의 clipped count를 계산한다.

    clipped count란?
    - 모델 캡션에 어떤 단어가 너무 많이 반복되었을 때 점수를 과하게 주지 않기 위한 방식이다.

    예:
    모델 캡션:
    "dog dog dog dog"

    정답 캡션:
    "a dog is running"

    그냥 세면 dog가 4번 맞은 것처럼 보일 수 있다.
    하지만 정답에는 dog가 1번만 있으므로 최대 1번만 인정한다.
    """

    match_count = 0

    for ngram, candidate_count in candidate_ngrams.items():
        max_reference_count = 0

        for reference_ngrams in reference_ngrams_list:
            reference_count = reference_ngrams.get(ngram, 0)
            max_reference_count = max(max_reference_count, reference_count)

        match_count += min(candidate_count, max_reference_count)

    return match_count


def get_closest_reference_length(
    candidate_length: int,
    reference_lengths: List[int]
) -> int:
    """
    모델 캡션 길이와 가장 가까운 정답 캡션 길이를 찾는다.

    BLEU는 너무 짧은 문장에 높은 점수를 주지 않기 위해 brevity penalty를 사용한다.
    이때 모델 캡션 길이와 비교할 reference 길이가 필요하다.
    """

    closest_length = min(
        reference_lengths,
        key=lambda ref_len: (abs(ref_len - candidate_length), ref_len)
    )

    return closest_length


def calculate_brevity_penalty(
    total_candidate_length: int,
    total_reference_length: int
) -> float:
    """
    Brevity Penalty를 계산한다.

    모델 캡션이 정답 캡션보다 너무 짧으면 점수를 낮춘다.

    예:
    정답: "a dog is running on the grass"
    예측: "dog"

    단어 하나만 맞췄다고 높은 점수를 주면 안 되므로 패널티를 준다.
    """

    if total_candidate_length == 0:
        return 0.0

    if total_candidate_length > total_reference_length:
        return 1.0

    return math.exp(1 - (total_reference_length / total_candidate_length))


def calculate_bleu_n(
    candidates: Dict[str, str],
    references: Dict[str, List[str]],
    n: int = 4,
    smoothing: bool = True
) -> float:
    """
    BLEU-N 점수를 계산한다.

    Parameters
    ----------
    candidates:
        모델이 생성한 캡션.

        예:
        {
            "img_001": "a dog is running on grass",
            "img_002": "a man is riding a bike"
        }

    references:
        사람이 작성한 정답 캡션들.

        예:
        {
            "img_001": [
                "a dog runs on the grass",
                "a brown dog is running in a field"
            ],
            "img_002": [
                "a man rides a bicycle",
                "a person is riding a bike"
            ]
        }

    n:
        몇 gram까지 볼 것인지.
        - n=1: BLEU-1
        - n=2: BLEU-2
        - n=3: BLEU-3
        - n=4: BLEU-4

    smoothing:
        True이면 n-gram match가 0일 때 점수가 완전히 0이 되는 것을 완화한다.

    Returns
    -------
    bleu_score:
        BLEU 점수.
        0~1 사이 값.
        1에 가까울수록 정답 캡션과 유사하다.
    """

    if len(candidates) == 0:
        raise ValueError("candidates가 비어 있습니다.")

    if len(references) == 0:
        raise ValueError("references가 비어 있습니다.")

    if n < 1:
        raise ValueError("n은 1 이상이어야 합니다.")

    clipped_counts = {gram_order: 0 for gram_order in range(1, n + 1)}
    total_counts = {gram_order: 0 for gram_order in range(1, n + 1)}

    total_candidate_length = 0
    total_reference_length = 0

    for image_id, candidate_caption in candidates.items():
        if image_id not in references:
            raise ValueError(f"references에 {image_id}에 대한 정답 캡션이 없습니다.")

        candidate_tokens = tokenize(candidate_caption)
        reference_tokens_list = [
            tokenize(reference_caption)
            for reference_caption in references[image_id]
        ]

        total_candidate_length += len(candidate_tokens)

        reference_lengths = [
            len(reference_tokens)
            for reference_tokens in reference_tokens_list
        ]

        closest_reference_length = get_closest_reference_length(
            candidate_length=len(candidate_tokens),
            reference_lengths=reference_lengths
        )

        total_reference_length += closest_reference_length

        for gram_order in range(1, n + 1):
            candidate_ngrams = get_ngrams(candidate_tokens, gram_order)

            reference_ngrams_list = [
                get_ngrams(reference_tokens, gram_order)
                for reference_tokens in reference_tokens_list
            ]

            clipped_match_count = get_clipped_match_count(
                candidate_ngrams=candidate_ngrams,
                reference_ngrams_list=reference_ngrams_list
            )

            clipped_counts[gram_order] += clipped_match_count
            total_counts[gram_order] += sum(candidate_ngrams.values())

    brevity_penalty = calculate_brevity_penalty(
        total_candidate_length=total_candidate_length,
        total_reference_length=total_reference_length
    )

    log_precision_sum = 0.0

    for gram_order in range(1, n + 1):
        match_count = clipped_counts[gram_order]
        total_count = total_counts[gram_order]

        if total_count == 0:
            return 0.0

        if match_count == 0:
            if smoothing:
                precision = 1e-9 / total_count
            else:
                return 0.0
        else:
            precision = match_count / total_count

        log_precision_sum += (1 / n) * math.log(precision)

    bleu_score = brevity_penalty * math.exp(log_precision_sum)

    return bleu_score


def calculate_bleu_scores(
    candidates: Dict[str, str],
    references: Dict[str, List[str]],
    smoothing: bool = True
) -> Dict[str, float]:
    """
    BLEU-1, BLEU-2, BLEU-3, BLEU-4를 한 번에 계산한다.
    """

    return {
        "bleu_1": calculate_bleu_n(candidates, references, n=1, smoothing=smoothing),
        "bleu_2": calculate_bleu_n(candidates, references, n=2, smoothing=smoothing),
        "bleu_3": calculate_bleu_n(candidates, references, n=3, smoothing=smoothing),
        "bleu_4": calculate_bleu_n(candidates, references, n=4, smoothing=smoothing),
    }