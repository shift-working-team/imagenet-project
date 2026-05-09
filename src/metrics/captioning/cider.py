import math
import re
from collections import Counter, defaultdict
from typing import Dict, List, Tuple

"""
사용 방법
from src.metrics.cider import calculate_cider


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

result = calculate_cider(candidates, references)

print("전체 평균 CIDEr:", result["average_score"])
print("이미지별 CIDEr:", result["image_scores"])
"""

def tokenize(sentence: str) -> List[str]:
    """
    문장을 단어 리스트로 변환한다.

    예:
    "A dog is running." -> ["a", "dog", "is", "running"]

    주의:
    - 실제 COCO 공식 평가에서는 PTBTokenizer 같은 별도 토크나이저를 사용한다.
    - 여기서는 학생들이 이해하기 쉽도록 간단한 토크나이저를 사용한다.
    """
    sentence = sentence.lower()
    tokens = re.findall(r"\b\w+\b", sentence)
    return tokens


def get_ngrams(tokens: List[str], n: int) -> Counter:
    """
    토큰 리스트에서 n-gram 개수를 센다.

    예:
    tokens = ["a", "dog", "is", "running"]

    1-gram:
    ("a",), ("dog",), ("is",), ("running",)

    2-gram:
    ("a", "dog"), ("dog", "is"), ("is", "running")
    """
    counter = Counter()

    if len(tokens) < n:
        return counter

    for i in range(len(tokens) - n + 1):
        ngram = tuple(tokens[i:i + n])
        counter[ngram] += 1

    return counter


def build_document_frequency(
    references: Dict[str, List[str]],
    max_n: int = 4
) -> Dict[int, Counter]:
    """
    전체 reference caption을 기준으로 document frequency를 계산한다.

    document frequency란?
    - 어떤 n-gram이 전체 이미지 중 몇 개의 이미지에서 등장했는지 세는 값이다.

    중요한 점:
    - caption 개수가 아니라 image 개수를 기준으로 센다.
    - 같은 이미지 안의 여러 reference에 같은 n-gram이 여러 번 나와도 1번만 센다.
    """
    document_frequency = {n: Counter() for n in range(1, max_n + 1)}

    for image_id, ref_captions in references.items():
        for n in range(1, max_n + 1):
            unique_ngrams_for_image = set()

            for caption in ref_captions:
                tokens = tokenize(caption)
                ngrams = get_ngrams(tokens, n)
                unique_ngrams_for_image.update(ngrams.keys())

            for ngram in unique_ngrams_for_image:
                document_frequency[n][ngram] += 1

    return document_frequency


def counts_to_tfidf(
    ngram_counts: Counter,
    document_frequency: Counter,
    num_images: int
) -> Dict[Tuple[str, ...], float]:
    """
    n-gram count를 TF-IDF 벡터로 변환한다.

    TF:
    - 한 문장 안에서 해당 n-gram이 얼마나 자주 나왔는지

    IDF:
    - 전체 데이터셋에서 너무 흔한 표현은 점수를 낮춘다.
    - 드문 표현은 더 중요한 표현으로 본다.

    예:
    "a", "the" 같은 흔한 단어는 중요도가 낮다.
    "golden retriever", "red bus" 같은 구체적인 표현은 중요도가 높을 수 있다.
    """
    tfidf_vector = {}

    total_count = sum(ngram_counts.values())
    if total_count == 0:
        return tfidf_vector

    for ngram, count in ngram_counts.items():
        tf = count / total_count

        # reference에 없는 n-gram이면 df를 1로 처리한다.
        # 그래야 division by zero를 피할 수 있다.
        df = document_frequency.get(ngram, 0)
        df = max(df, 1)

        idf = math.log(num_images / df)

        tfidf_vector[ngram] = tf * idf

    return tfidf_vector


def cosine_similarity(
    vector_a: Dict[Tuple[str, ...], float],
    vector_b: Dict[Tuple[str, ...], float]
) -> float:
    """
    두 TF-IDF 벡터의 cosine similarity를 계산한다.

    결과:
    - 1에 가까울수록 비슷함
    - 0에 가까울수록 다름
    """
    if not vector_a or not vector_b:
        return 0.0

    dot_product = 0.0

    for key, value in vector_a.items():
        dot_product += value * vector_b.get(key, 0.0)

    norm_a = math.sqrt(sum(value ** 2 for value in vector_a.values()))
    norm_b = math.sqrt(sum(value ** 2 for value in vector_b.values()))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


def calculate_cider(
    candidates: Dict[str, str],
    references: Dict[str, List[str]],
    max_n: int = 4
) -> Dict[str, object]:
    """
    CIDEr 점수를 계산한다.

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
                "a dog is playing outside"
            ],
            "img_002": [
                "a man rides a bicycle",
                "a person is riding a bike"
            ]
        }

    max_n:
        몇 gram까지 볼 것인지.
        CIDEr는 보통 1-gram부터 4-gram까지 사용한다.

    Returns
    -------
    {
        "average_score": 전체 이미지 평균 CIDEr 점수,
        "image_scores": 이미지별 CIDEr 점수
    }
    """
    num_images = len(references)

    if num_images == 0:
        raise ValueError("references가 비어 있습니다.")

    document_frequency = build_document_frequency(references, max_n=max_n)

    image_scores = {}

    for image_id, candidate_caption in candidates.items():
        if image_id not in references:
            raise ValueError(f"references에 {image_id}에 대한 정답 캡션이 없습니다.")

        candidate_tokens = tokenize(candidate_caption)
        reference_captions = references[image_id]

        n_scores = []

        for n in range(1, max_n + 1):
            candidate_counts = get_ngrams(candidate_tokens, n)

            candidate_vector = counts_to_tfidf(
                candidate_counts,
                document_frequency[n],
                num_images
            )

            reference_scores = []

            for reference_caption in reference_captions:
                reference_tokens = tokenize(reference_caption)
                reference_counts = get_ngrams(reference_tokens, n)

                reference_vector = counts_to_tfidf(
                    reference_counts,
                    document_frequency[n],
                    num_images
                )

                similarity = cosine_similarity(candidate_vector, reference_vector)
                reference_scores.append(similarity)

            if reference_scores:
                n_score = sum(reference_scores) / len(reference_scores)
            else:
                n_score = 0.0

            n_scores.append(n_score)

        # CIDEr는 보통 1~4 gram 점수 평균에 10을 곱한다.
        cider_score = 10.0 * sum(n_scores) / max_n

        image_scores[image_id] = cider_score

    average_score = sum(image_scores.values()) / len(image_scores)

    return {
        "average_score": average_score,
        "image_scores": image_scores
    }