import numpy as np


def cosine_similarity(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    두 벡터 간의 코사인 유사도 계산 (기본 버전)

    Args:
        vector_a: 첫 번째 벡터
        vector_b: 두 번째 벡터

    Returns:
        코사인 유사도 값 (-1 ~ 1)
    """
    # 벡터의 내적 계산
    dot_product = np.dot(vector_a, vector_b)

    # 각 벡터의 크기(norm) 계산
    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    # 0으로 나누는 것을 방지
    if norm_a == 0 or norm_b == 0:
        return 0.0

    # 코사인 유사도 = 내적 / (크기_a * 크기_b)
    similarity = dot_product / (norm_a * norm_b)

    return similarity


def cosine_similarity_matrix(
    matrix_a: np.ndarray, matrix_b: np.ndarray = None
) -> np.ndarray:
    """
    행렬 간의 코사인 유사도 계산 (배치 처리)

    Args:
        matrix_a: 첫 번째 행렬 (n_samples_a, n_features)
        matrix_b: 두 번째 행렬 (n_samples_b, n_features). None이면 matrix_a와 자기 자신

    Returns:
        유사도 행렬 (n_samples_a, n_samples_b)
    """
    if matrix_b is None:
        matrix_b = matrix_a

    # 각 행을 단위 벡터로 정규화
    norm_a = np.linalg.norm(matrix_a, axis=1, keepdims=True)
    norm_b = np.linalg.norm(matrix_b, axis=1, keepdims=True)

    # 0으로 나누는 것을 방지
    norm_a = np.where(norm_a == 0, 1, norm_a)
    norm_b = np.where(norm_b == 0, 1, norm_b)

    # 정규화된 벡터
    normalized_a = matrix_a / norm_a
    normalized_b = matrix_b / norm_b

    # 행렬 곱셈으로 모든 쌍의 코사인 유사도 계산
    similarity_matrix = np.dot(normalized_a, normalized_b.T)

    return similarity_matrix


def cosine_similarity_optimized(vector_a: np.ndarray, vector_b: np.ndarray) -> float:
    """
    최적화된 코사인 유사도 계산 (벡터화 연산 활용)
    """
    # 정규화를 먼저 수행
    norm_a = np.sqrt(np.sum(vector_a**2))
    norm_b = np.sqrt(np.sum(vector_b**2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    # 정규화된 벡터들의 내적
    return np.sum((vector_a / norm_a) * (vector_b / norm_b))


def find_top_k_similar(
    query_vector: np.ndarray, document_vectors: np.ndarray, k: int = 5
) -> tuple:
    """
    쿼리 벡터와 가장 유사한 k개 문서 찾기

    Args:
        query_vector: 쿼리 벡터 (1차원)
        document_vectors: 문서 벡터들 (2차원: n_docs, n_features)
        k: 반환할 상위 문서 수

    Returns:
        (인덱스 배열, 유사도 배열)
    """
    # 쿼리 벡터를 2차원으로 변환 (1, n_features)
    query_vector_2d = query_vector.reshape(1, -1)

    # 모든 문서와의 유사도 계산
    similarities = cosine_similarity_matrix(query_vector_2d, document_vectors).flatten()

    # 상위 k개 인덱스 찾기 (내림차순)
    top_k_indices = np.argsort(similarities)[::-1][:k]
    top_k_similarities = similarities[top_k_indices]

    return top_k_indices, top_k_similarities


def cosine_similarity_sparse(
    vector_a: np.ndarray, vector_b: np.ndarray, threshold: float = 0.01
) -> float:
    """
    희소 벡터를 위한 효율적인 코사인 유사도 계산
    작은 값들을 0으로 처리하여 계산 효율성 향상
    """
    # 임계값 이하의 값들을 0으로 설정
    vector_a_sparse = np.where(np.abs(vector_a) < threshold, 0, vector_a)
    vector_b_sparse = np.where(np.abs(vector_b) < threshold, 0, vector_b)

    # 0이 아닌 요소들만 고려한 유사도 계산
    nonzero_a = vector_a_sparse != 0
    nonzero_b = vector_b_sparse != 0
    common_indices = nonzero_a & nonzero_b

    if not np.any(common_indices):
        return 0.0

    # 공통 인덱스에서만 계산
    dot_product = np.sum(
        vector_a_sparse[common_indices] * vector_b_sparse[common_indices]
    )
    norm_a = np.sqrt(np.sum(vector_a_sparse**2))
    norm_b = np.sqrt(np.sum(vector_b_sparse**2))

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return dot_product / (norm_a * norm_b)


# 사용 예시 및 테스트
def demo_cosine_similarity():
    """코사인 유사도 함수들 사용 예시"""

    print("=" * 50)
    print("코사인 유사도 계산 예시")
    print("=" * 50)

    # 1. 기본 벡터 예시
    print("\n1. 기본 벡터 유사도 계산")
    vector1 = np.array([1, 2, 3, 4])
    vector2 = np.array([2, 4, 6, 8])  # vector1의 2배
    vector3 = np.array([1, 0, 0, 0])  # 전혀 다른 방향

    sim12 = cosine_similarity(vector1, vector2)
    sim13 = cosine_similarity(vector1, vector3)

    print(f"벡터1: {vector1}")
    print(f"벡터2: {vector2}")
    print(f"벡터3: {vector3}")
    print(f"벡터1 vs 벡터2 유사도: {sim12:.4f}")  # 1.0에 가까움 (방향이 같음)
    print(f"벡터1 vs 벡터3 유사도: {sim13:.4f}")  # 낮은 값

    # 2. 행렬 유사도 계산
    print("\n2. 행렬 유사도 계산")
    matrix_docs = np.array(
        [
            [1, 2, 3],  # 문서1
            [2, 4, 6],  # 문서2 (문서1과 유사)
            [1, 0, 0],  # 문서3
            [0, 1, 2],  # 문서4
        ]
    )

    # 모든 문서 간 유사도 계산
    similarity_matrix = cosine_similarity_matrix(matrix_docs)
    print(f"문서 행렬:\n{matrix_docs}")
    print(f"유사도 행렬:\n{similarity_matrix}")

    # 3. 상위 K개 유사 문서 찾기
    print("\n3. 상위 K개 유사 문서 찾기")
    query = np.array([1, 2, 3])
    top_indices, top_similarities = find_top_k_similar(query, matrix_docs, k=3)

    print(f"쿼리 벡터: {query}")
    print("상위 3개 유사 문서:")
    for i, (idx, sim) in enumerate(zip(top_indices, top_similarities)):
        print(f"  {i + 1}위: 문서{idx} (유사도: {sim:.4f})")

    # 4. 텍스트 벡터 시뮬레이션 (TF-IDF 스타일)
    print("\n4. 텍스트 벡터 시뮬레이션")
    # 가상의 TF-IDF 벡터들 (단어: ['cat', 'dog', 'animal', 'pet', 'cute'])
    text_vectors = np.array(
        [
            [0.8, 0.1, 0.5, 0.3, 0.2],  # "고양이는 귀여운 애완동물"
            [0.1, 0.9, 0.4, 0.4, 0.1],  # "강아지는 충성스러운 애완동물"
            [0.2, 0.2, 0.8, 0.1, 0.0],  # "동물원에는 많은 동물들"
            [0.0, 0.0, 0.1, 0.0, 0.9],  # "귀여운 아기"
        ]
    )

    query_text = np.array([0.5, 0.5, 0.3, 0.4, 0.2])  # "귀여운 강아지와 고양이"

    text_similarities = cosine_similarity_matrix(
        query_text.reshape(1, -1), text_vectors
    ).flatten()

    print(f"쿼리 텍스트 벡터: {query_text}")
    print("텍스트 유사도:")
    texts = ["고양이 문서", "강아지 문서", "동물원 문서", "아기 문서"]
    for i, (text, sim) in enumerate(zip(texts, text_similarities)):
        print(f"  {text}: {sim:.4f}")

    # 5. 성능 비교
    print("\n5. 성능 테스트")
    large_vector1 = np.random.rand(1000)
    large_vector2 = np.random.rand(1000)

    import time

    # 기본 버전
    start = time.time()
    sim_basic = cosine_similarity(large_vector1, large_vector2)
    time_basic = time.time() - start

    # 최적화 버전
    start = time.time()
    sim_optimized = cosine_similarity_optimized(large_vector1, large_vector2)
    time_optimized = time.time() - start

    print(f"기본 버전: {sim_basic:.6f} ({time_basic:.6f}초)")
    print(f"최적화 버전: {sim_optimized:.6f} ({time_optimized:.6f}초)")
    print(f"결과 차이: {abs(sim_basic - sim_optimized):.10f}")


if __name__ == "__main__":
    demo_cosine_similarity()
