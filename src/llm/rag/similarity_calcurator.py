# flake8: noqa: E501

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import (
    cosine_similarity,
    euclidean_distances,
    manhattan_distances,
)


class SimilarityCalculator:
    """다양한 유사도 계산 방법을 구현한 클래스"""

    @staticmethod
    def cosine_similarity_manual(vector_a, vector_b):
        """
        코사인 유사도 수동 계산
        cos(θ) = (A · B) / (||A|| × ||B||)
        """
        # 내적 계산
        dot_product = np.dot(vector_a, vector_b)

        # 벡터의 크기(norm) 계산
        norm_a = np.linalg.norm(vector_a)
        norm_b = np.linalg.norm(vector_b)

        # 코사인 유사도 계산
        if norm_a == 0 or norm_b == 0:
            return 0

        cosine_sim = dot_product / (norm_a * norm_b)
        return cosine_sim

    @staticmethod
    def euclidean_distance_manual(vector_a, vector_b):
        """
        유클리드 거리 수동 계산
        distance = sqrt(Σ(ai - bi)²)
        """
        return np.sqrt(np.sum((vector_a - vector_b) ** 2))

    @staticmethod
    def manhattan_distance_manual(vector_a, vector_b):
        """
        맨하탄 거리 수동 계산
        distance = Σ|ai - bi|
        """
        return np.sum(np.abs(vector_a - vector_b))

    @staticmethod
    def euclidean_to_similarity(distance, max_distance=None):
        """유클리드 거리를 유사도로 변환 (0~1 범위)"""
        if max_distance is None:
            # 간단한 변환: 1 / (1 + distance)
            return 1 / (1 + distance)
        else:
            # 정규화된 변환: 1 - (distance / max_distance)
            return 1 - (distance / max_distance)

    @staticmethod
    def manhattan_to_similarity(distance, max_distance=None):
        """맨하탄 거리를 유사도로 변환 (0~1 범위)"""
        if max_distance is None:
            return 1 / (1 + distance)
        else:
            return 1 - (distance / max_distance)


def create_sample_vectors():
    """예시 벡터 데이터 생성"""
    # 문서 임베딩을 시뮬레이션한 벡터들
    vectors = {
        "인공지능 기계학습": np.array([0.8, 0.6, 0.2, 0.9, 0.1]),
        "AI 딥러닝": np.array([0.7, 0.8, 0.1, 0.8, 0.2]),
        "자연어처리 NLP": np.array([0.6, 0.7, 0.3, 0.7, 0.3]),
        "컴퓨터비전 이미지": np.array([0.5, 0.4, 0.8, 0.6, 0.7]),
        "요리 레시피": np.array([0.1, 0.2, 0.1, 0.3, 0.9]),
        "스포츠 축구": np.array([0.2, 0.1, 0.9, 0.2, 0.8]),
    }
    return vectors


def test_similarity_methods():
    """다양한 유사도 계산 방법 테스트"""
    print("=" * 60)
    print("유사도 계산 방법 비교 테스트")
    print("=" * 60)

    # 예시 벡터 생성
    vectors = create_sample_vectors()
    vector_names = list(vectors.keys())

    calc = SimilarityCalculator()

    # 기준 벡터 (첫 번째 벡터)
    base_name = vector_names[0]
    base_vector = vectors[base_name]

    print(f"\n기준 벡터: '{base_name}'")
    print(f"벡터 값: {base_vector}")
    print("\n다른 벡터들과의 유사도 비교:")
    print("-" * 80)

    results = []

    for name, vector in vectors.items():
        if name == base_name:
            continue

        # 1. 코사인 유사도 계산
        cos_sim_manual = calc.cosine_similarity_manual(base_vector, vector)
        cos_sim_sklearn = cosine_similarity([base_vector], [vector])[0][0]

        # 2. 유클리드 거리 계산
        euclidean_dist_manual = calc.euclidean_distance_manual(base_vector, vector)
        euclidean_dist_sklearn = euclidean_distances([base_vector], [vector])[0][0]
        euclidean_sim = calc.euclidean_to_similarity(euclidean_dist_manual)

        # 3. 맨하탄 거리 계산
        manhattan_dist_manual = calc.manhattan_distance_manual(base_vector, vector)
        manhattan_dist_sklearn = manhattan_distances([base_vector], [vector])[0][0]
        manhattan_sim = calc.manhattan_to_similarity(manhattan_dist_manual)

        results.append(
            {
                "Document": name,
                "Cosine_Similarity": cos_sim_manual,
                "Euclidean_Distance": euclidean_dist_manual,
                "Euclidean_Similarity": euclidean_sim,
                "Manhattan_Distance": manhattan_dist_manual,
                "Manhattan_Similarity": manhattan_sim,
            }
        )

        print(f"벡터: '{name}'")
        print(
            f"  코사인 유사도:     {cos_sim_manual:.4f} (sklearn: {cos_sim_sklearn:.4f})"
        )
        print(
            f"  유클리드 거리:     {euclidean_dist_manual:.4f} (sklearn: {euclidean_dist_sklearn:.4f})"
        )
        print(f"  유클리드 유사도:   {euclidean_sim:.4f}")
        print(
            f"  맨하탄 거리:       {manhattan_dist_manual:.4f} (sklearn: {manhattan_dist_sklearn:.4f})"
        )
        print(f"  맨하탄 유사도:     {manhattan_sim:.4f}")
        print()

    return results


def visualize_similarities(results):
    """유사도 결과 시각화"""
    df = pd.DataFrame(results)

    # 그래프 설정
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle("다양한 유사도 계산 방법 비교", fontsize=16, fontweight="bold")

    # 1. 코사인 유사도
    axes[0, 0].bar(range(len(df)), df["Cosine_Similarity"], color="skyblue", alpha=0.7)
    axes[0, 0].set_title("코사인 유사도 (Cosine Similarity)")
    axes[0, 0].set_ylabel("유사도 점수")
    axes[0, 0].set_ylim(0, 1)
    axes[0, 0].set_xticks(range(len(df)))
    axes[0, 0].set_xticklabels(
        [doc[:10] + "..." for doc in df["Document"]], rotation=45
    )

    # 2. 유클리드 거리
    axes[0, 1].bar(
        range(len(df)), df["Euclidean_Distance"], color="lightcoral", alpha=0.7
    )
    axes[0, 1].set_title("유클리드 거리 (Euclidean Distance)")
    axes[0, 1].set_ylabel("거리")
    axes[0, 1].set_xticks(range(len(df)))
    axes[0, 1].set_xticklabels(
        [doc[:10] + "..." for doc in df["Document"]], rotation=45
    )

    # 3. 맨하탄 거리
    axes[1, 0].bar(
        range(len(df)), df["Manhattan_Distance"], color="lightgreen", alpha=0.7
    )
    axes[1, 0].set_title("맨하탄 거리 (Manhattan Distance)")
    axes[1, 0].set_ylabel("거리")
    axes[1, 0].set_xticks(range(len(df)))
    axes[1, 0].set_xticklabels(
        [doc[:10] + "..." for doc in df["Document"]], rotation=45
    )

    # 4. 모든 유사도 비교
    x = range(len(df))
    width = 0.25

    axes[1, 1].bar(
        [i - width for i in x],
        df["Cosine_Similarity"],
        width,
        label="코사인 유사도",
        alpha=0.7,
    )
    axes[1, 1].bar(
        x, df["Euclidean_Similarity"], width, label="유클리드 유사도", alpha=0.7
    )
    axes[1, 1].bar(
        [i + width for i in x],
        df["Manhattan_Similarity"],
        width,
        label="맨하탄 유사도",
        alpha=0.7,
    )

    axes[1, 1].set_title("유사도 방법 비교")
    axes[1, 1].set_ylabel("유사도 점수")
    axes[1, 1].set_ylim(0, 1)
    axes[1, 1].set_xticks(range(len(df)))
    axes[1, 1].set_xticklabels(
        [doc[:10] + "..." for doc in df["Document"]], rotation=45
    )
    axes[1, 1].legend()

    plt.tight_layout()
    plt.show()


def demonstrate_vector_operations():
    """벡터 연산 시연"""
    print("\n" + "=" * 60)
    print("벡터 연산 기본 원리 시연")
    print("=" * 60)

    # 간단한 2D 벡터로 시연
    vec_a = np.array([3, 4])
    vec_b = np.array([1, 2])

    print(f"벡터 A: {vec_a}")
    print(f"벡터 B: {vec_b}")

    calc = SimilarityCalculator()

    # 내적 계산
    dot_product = np.dot(vec_a, vec_b)
    print(f"\n내적 (A · B): {dot_product}")

    # 벡터 크기 계산
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    print(f"벡터 A의 크기: {norm_a:.4f}")
    print(f"벡터 B의 크기: {norm_b:.4f}")

    # 각 방법별 계산
    cos_sim = calc.cosine_similarity_manual(vec_a, vec_b)
    euclidean_dist = calc.euclidean_distance_manual(vec_a, vec_b)
    manhattan_dist = calc.manhattan_distance_manual(vec_a, vec_b)

    print(f"\n코사인 유사도: {cos_sim:.4f}")
    print(f"유클리드 거리: {euclidean_dist:.4f}")
    print(f"맨하탄 거리: {manhattan_dist:.4f}")

    # 각도 계산
    angle_radians = np.arccos(cos_sim)
    angle_degrees = np.degrees(angle_radians)
    print(f"두 벡터 간 각도: {angle_degrees:.2f}도")


def create_similarity_matrix():
    """모든 벡터 간 유사도 매트릭스 생성"""
    print("\n" + "=" * 60)
    print("전체 유사도 매트릭스")
    print("=" * 60)

    vectors = create_sample_vectors()
    vector_names = list(vectors.keys())
    n = len(vectors)

    calc = SimilarityCalculator()

    # 유사도 매트릭스 초기화
    cosine_matrix = np.zeros((n, n))
    euclidean_matrix = np.zeros((n, n))
    manhattan_matrix = np.zeros((n, n))

    # 모든 벡터 쌍에 대해 유사도 계산
    for i, name_i in enumerate(vector_names):
        for j, name_j in enumerate(vector_names):
            vec_i = vectors[name_i]
            vec_j = vectors[name_j]

            cosine_matrix[i][j] = calc.cosine_similarity_manual(vec_i, vec_j)
            euclidean_dist = calc.euclidean_distance_manual(vec_i, vec_j)
            euclidean_matrix[i][j] = calc.euclidean_to_similarity(euclidean_dist)
            manhattan_dist = calc.manhattan_distance_manual(vec_i, vec_j)
            manhattan_matrix[i][j] = calc.manhattan_to_similarity(manhattan_dist)

    # 결과 출력
    print("코사인 유사도 매트릭스:")
    short_names = [name[:15] for name in vector_names]
    df_cosine = pd.DataFrame(cosine_matrix, index=short_names, columns=short_names)
    print(df_cosine.round(3))

    print("\n유클리드 유사도 매트릭스:")
    df_euclidean = pd.DataFrame(
        euclidean_matrix, index=short_names, columns=short_names
    )
    print(df_euclidean.round(3))

    return cosine_matrix, euclidean_matrix, manhattan_matrix


# 실행 함수
def main():
    """메인 실행 함수"""
    print("벡터 유사도 계산 데모 프로그램")
    print("=" * 60)

    # 1. 기본 벡터 연산 시연
    demonstrate_vector_operations()

    # 2. 다양한 유사도 방법 테스트
    results = test_similarity_methods()

    # 3. 유사도 매트릭스 생성
    cosine_matrix, euclidean_matrix, manhattan_matrix = create_similarity_matrix()

    # 4. 시각화 (matplotlib 사용 가능한 환경에서)
    try:
        visualize_similarities(results)
    except Exception as e:
        print(f"\n시각화를 건너뜁니다: {e}")

    # 5. 요약 및 해석
    print("\n" + "=" * 60)
    print("결과 해석 및 요약")
    print("=" * 60)
    print("1. 코사인 유사도: 벡터 간 각도 기반, 방향성 중요")
    print("2. 유클리드 거리: 공간상 직선 거리, 크기와 방향 모두 고려")
    print("3. 맨하탄 거리: 각 차원별 차이의 합, 특정 차원의 영향 명확")
    print("\n각 방법은 데이터의 특성과 용도에 따라 적절히 선택해야 합니다.")


if __name__ == "__main__":
    main()
