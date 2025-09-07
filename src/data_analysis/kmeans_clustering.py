import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

plt.rcParams["font.family"] = "AppleGothic"  # MacOS


class KMeans:
    """
    K-means Clustering 구현 클래스
    라이브러리 없이 pandas와 numpy만 사용
    """

    def __init__(self, k=3, max_iters=100, random_state=None):
        """
        Parameters:
        - k: 클러스터 개수
        - max_iters: 최대 반복 횟수
        - random_state: 랜덤 시드
        """
        self.k = k
        self.max_iters = max_iters
        self.random_state = random_state

        # 결과 저장용
        self.centroids = None
        self.labels = None
        self.history = []  # 학습 과정 기록

    def _initialize_centroids(self, X):
        """중심점 초기화"""
        if self.random_state:
            np.random.seed(self.random_state)

        n_samples, n_features = X.shape

        # 방법 1: 랜덤 초기화
        # centroids = np.random.uniform(X.min(), X.max(), (self.k, n_features))

        # 방법 2: 데이터 범위 내에서 랜덤 선택 (더 안정적)
        min_vals = np.min(X, axis=0)
        max_vals = np.max(X, axis=0)
        centroids = np.random.uniform(min_vals, max_vals, (self.k, n_features))

        return centroids

    def _calculate_distance(self, X, centroids):
        """각 데이터 포인트와 중심점 간의 유클리드 거리 계산"""
        distances = np.zeros((X.shape[0], self.k))

        for i, centroid in enumerate(centroids):
            # 유클리드 거리: sqrt(sum((x - centroid)^2))
            distances[:, i] = np.sqrt(np.sum((X - centroid) ** 2, axis=1))

        return distances

    def _assign_clusters(self, distances):
        """가장 가까운 중심점에 데이터 포인트 할당"""
        return np.argmin(distances, axis=1)

    def _update_centroids(self, X, labels):
        """각 클러스터의 중심점 업데이트"""
        new_centroids = np.zeros((self.k, X.shape[1]))

        for i in range(self.k):
            # 클러스터 i에 속하는 데이터들의 평균으로 중심점 업데이트
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                new_centroids[i] = np.mean(cluster_points, axis=0)
            else:
                # 빈 클러스터인 경우 기존 중심점 유지
                new_centroids[i] = self.centroids[i]

        return new_centroids

    def _calculate_wcss(self, X, labels, centroids):
        """WCSS (Within-Cluster Sum of Squares) 계산"""
        wcss = 0
        for i in range(self.k):
            cluster_points = X[labels == i]
            if len(cluster_points) > 0:
                wcss += np.sum((cluster_points - centroids[i]) ** 2)
        return wcss

    def fit(self, X):
        """K-means 알고리즘 실행"""
        # pandas DataFrame을 numpy array로 변환
        if isinstance(X, pd.DataFrame):
            X = X.values

        # 중심점 초기화
        self.centroids = self._initialize_centroids(X)

        prev_centroids = None

        for iteration in range(self.max_iters):
            # 1. 거리 계산
            distances = self._calculate_distance(X, self.centroids)

            # 2. 클러스터 할당
            self.labels = self._assign_clusters(distances)

            # 3. 중심점 업데이트
            prev_centroids = self.centroids.copy()
            self.centroids = self._update_centroids(X, self.labels)

            # 4. WCSS 계산
            wcss = self._calculate_wcss(X, self.labels, self.centroids)

            # 5. 수렴 체크 및 기록
            centroid_shift = np.sum(
                np.sqrt(np.sum((self.centroids - prev_centroids) ** 2, axis=1))
            )

            self.history.append(
                {
                    "iteration": iteration,
                    "wcss": wcss,
                    "centroid_shift": centroid_shift,
                    "centroids": self.centroids.copy(),
                    "labels": self.labels.copy(),
                }
            )

            # 수렴 조건: 중심점 변화가 매우 작을 때
            if centroid_shift < 1e-6:
                print(f"수렴 완료: {iteration + 1}번째 반복에서 중심점 변화 < 1e-6")
                break
        else:
            print(f"최대 반복 횟수 {self.max_iters}에 도달")

        return self

    def predict(self, X):
        """새로운 데이터에 대해 클러스터 예측"""
        if isinstance(X, pd.DataFrame):
            X = X.values

        distances = self._calculate_distance(X, self.centroids)
        return self._assign_clusters(distances)

    def get_cluster_info(self):
        """클러스터 정보 반환"""
        if self.labels is None:
            return None

        info = []
        for i in range(self.k):
            cluster_size = np.sum(self.labels == i)
            info.append(
                {
                    "cluster": i,
                    "size": cluster_size,
                    "centroid": self.centroids[i],
                    "percentage": cluster_size / len(self.labels) * 100,
                }
            )

        return pd.DataFrame(info)


def generate_sample_data(n_samples=300, random_state=42):
    """샘플 데이터 생성"""
    np.random.seed(random_state)

    # 3개의 클러스터를 가진 샘플 데이터 생성
    cluster1 = np.random.normal([2, 2], [0.5, 0.5], (n_samples // 3, 2))
    cluster2 = np.random.normal([6, 6], [0.8, 0.8], (n_samples // 3, 2))
    cluster3 = np.random.normal([2, 6], [0.6, 0.6], (n_samples // 3, 2))

    data = np.vstack([cluster1, cluster2, cluster3])

    # DataFrame으로 변환
    df = pd.DataFrame(data, columns=["feature_1", "feature_2"])

    return df


def plot_clusters(X, labels, centroids, title="K-means Clustering Results"):
    """클러스터링 결과 시각화"""
    if isinstance(X, pd.DataFrame):
        X = X.values

    plt.figure(figsize=(10, 8))

    # 색상 맵 설정
    colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

    # 각 클러스터별로 점 그리기
    for i in range(len(np.unique(labels))):
        cluster_points = X[labels == i]
        plt.scatter(
            cluster_points[:, 0],
            cluster_points[:, 1],
            c=colors[i],
            alpha=0.6,
            s=50,
            label=f"Cluster {i}",
        )

    # 중심점 그리기
    plt.scatter(
        centroids[:, 0],
        centroids[:, 1],
        c="black",
        marker="x",
        s=300,
        linewidths=3,
        label="Centroids",
    )

    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.show()


def elbow_method(X, max_k=10):
    """엘보우 방법으로 최적의 k 찾기"""
    if isinstance(X, pd.DataFrame):
        X_array = X.values
    else:
        X_array = X

    wcss_values = []
    k_values = range(1, max_k + 1)

    for k in k_values:
        if k == 1:
            # k=1일 때는 전체 데이터의 중심점으로부터의 거리
            centroid = np.mean(X_array, axis=0)
            wcss = np.sum((X_array - centroid) ** 2)
        else:
            kmeans = KMeans(k=k, random_state=42)
            kmeans.fit(X_array)
            wcss = kmeans.history[-1]["wcss"]

        wcss_values.append(wcss)

    # 엘보우 그래프 그리기
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, wcss_values, "bo-")
    plt.title("Elbow Method for Optimal k")
    plt.xlabel("Number of Clusters (k)")
    plt.ylabel("WCSS (Within-Cluster Sum of Squares)")
    plt.grid(True, alpha=0.3)
    plt.show()

    return list(zip(k_values, wcss_values))


def analyze_convergence(kmeans_model):
    """수렴 과정 분석"""
    if not kmeans_model.history:
        print("학습 기록이 없습니다.")
        return

    iterations = [h["iteration"] for h in kmeans_model.history]
    wcss_values = [h["wcss"] for h in kmeans_model.history]
    shifts = [h["centroid_shift"] for h in kmeans_model.history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # WCSS 변화
    ax1.plot(iterations, wcss_values, "b-o")
    ax1.set_title("WCSS vs Iterations")
    ax1.set_xlabel("Iteration")
    ax1.set_ylabel("WCSS")
    ax1.grid(True, alpha=0.3)

    # 중심점 이동 거리
    ax2.plot(iterations, shifts, "r-o")
    ax2.set_title("Centroid Shift vs Iterations")
    ax2.set_xlabel("Iteration")
    ax2.set_ylabel("Centroid Shift Distance")
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()


def main():
    print("=== K-means Clustering from Scratch ===\n")

    # 1. 샘플 데이터 생성
    print("1. 샘플 데이터 생성")
    print("-" * 40)
    data = generate_sample_data(n_samples=300, random_state=42)
    print("데이터 형태:", data.shape)
    print("데이터 미리보기:")
    print(data.head())
    print()

    print("데이터 기본 통계:")
    print(data.describe())
    print()

    # 2. 원본 데이터 시각화
    print("2. 원본 데이터 시각화")
    plt.figure(figsize=(8, 6))
    plt.scatter(data["feature_1"], data["feature_2"], alpha=0.6)
    plt.title("Original Data")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    plt.grid(True, alpha=0.3)
    plt.show()

    # 3. 엘보우 방법으로 최적 k 찾기
    print("3. 엘보우 방법으로 최적 k 찾기")
    print("-" * 40)
    elbow_results = elbow_method(data, max_k=8)
    print("k별 WCSS 값:")
    for k, wcss in elbow_results:
        print(f"k={k}: WCSS={wcss:.2f}")
    print()

    # 4. K-means 실행 (k=3)
    print("4. K-means 클러스터링 실행 (k=3)")
    print("-" * 40)
    kmeans = KMeans(k=3, max_iters=100, random_state=42)
    kmeans.fit(data)

    print("최종 중심점:")
    for i, centroid in enumerate(kmeans.centroids):
        print(f"클러스터 {i}: ({centroid[0]:.3f}, {centroid[1]:.3f})")
    print()

    # 5. 클러스터링 결과 분석
    print("5. 클러스터링 결과 분석")
    print("-" * 40)
    cluster_info = kmeans.get_cluster_info()
    print(cluster_info)
    print()

    # 6. 결과 시각화
    print("6. 클러스터링 결과 시각화")
    plot_clusters(data, kmeans.labels, kmeans.centroids)

    # 7. 수렴 과정 분석
    print("7. 수렴 과정 분석")
    analyze_convergence(kmeans)

    # 8. 새로운 데이터 예측
    print("8. 새로운 데이터 예측")
    print("-" * 40)
    new_data = pd.DataFrame(
        {"feature_1": [1.5, 5.5, 2.5], "feature_2": [2.0, 6.0, 5.5]}
    )

    predictions = kmeans.predict(new_data)
    print("새로운 데이터:")
    print(new_data)
    print("예측된 클러스터:", predictions)
    print()

    # 9. 다른 k 값들과 비교
    print("9. 다른 k 값들과 비교")
    print("-" * 40)

    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    k_values = [2, 3, 4, 5]

    for idx, k in enumerate(k_values):
        row = idx // 2
        col = idx % 2

        kmeans_test = KMeans(k=k, random_state=42)
        kmeans_test.fit(data)

        if isinstance(data, pd.DataFrame):
            X_plot = data.values
        else:
            X_plot = data

        colors = ["red", "blue", "green", "purple", "orange"]

        for i in range(k):
            cluster_points = X_plot[kmeans_test.labels == i]
            axes[row, col].scatter(
                cluster_points[:, 0], cluster_points[:, 1], c=colors[i], alpha=0.6, s=30
            )

        axes[row, col].scatter(
            kmeans_test.centroids[:, 0],
            kmeans_test.centroids[:, 1],
            c="black",
            marker="x",
            s=200,
            linewidths=2,
        )
        axes[row, col].set_title(f"k = {k}")
        axes[row, col].set_xlabel("Feature 1")
        axes[row, col].set_ylabel("Feature 2")
        axes[row, col].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    # 10. 실제 데이터셋 예제 (iris 형태의 데이터)
    print("10. 복잡한 데이터셋 예제")
    print("-" * 40)

    # 더 복잡한 데이터 생성
    np.random.seed(100)
    complex_data = pd.DataFrame(
        {
            "sepal_length": np.concatenate(
                [
                    np.random.normal(5.0, 0.5, 50),  # 클러스터 1
                    np.random.normal(6.5, 0.6, 50),  # 클러스터 2
                    np.random.normal(7.5, 0.4, 50),  # 클러스터 3
                ]
            ),
            "sepal_width": np.concatenate(
                [
                    np.random.normal(3.5, 0.4, 50),
                    np.random.normal(3.0, 0.5, 50),
                    np.random.normal(3.2, 0.3, 50),
                ]
            ),
            "petal_length": np.concatenate(
                [
                    np.random.normal(1.5, 0.3, 50),
                    np.random.normal(4.5, 0.5, 50),
                    np.random.normal(6.0, 0.4, 50),
                ]
            ),
            "petal_width": np.concatenate(
                [
                    np.random.normal(0.3, 0.1, 50),
                    np.random.normal(1.3, 0.2, 50),
                    np.random.normal(2.0, 0.2, 50),
                ]
            ),
        }
    )

    print("복잡한 데이터셋 정보:")
    print(complex_data.describe())
    print()

    # 특성 2개만 선택해서 클러스터링
    features_2d = complex_data[["petal_length", "petal_width"]]
    kmeans_complex = KMeans(k=3, random_state=42)
    kmeans_complex.fit(features_2d)

    print("복잡한 데이터 클러스터링 결과:")
    complex_cluster_info = kmeans_complex.get_cluster_info()
    print(complex_cluster_info)

    plot_clusters(
        features_2d,
        kmeans_complex.labels,
        kmeans_complex.centroids,
        "Complex Dataset Clustering (Petal Length vs Width)",
    )

    print("\n=== K-means 구현 완료 ===")
    print("✅ 주요 기능:")
    print("• 순수 pandas + numpy 구현")
    print("• 엘보우 방법으로 최적 k 찾기")
    print("• 수렴 과정 시각화")
    print("• 새로운 데이터 예측")
    print("• 다양한 k 값 비교")
    print("• 클러스터 정보 분석")


if __name__ == "__main__":
    main()
