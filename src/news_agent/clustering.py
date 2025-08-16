"""
K-Means 클러스터링 기반 뉴스 분류 모듈 (auto_optimize + 내부 tfidf_matrix 보관)
"""

import json
import warnings
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


class NewsClustering:
    """뉴스 기사 K-Means 클러스터링 클래스"""

    def __init__(
        self,
        n_clusters: int = 3,
        random_state: int = 42,
        auto_optimize: bool = False,
        min_clusters: int = 2,
        max_clusters: int = 8,
        tfidf_matrix: Optional[Any] = None,
        feature_names: Optional[np.ndarray] = None,
    ):
        """
        Args:
            n_clusters: 기본 클러스터 개수
            random_state: 랜덤 시드
            auto_optimize: fit 시 find_optimal_clusters 실행 여부
            min_clusters: 자동 최적화 시 최소 클러스터 수
            max_clusters: 자동 최적화 시 최대 클러스터 수
            tfidf_matrix: 사전 계산된 TF-IDF 행렬 (CSR 등 희소행렬)
            feature_names: TF-IDF 특성 이름 배열 (vectorizer.get_feature_names_out())
        """
        self.tfidf_matrix = tfidf_matrix
        self.feature_names = feature_names

        self.n_clusters = max(2, n_clusters)  # 실루엣 계산을 위해 최소 2
        self.random_state = random_state

        # 최적화 설정
        self.auto_optimize = auto_optimize
        self.min_clusters = max(2, min_clusters)
        self.max_clusters = max(self.min_clusters, max_clusters)

        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state)
        self.labels: Optional[np.ndarray] = None
        self.is_fitted = False
        self.cluster_info: Dict[int, Any] = {}

    # ---------- 유틸 ----------
    def set_features(
        self,
        tfidf_matrix: Any,
        feature_names: Optional[np.ndarray] = None,
    ) -> None:
        """외부에서 TF-IDF 행렬(및 특성명)을 교체/설정"""
        self.tfidf_matrix = tfidf_matrix
        if feature_names is not None:
            self.feature_names = feature_names
        # 행렬이 바뀌면 기존 학습 상태는 무효화
        self.is_fitted = False
        self.labels = None

    def _require_matrix(self) -> None:
        if self.tfidf_matrix is None:
            raise ValueError(
                "tfidf_matrix가 설정되지 않았습니다. 생성자에 전달하거나 set_features()를 호출하세요."
            )

    # ---------- 모델 ----------
    def fit_predict(self) -> np.ndarray:
        """
        클러스터링 수행 (auto_optimize=True면 최적 k 탐색 후 적용)
        Returns:
            클러스터 라벨 배열
        """
        self._require_matrix()

        # ✅ 자동 최적화: 현재 데이터 기준 최적 k 산출 후 적용
        if self.auto_optimize:
            optimal_k, _ = self.find_optimal_clusters(
                max_clusters=self.max_clusters,
                min_clusters=self.min_clusters,
            )
            if optimal_k != self.n_clusters:
                logger.info(
                    f"최적 k={optimal_k} 적용 (기존 {self.n_clusters} → {optimal_k})"
                )
                self.n_clusters = int(optimal_k)
                self.kmeans = KMeans(
                    n_clusters=self.n_clusters, random_state=self.random_state
                )

        self.labels = self.kmeans.fit_predict(self.tfidf_matrix)
        self.is_fitted = True

        logger.success(f"K-Means 클러스터링 완료 ({self.n_clusters}개 클러스터)")
        try:
            logger.info(f"실루엣 점수: {self.get_silhouette_score():.3f}")
        except Exception as e:
            logger.warning(f"실루엣 점수 계산 실패: {e}")

        return self.labels

    def predict(self) -> np.ndarray:
        """새로운 데이터에 대해 클러스터 예측 (사전 학습 필요)"""
        if not self.is_fitted:
            raise ValueError(
                "모델이 학습되지 않았습니다. fit_predict()를 먼저 호출하세요."
            )
        self._require_matrix()
        return self.kmeans.predict(self.tfidf_matrix)

    def get_silhouette_score(self) -> float:
        """실루엣 점수 계산"""
        self._require_matrix()
        if self.labels is None or len(set(self.labels)) < 2:
            return 0.0
        return silhouette_score(self.tfidf_matrix, self.labels)

    # ---------- 분석/시각화 ----------
    def analyze_clusters(
        self,
        df: pd.DataFrame,
        top_keywords: int = 10,
    ) -> pd.DataFrame:
        """
        클러스터 분석: df에 'cluster' 컬럼 추가 및 키워드/샘플 로깅
        """
        if not self.is_fitted:
            raise ValueError("클러스터링이 수행되지 않았습니다.")
        self._require_matrix()

        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = self.labels

        logger.info("클러스터 분석 결과")
        cluster_info: Dict[int, Any] = {}

        for cluster_id in range(self.n_clusters):
            cluster_docs = df_with_clusters[df_with_clusters["cluster"] == cluster_id]
            cluster_size = len(cluster_docs)
            logger.info(f"\n📊 클러스터 {cluster_id} ({cluster_size}개 기사):")

            # 샘플 제목
            sample_titles = (
                cluster_docs["Title"].head(3).tolist()
                if "Title" in cluster_docs.columns
                else []
            )
            for title in sample_titles:
                logger.info(f"   • {title[:60]}...")

            # 클러스터 중심 키워드
            keywords = []
            if (
                hasattr(self.kmeans, "cluster_centers_")
                and self.feature_names is not None
            ):
                cluster_center = self.kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-top_keywords:][::-1]

                logger.info("   🔑 주요 키워드:")
                for i in top_indices:
                    if cluster_center[i] > 0.01:
                        keyword = self.feature_names[i]
                        score = cluster_center[i]
                        keywords.append({"keyword": keyword, "score": score})
                        logger.info(f"      - {keyword} ({score:.3f})")
            elif self.feature_names is None:
                logger.warning(
                    "feature_names가 없어 키워드 로깅을 건너뜁니다. (vectorizer.get_feature_names_out() 전달 필요)"  # noqa: E501
                )

            cluster_info[cluster_id] = {
                "size": cluster_size,
                "sample_titles": sample_titles,
                "keywords": keywords,
            }

        self.cluster_info = cluster_info
        return df_with_clusters

    def find_optimal_clusters(
        self, max_clusters: int = 8, min_clusters: int = 2
    ) -> Tuple[int, List[float]]:
        """최적 클러스터 수 찾기 (실루엣 기준)"""
        self._require_matrix()

        n_samples = self.tfidf_matrix.shape[0]
        # 표본 수에 따른 상한 보정
        max_clusters = min(max_clusters, max(2, n_samples // 2))
        min_clusters = max(2, min_clusters)

        if max_clusters < min_clusters:
            logger.warning("표본 수가 적어 최적화 범위를 축소했습니다.")
            return min_clusters, []

        logger.info(
            f"\n🔍 최적 클러스터 수 탐색 중... (범위: {min_clusters}-{max_clusters})"
        )

        silhouette_scores: List[float] = []
        cluster_range = list(range(min_clusters, max_clusters + 1))

        for n_clusters in cluster_range:
            try:
                kmeans_temp = KMeans(
                    n_clusters=n_clusters, random_state=self.random_state
                )
                temp_labels = kmeans_temp.fit_predict(self.tfidf_matrix)
                # 실루엣은 최소 2개 클러스터 필요
                if len(set(temp_labels)) < 2:
                    silhouette_avg = -1.0
                else:
                    silhouette_avg = silhouette_score(self.tfidf_matrix, temp_labels)
                silhouette_scores.append(silhouette_avg)
                logger.info(
                    f"   클러스터 수 {n_clusters}: 실루엣 점수 = {silhouette_avg:.3f}"
                )
            except Exception as e:
                logger.warning(f"   클러스터 수 {n_clusters}: 계산 실패({e})")
                silhouette_scores.append(-1.0)

        optimal_idx = int(np.argmax(silhouette_scores))
        optimal_clusters = cluster_range[optimal_idx]
        logger.info(
            f"\n✅ 최적 클러스터 수: {optimal_clusters} (실루엣 점수: {silhouette_scores[optimal_idx]:.3f})"  # noqa: E501
        )

        return optimal_clusters, silhouette_scores

    def visualize_clusters(
        self,
        df_with_clusters: pd.DataFrame,
        save_path: str = "news_clustering_analysis.png",
    ) -> None:
        """클러스터 시각화"""
        if not self.is_fitted:
            raise ValueError("클러스터링이 수행되지 않았습니다.")
        self._require_matrix()

        logger.info("\n📈 클러스터 시각화 생성 중...")

        # PCA 2D
        pca = PCA(n_components=2, random_state=self.random_state)
        coords_2d = pca.fit_transform(self.tfidf_matrix.toarray())

        colors = [
            "red",
            "blue",
            "green",
            "purple",
            "orange",
            "brown",
            "pink",
            "gray",
            "cyan",
            "magenta",
        ]
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))

        # 1) 산점도
        ax1 = axes[0]
        for i in range(self.n_clusters):
            cluster_points = coords_2d[self.labels == i]
            ax1.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=colors[i % len(colors)],
                label=f"Cluster {i}",
                alpha=0.7,
                s=50,
            )
        ax1.set_title("News Clusters (PCA 2D)", fontsize=14)
        ax1.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)")
        ax1.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)")
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # 2) 크기 분포
        ax2 = axes[1]
        cluster_counts = df_with_clusters["cluster"].value_counts().sort_index()
        bars = ax2.bar(
            range(self.n_clusters),
            cluster_counts.values,
            color=[colors[i % len(colors)] for i in range(self.n_clusters)],
            alpha=0.7,
        )
        ax2.set_title("Cluster Size Distribution", fontsize=14)
        ax2.set_xlabel("Cluster ID")
        ax2.set_ylabel("Number of Articles")
        ax2.grid(True, alpha=0.3)
        for bar, count in zip(bars, cluster_counts.values):
            ax2.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.5,
                str(count),
                ha="center",
                va="bottom",
            )

        # 3) 실루엣 비교
        ax3 = axes[2]
        silhouette_current = self.get_silhouette_score()
        cluster_range = range(2, min(9, len(df_with_clusters) // 2 + 1))
        silhouette_scores = []
        for n in cluster_range:
            try:
                kmeans_temp = KMeans(n_clusters=n, random_state=self.random_state)
                temp_labels = kmeans_temp.fit_predict(self.tfidf_matrix)
                score = (
                    silhouette_score(self.tfidf_matrix, temp_labels)
                    if len(set(temp_labels)) > 1
                    else -1.0
                )
                silhouette_scores.append(score)
            except Exception:
                silhouette_scores.append(-1.0)

        if silhouette_scores:
            ax3.plot(
                list(cluster_range), silhouette_scores, "bo-", linewidth=2, markersize=6
            )
            ax3.axvline(
                x=self.n_clusters,
                color="red",
                linestyle="--",
                linewidth=2,
                label=f"Current ({self.n_clusters} clusters)",
            )
            ax3.axhline(
                y=silhouette_current,
                color="red",
                linestyle=":",
                alpha=0.7,
                label=f"Score: {silhouette_current:.3f}",
            )

        ax3.set_title("Silhouette Score by Cluster Count", fontsize=14)
        ax3.set_xlabel("Number of Clusters")
        ax3.set_ylabel("Silhouette Score")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"   현재 클러스터링 실루엣 점수: {silhouette_current:.3f}")
        logger.info(f"   📁 시각화 결과가 '{save_path}'로 저장되었습니다.")

    # ---------- 결과 저장/요약 ----------
    def export_results(
        self,
        df_with_clusters: pd.DataFrame,
        output_prefix: str = "clustering",
    ) -> Dict[str, str]:
        """클러스터링 결과 내보내기"""
        if not self.is_fitted:
            raise ValueError("클러스터링이 수행되지 않았습니다.")

        saved_files: Dict[str, str] = {}
        logger.info("\n💾 클러스터링 결과 저장 중...")

        # 1) 결과 CSV
        results_file = f"{output_prefix}_results.csv"
        df_with_clusters.to_csv(results_file, index=False, encoding="utf-8-sig")
        saved_files["results"] = results_file

        # 2) 클러스터별 키워드 JSON
        cluster_keywords: Dict[str, Any] = {}
        if hasattr(self.kmeans, "cluster_centers_") and self.feature_names is not None:
            for cluster_id in range(self.n_clusters):
                cluster_center = self.kmeans.cluster_centers_[cluster_id]
                top_indices = cluster_center.argsort()[-10:][::-1]
                keywords = []
                for i in top_indices:
                    if cluster_center[i] > 0.01:
                        keywords.append(
                            {
                                "keyword": self.feature_names[i],
                                "score": float(cluster_center[i]),
                            }
                        )
                cluster_keywords[f"cluster_{cluster_id}"] = keywords
        else:
            logger.warning("feature_names가 없어 키워드 JSON이 비어 있을 수 있습니다.")

        keywords_file = f"{output_prefix}_keywords.json"
        with open(keywords_file, "w", encoding="utf-8") as f:
            json.dump(cluster_keywords, f, ensure_ascii=False, indent=2)
        saved_files["keywords"] = keywords_file

        # 3) 요약 CSV
        summary_data = []
        for cluster_id in range(self.n_clusters):
            cluster_docs = df_with_clusters[df_with_clusters["cluster"] == cluster_id]
            summary = {
                "cluster_id": cluster_id,
                "article_count": len(cluster_docs),
                "percentage": len(cluster_docs) / len(df_with_clusters) * 100,
                "sample_titles": (
                    cluster_docs["Title"].head(3).tolist()
                    if "Title" in cluster_docs.columns
                    else []
                ),
                "top_keywords": (
                    [
                        kw["keyword"]
                        for kw in cluster_keywords.get(f"cluster_{cluster_id}", [])[:5]
                    ]
                    if cluster_keywords
                    else []
                ),
            }
            summary_data.append(summary)

        summary_file = f"{output_prefix}_summary.csv"
        pd.DataFrame(summary_data).to_csv(
            summary_file, index=False, encoding="utf-8-sig"
        )
        saved_files["summary"] = summary_file

        logger.info(f"   📄 클러스터링 결과: {results_file}")
        logger.info(f"   🔑 클러스터 키워드: {keywords_file}")
        logger.info(f"   📊 클러스터 요약: {summary_file}")

        return saved_files

    def get_cluster_summary(self) -> Dict[str, Any]:
        """클러스터링 요약 정보 반환"""
        if not self.is_fitted:
            return {}
        cluster_counts = np.bincount(self.labels)
        return {
            "n_clusters": self.n_clusters,
            "total_documents": len(self.labels),
            "cluster_sizes": cluster_counts.tolist(),
            "largest_cluster": int(np.argmax(cluster_counts)),
            "smallest_cluster": int(np.argmin(cluster_counts)),
            "size_std": float(np.std(cluster_counts)),
        }


if __name__ == "__main__":
    logger.info("📰 뉴스 클러스터링 모듈 테스트")

    df = pd.read_csv("processed_news.csv")

    vectorizer = TfidfVectorizer(
        max_features=1000,  # 최대 특성 수
        stop_words="english",  # 영어 불용어 제거
        ngram_range=(1, 2),  # 1-gram과 2-gram 사용
        min_df=2,  # 최소 문서 빈도
        max_df=0.8,  # 최대 문서 빈도
    )

    tfidf_matrix = vectorizer.fit_transform(df["processed_text"])
    feature_names = vectorizer.get_feature_names_out()

    # ✅ 내부에 tfidf_matrix/feature_names를 보관하여 사용
    clustering = NewsClustering(
        n_clusters=3,
        auto_optimize=True,
        min_clusters=2,
        max_clusters=6,
        tfidf_matrix=tfidf_matrix,
        feature_names=feature_names,
    )

    labels = clustering.fit_predict()

    # 분석 결과
    df_with_clusters = clustering.analyze_clusters(df, top_keywords=10)

    # 시각화 (옵션)
    clustering.visualize_clusters(df_with_clusters)

    # 요약 정보
    summary = clustering.get_cluster_summary()
    logger.info(f"\n클러스터링 요약: {summary}")
