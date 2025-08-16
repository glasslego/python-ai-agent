"""
문서 유사도 분석 모듈
"""

import warnings
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings(
    "ignore", category=UserWarning, module="matplotlib.font_manager"
)

# 폰트 설정
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "AppleGothic",
    "Malgun Gothic",
    "gulim",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


class SimilarityAnalyzer:
    """코사인 유사도 기반 문서 유사도 분석 클래스"""

    def __init__(self, tfidf_matrix=None):
        """
        Args:
            tfidf_matrix: TF-IDF 행렬 (선택적)
        """
        self.tfidf_matrix = tfidf_matrix
        self.similarity_matrix = None
        self.is_computed = False

    def set_tfidf_matrix(self, tfidf_matrix):
        """TF-IDF 행렬 설정"""
        self.tfidf_matrix = tfidf_matrix
        self.similarity_matrix = None
        self.is_computed = False

    def compute_similarity_matrix(self):
        """
        코사인 유사도 행렬 계산

        Returns:
            코사인 유사도 행렬
        """
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF 행렬이 설정되지 않았습니다.")

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        self.is_computed = True

        logger.success(f" 유사도 행렬 계산 완료 ({self.similarity_matrix.shape})")
        logger.info(f"   평균 유사도: {self._get_avg_similarity():.3f}")

        return self.similarity_matrix

    def _get_avg_similarity(self) -> float:
        """평균 유사도 계산 (대각선 제외)"""
        if self.similarity_matrix is None:
            return 0.0

        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]
        return similarities.mean()

    def find_similar_documents(
        self, doc_index: int, df: pd.DataFrame, n_similar: int = 5
    ) -> List[Dict[str, Any]]:
        """
        특정 문서와 유사한 문서들 찾기

        Args:
            doc_index: 기준 문서의 인덱스
            df: 원본 DataFrame
            n_similar: 찾을 유사 문서 수

        Returns:
            유사 문서 정보 리스트
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        if doc_index >= len(df) or doc_index < 0:
            raise ValueError(f"유효하지 않은 문서 인덱스: {doc_index}")

        # 자기 자신을 제외한 유사도 점수
        similarities = self.similarity_matrix[doc_index].copy()
        similarities[doc_index] = -1  # 자기 자신 제외

        # 상위 유사 문서 인덱스
        similar_indices = similarities.argsort()[-n_similar:][::-1]

        reference_title = df.iloc[doc_index].get("Title", f"문서 {doc_index}")
        logger.info(
            f"\n🔗 '{reference_title[:50]}...'와 유사한 상위 {n_similar}개 문서:"
        )

        similar_documents = []
        for i, idx in enumerate(similar_indices, 1):
            if similarities[idx] > 0:  # 유효한 유사도만
                title = df.iloc[idx].get("Title", f"문서 {idx}")
                similarity_score = similarities[idx]
                logger.info(f"   {i}. {title[:60]}...")
                logger.info(f"      유사도: {similarity_score:.3f}")

                similar_documents.append(
                    {
                        "index": int(idx),
                        "title": title,
                        "similarity": float(similarity_score),
                    }
                )

        return similar_documents

    def get_similarity_statistics(self) -> Dict[str, float]:
        """
        전체 유사도 통계 반환

        Returns:
            유사도 통계 딕셔너리
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        # 대각선 제외 (자기 자신과의 유사도 제외)
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]

        return {
            "mean_similarity": float(similarities.mean()),
            "max_similarity": float(similarities.max()),
            "min_similarity": float(similarities.min()),
            "std_similarity": float(similarities.std()),
            "median_similarity": float(np.median(similarities)),
        }

    def find_most_similar_pairs(
        self, df: pd.DataFrame, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        가장 유사한 문서 쌍들 찾기

        Args:
            df: 원본 DataFrame
            top_k: 찾을 상위 쌍 수

        Returns:
            유사한 문서 쌍 리스트
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        # 상삼각 행렬에서 유사도 추출 (중복 및 자기 자신 제거)
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]

        # 상위 k개 유사도 인덱스 찾기
        top_indices = similarities.argsort()[-top_k:][::-1]

        # 2D 인덱스로 변환
        mask_indices = np.where(mask)
        similar_pairs = []

        logger.info(f"\n🏆 가장 유사한 상위 {top_k}개 문서 쌍:")

        for rank, idx in enumerate(top_indices, 1):
            i, j = mask_indices[0][idx], mask_indices[1][idx]
            similarity = self.similarity_matrix[i, j]

            doc1_title = df.iloc[i].get("Title", f"문서 {i}")
            doc2_title = df.iloc[j].get("Title", f"문서 {j}")

            logger.info(f"   {rank}. 유사도: {similarity:.3f}")
            logger.info(f"      • {doc1_title[:50]}...")
            logger.info(f"      • {doc2_title[:50]}...")

            similar_pairs.append(
                {
                    "rank": rank,
                    "doc1_index": int(i),
                    "doc1_title": doc1_title,
                    "doc2_index": int(j),
                    "doc2_title": doc2_title,
                    "similarity": float(similarity),
                }
            )

        return similar_pairs

    def find_outliers(
        self, df: pd.DataFrame, threshold: float = 0.1
    ) -> List[Dict[str, Any]]:
        """
        평균 유사도가 낮은 이상치 문서 찾기

        Args:
            df: 원본 DataFrame
            threshold: 이상치 판단 임계값

        Returns:
            이상치 문서 리스트
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        logger.info(f"\n🎯 이상치 문서 탐지 (임계값: {threshold})")

        outliers = []

        for i in range(len(self.similarity_matrix)):
            # 자기 자신을 제외한 평균 유사도 계산
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = 0  # 자기 자신 제외
            avg_similarity = similarities.mean()

            if avg_similarity < threshold:
                title = df.iloc[i].get("Title", f"문서 {i}")
                outliers.append(
                    {
                        "index": int(i),
                        "title": title,
                        "avg_similarity": float(avg_similarity),
                    }
                )

        if outliers:
            logger.info(f"   📋 발견된 이상치 문서: {len(outliers)}개")
            for outlier in outliers:
                logger.info(f"   • {outlier['title'][:60]}...")
                logger.info(f"     평균 유사도: {outlier['avg_similarity']:.3f}")
        else:
            logger.info("   ✅ 임계값 이하의 이상치 문서가 없습니다.")

        return outliers

    def visualize_similarity_matrix(
        self,
        df: pd.DataFrame,
        top_n: Optional[int] = None,
        save_path: str = "similarity_matrix_heatmap.png",
    ) -> None:
        """
        유사도 행렬 히트맵 시각화

        Args:
            df: 원본 DataFrame
            top_n: 시각화할 상위 문서 수 (None이면 모든 문서)
            save_path: 저장할 파일 경로
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        logger.info("\n📊 유사도 행렬 히트맵 생성 중...")

        # 상위 N개 문서만 시각화 (가독성을 위해)
        if top_n is None:
            top_n = min(20, len(df))

        matrix_subset = self.similarity_matrix[:top_n, :top_n]
        titles_subset = [
            title[:25] + "..." if len(title) > 25 else title
            for title in df["Title"].head(top_n)
        ]

        plt.figure(figsize=(12, 10))

        # 히트맵 생성 (하삼각 마스크 적용)
        mask = np.triu(np.ones_like(matrix_subset, dtype=bool), k=1)

        sns.heatmap(
            matrix_subset,
            mask=mask,
            xticklabels=titles_subset,
            yticklabels=titles_subset,
            annot=True if top_n <= 10 else False,
            fmt=".2f",
            cmap="RdYlBu_r",
            center=0.5,
            square=True,
            cbar_kws={"label": "Cosine Similarity"},
            linewidths=0.5,
        )

        plt.title(
            f"Document Similarity Matrix (Top {top_n})", fontsize=16, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"   📁 히트맵이 '{save_path}'로 저장되었습니다.")

    def visualize_similarity_distribution(
        self, save_path: str = "similarity_distribution.png"
    ) -> None:
        """
        유사도 분포 시각화

        Args:
            save_path: 저장할 파일 경로
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        logger.info("\n📈 유사도 분포 분석 중...")

        # 대각선 제외한 유사도 값들
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        # 1. 히스토그램
        axes[0].hist(
            similarities,
            bins=30,
            alpha=0.7,
            color="skyblue",
            edgecolor="black",
            density=True,
        )
        axes[0].axvline(
            similarities.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {similarities.mean():.3f}",
        )
        axes[0].axvline(
            np.median(similarities),
            color="green",
            linestyle="--",
            label=f"Median: {np.median(similarities):.3f}",
        )
        axes[0].set_title("Similarity Distribution", fontweight="bold")
        axes[0].set_xlabel("Cosine Similarity")
        axes[0].set_ylabel("Density")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # 2. 박스플롯
        box_plot = axes[1].boxplot(similarities, patch_artist=True)
        box_plot["boxes"][0].set_facecolor("lightblue")
        box_plot["boxes"][0].set_alpha(0.7)
        axes[1].set_title("Similarity Box Plot", fontweight="bold")
        axes[1].set_ylabel("Cosine Similarity")
        axes[1].grid(True, alpha=0.3)

        # 3. 누적 분포 함수
        sorted_sim = np.sort(similarities)
        cumulative = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
        axes[2].plot(sorted_sim, cumulative, linewidth=2, color="darkblue")
        axes[2].set_title("Cumulative Distribution Function", fontweight="bold")
        axes[2].set_xlabel("Cosine Similarity")
        axes[2].set_ylabel("Cumulative Probability")
        axes[2].grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"   📁 분포 분석이 '{save_path}'로 저장되었습니다.")

        # 통계 정보 출력
        stats = self.get_similarity_statistics()
        logger.info("   📊 유사도 통계:")
        logger.info(f"      • 평균: {stats['mean_similarity']:.3f}")
        logger.info(f"      • 중앙값: {stats['median_similarity']:.3f}")
        logger.info(f"      • 표준편차: {stats['std_similarity']:.3f}")
        logger.info(f"      • 최대값: {stats['max_similarity']:.3f}")
        logger.info(f"      • 최소값: {stats['min_similarity']:.3f}")

    def create_similarity_network(
        self,
        df: pd.DataFrame,
        threshold: float = 0.3,
        top_n: int = 15,
        save_path: str = "similarity_network.png",
    ) -> None:
        """
        유사도 기반 네트워크 그래프 생성

        Args:
            df: 원본 DataFrame
            threshold: 연결 임계값
            top_n: 표시할 상위 문서 수
            save_path: 저장할 파일 경로
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        logger.info(f"\n🕸️  유사도 네트워크 생성 중... (임계값: {threshold})")

        # 상위 N개 문서만 사용
        matrix_subset = self.similarity_matrix[:top_n, :top_n]
        titles_subset = df["Title"].head(top_n).tolist()

        # 네트워크 그래프 생성
        G = nx.Graph()

        # 노드 추가
        for i, title in enumerate(titles_subset):
            short_title = title[:30] + "..." if len(title) > 30 else title
            G.add_node(i, title=short_title, full_title=title)

        # 엣지 추가 (임계값 이상의 유사도)
        edges_added = 0
        for i in range(len(matrix_subset)):
            for j in range(i + 1, len(matrix_subset)):
                similarity = matrix_subset[i, j]
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)
                    edges_added += 1

        logger.info(f"   🔗 추가된 엣지 수: {edges_added}")

        # 그래프 시각화
        plt.figure(figsize=(14, 10))

        if len(G.edges()) > 0:
            # 레이아웃 설정
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            # 엣지가 없으면 원형 배치
            pos = nx.circular_layout(G)

        # 노드 크기 (연결 수에 비례)
        node_sizes = [300 + G.degree(node) * 100 for node in G.nodes()]

        # 노드 색상 (연결 수에 따라)
        node_colors = [G.degree(node) for node in G.nodes()]

        # 엣지 두께 (유사도에 비례)
        if len(G.edges()) > 0:
            edge_weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]
        else:
            edge_weights = []

        # 노드 그리기
        nodes = nx.draw_networkx_nodes(
            G,
            pos,
            node_size=node_sizes,
            node_color=node_colors,
            cmap=plt.cm.viridis,
            alpha=0.8,
        )

        # 엣지 그리기
        if len(G.edges()) > 0:
            nx.draw_networkx_edges(
                G, pos, width=edge_weights, alpha=0.6, edge_color="gray"
            )

        # 노드 라벨
        labels = {node: data["title"] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

        # 컬러바 추가
        if nodes is not None:
            plt.colorbar(nodes, label="Number of Connections")

        plt.title(
            f"Document Similarity Network (Threshold: {threshold})",
            fontsize=16,
            fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"   📁 네트워크 그래프가 '{save_path}'로 저장되었습니다.")

        # 네트워크 통계
        if len(G.nodes()) > 0:
            logger.info("   📊 네트워크 통계:")
            logger.info(f"      • 노드 수: {len(G.nodes())}")
            logger.info(f"      • 엣지 수: {len(G.edges())}")
            if len(G.edges()) > 0:
                logger.info(
                    f"      • 평균 연결도: {sum(dict(G.degree()).values()) / len(G.nodes()):.2f}"  # noqa: E501
                )
                logger.info(f"      • 밀도: {nx.density(G):.3f}")

    def export_results(
        self, df: pd.DataFrame, output_prefix: str = "similarity_analysis"
    ) -> Dict[str, str]:
        """
        유사도 분석 결과 내보내기

        Args:
            df: 원본 DataFrame
            output_prefix: 출력 파일 접두사

        Returns:
            저장된 파일 경로들
        """
        if not self.is_computed:
            self.compute_similarity_matrix()

        saved_files = {}

        logger.info("\n💾 유사도 분석 결과 저장 중...")

        # 1. 유사도 행렬 저장
        similarity_df = pd.DataFrame(
            self.similarity_matrix,
            columns=[f"Doc_{i}" for i in range(len(df))],
            index=[f"Doc_{i}" for i in range(len(df))],
        )
        matrix_file = f"{output_prefix}_matrix.csv"
        similarity_df.to_csv(matrix_file, encoding="utf-8-sig")
        saved_files["matrix"] = matrix_file

        # 2. 유사한 문서 쌍 저장
        similar_pairs = self.find_most_similar_pairs(df, top_k=10)
        if similar_pairs:
            pairs_df = pd.DataFrame(similar_pairs)
            pairs_file = f"{output_prefix}_pairs.csv"
            pairs_df.to_csv(pairs_file, index=False, encoding="utf-8-sig")
            saved_files["pairs"] = pairs_file

        # 3. 문서별 평균 유사도 저장
        doc_similarities = []
        for i in range(len(self.similarity_matrix)):
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = 0  # 자기 자신 제외
            avg_similarity = similarities.mean()
            max_similarity = similarities.max()

            doc_similarities.append(
                {
                    "doc_index": i,
                    "title": df.iloc[i].get("Title", f"문서 {i}"),
                    "avg_similarity": avg_similarity,
                    "max_similarity": max_similarity,
                }
            )

        doc_sim_df = pd.DataFrame(doc_similarities)
        doc_sim_df = doc_sim_df.sort_values("avg_similarity", ascending=False)
        doc_file = f"{output_prefix}_document_stats.csv"
        doc_sim_df.to_csv(doc_file, index=False, encoding="utf-8-sig")
        saved_files["document_stats"] = doc_file

        # 4. 이상치 문서 저장
        outliers = self.find_outliers(df, threshold=0.1)
        if outliers:
            outliers_df = pd.DataFrame(outliers)
            outliers_file = f"{output_prefix}_outliers.csv"
            outliers_df.to_csv(outliers_file, index=False, encoding="utf-8-sig")
            saved_files["outliers"] = outliers_file

        # 5. 통계 요약 저장
        stats = self.get_similarity_statistics()
        stats_df = pd.DataFrame([stats])
        stats_file = f"{output_prefix}_statistics.csv"
        stats_df.to_csv(stats_file, index=False, encoding="utf-8-sig")
        saved_files["statistics"] = stats_file

        logger.info(f"   📊 유사도 행렬: {matrix_file}")
        logger.info(f"   🔗 유사 문서 쌍: {pairs_file}")
        logger.info(f"   📄 문서별 통계: {doc_file}")
        logger.info(f"   📈 전체 통계: {stats_file}")
        if outliers:
            logger.info(f"   🎯 이상치 문서: {outliers_file}")

        return saved_files


if __name__ == "__main__":
    logger.info("🔍 문서 유사도 분석 모듈 테스트")

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

    # 유사도 분석 수행
    analyzer = SimilarityAnalyzer(tfidf_matrix)
    analyzer.compute_similarity_matrix()

    # 통계 출력
    stats = analyzer.get_similarity_statistics()
    logger.info(f"\n유사도 통계: {stats}")

    # 유사한 문서 찾기
    similar_docs = analyzer.find_similar_documents(0, df, 3)
    logger.info(f"\n유사한 문서: {similar_docs}")
