# flake8: noqa: E501

import warnings

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 한글 폰트 설정
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "AppleGothic",
    "Malgun Gothic",
    "gulim",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지

# 폰트 경고 무시
warnings.filterwarnings(
    "ignore", category=UserWarning, module="matplotlib.font_manager"
)

# 한글 폰트 설정
plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False


class SimilarityAnalyzer:
    """문서 유사도 분석"""

    def __init__(self, tfidf_matrix):
        self.tfidf_matrix = tfidf_matrix
        self.similarity_matrix = None

    def compute_similarity_matrix(self):
        """코사인 유사도 행렬 계산"""
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

        print(f"✅ 유사도 행렬 계산 완료 ({self.similarity_matrix.shape})")

        return self.similarity_matrix

    def find_similar_articles(self, article_index, df, n_similar=3):
        """특정 기사와 유사한 기사들 찾기"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # 자기 자신을 제외한 유사도 점수
        similarities = self.similarity_matrix[article_index].copy()
        similarities[article_index] = -1  # 자기 자신 제외

        # 상위 유사 기사 인덱스
        similar_indices = similarities.argsort()[-n_similar:][::-1]

        print(f"\n🔗 '{df.iloc[article_index]['Title'][:40]}...'와 유사한 기사들:")

        similar_articles = []
        for i, idx in enumerate(similar_indices, 1):
            title = df.iloc[idx]["Title"]
            similarity_score = similarities[idx]
            print(f"   {i}. {title[:50]}... (유사도: {similarity_score:.3f})")

            similar_articles.append(
                {"index": idx, "title": title, "similarity": similarity_score}
            )

        return similar_articles

    def analyze_overall_similarity(self, df):
        """전체 유사도 분석"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # 대각선 제외 (자기 자신과의 유사도 제외)
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]

        print("\n📊 전체 유사도 통계:")
        print(f"   평균 유사도: {similarities.mean():.3f}")
        print(f"   최대 유사도: {similarities.max():.3f}")
        print(f"   최소 유사도: {similarities.min():.3f}")

        # 가장 유사한 기사 쌍 찾기
        max_sim_value = similarities.max()
        max_indices = np.where(self.similarity_matrix == max_sim_value)
        i, j = max_indices[0][0], max_indices[1][0]

        print("\n🎯 가장 유사한 기사 쌍:")
        print(f"   1. {df.iloc[i]['Title'][:40]}...")
        print(f"   2. {df.iloc[j]['Title'][:40]}...")
        print(f"   유사도: {max_sim_value:.3f}")

        return {
            "mean_similarity": similarities.mean(),
            "max_similarity": similarities.max(),
            "min_similarity": similarities.min(),
            "most_similar_pair": (i, j, max_sim_value),
        }

    def visualize_similarity_matrix(self, df, top_n=None):
        """유사도 행렬 시각화"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        print("\n📊 유사도 행렬 시각화 생성 중...")

        # 상위 N개 문서만 시각화 (가독성을 위해)
        if top_n is None:
            top_n = min(20, len(df))

        matrix_subset = self.similarity_matrix[:top_n, :top_n]
        titles_subset = [
            title[:20] + "..." if len(title) > 20 else title
            for title in df["Title"].head(top_n)
        ]

        plt.figure(figsize=(12, 10))

        # 히트맵 생성
        mask = np.triu(np.ones_like(matrix_subset, dtype=bool))  # 상삼각 마스크

        sns.heatmap(
            matrix_subset,
            mask=mask,
            xticklabels=titles_subset,
            yticklabels=titles_subset,
            annot=True if top_n <= 10 else False,
            fmt=".2f",
            cmap="coolwarm",
            center=0.5,
            square=True,
            cbar_kws={"label": "Cosine Similarity"},
        )

        plt.title(
            f"Document Similarity Matrix (Top {top_n})", fontsize=16, fontweight="bold"
        )
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.savefig("similarity_matrix_heatmap.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("   📁 히트맵이 'similarity_matrix_heatmap.png'로 저장되었습니다.")

    def create_similarity_distribution(self):
        """유사도 분포 히스토그램"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        print("\n📈 유사도 분포 분석 중...")

        # 대각선 제외한 유사도 값들
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]

        plt.figure(figsize=(15, 5))

        # 1. 히스토그램
        plt.subplot(1, 3, 1)
        plt.hist(similarities, bins=30, alpha=0.7, color="skyblue", edgecolor="black")
        plt.axvline(
            similarities.mean(),
            color="red",
            linestyle="--",
            label=f"Mean: {similarities.mean():.3f}",
        )
        plt.title("Similarity Distribution", fontweight="bold")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Frequency")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 박스플롯
        plt.subplot(1, 3, 2)
        plt.boxplot(similarities)
        plt.title("Similarity Box Plot", fontweight="bold")
        plt.ylabel("Cosine Similarity")
        plt.grid(True, alpha=0.3)

        # 3. 누적 분포
        plt.subplot(1, 3, 3)
        sorted_sim = np.sort(similarities)
        cumulative = np.arange(1, len(sorted_sim) + 1) / len(sorted_sim)
        plt.plot(sorted_sim, cumulative, linewidth=2)
        plt.title("Cumulative Distribution", fontweight="bold")
        plt.xlabel("Cosine Similarity")
        plt.ylabel("Cumulative Probability")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            "similarity_distribution_analysis.png", dpi=300, bbox_inches="tight"
        )
        plt.show()

        print(
            "   📁 분포 분석이 'similarity_distribution_analysis.png'로 저장되었습니다."
        )

    def create_similarity_network(self, df, threshold=0.3, top_n=15):
        """유사도 기반 네트워크 그래프"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        print(f"\n🕸️  유사도 네트워크 생성 중... (임계값: {threshold})")

        # 상위 N개 문서만 사용
        matrix_subset = self.similarity_matrix[:top_n, :top_n]
        titles_subset = df["Title"].head(top_n).tolist()

        # 네트워크 그래프 생성
        G = nx.Graph()

        # 노드 추가
        for i, title in enumerate(titles_subset):
            G.add_node(i, title=title[:30] + "..." if len(title) > 30 else title)

        # 엣지 추가 (임계값 이상의 유사도)
        for i in range(len(matrix_subset)):
            for j in range(i + 1, len(matrix_subset)):
                similarity = matrix_subset[i, j]
                if similarity > threshold:
                    G.add_edge(i, j, weight=similarity)

        # 그래프 시각화
        plt.figure(figsize=(14, 10))

        # 레이아웃 설정
        if len(G.edges()) > 0:
            pos = nx.spring_layout(G, k=2, iterations=50)
        else:
            pos = nx.circular_layout(G)

        # 노드 크기 (연결 수에 비례)
        node_sizes = [300 + G.degree(node) * 100 for node in G.nodes()]

        # 엣지 두께 (유사도에 비례)
        edge_weights = [G[u][v]["weight"] * 5 for u, v in G.edges()]

        # 그래프 그리기
        nx.draw_networkx_nodes(
            G, pos, node_size=node_sizes, node_color="lightblue", alpha=0.7
        )

        if len(G.edges()) > 0:
            nx.draw_networkx_edges(
                G, pos, width=edge_weights, alpha=0.6, edge_color="gray"
            )

        # 노드 라벨
        labels = {node: data["title"] for node, data in G.nodes(data=True)}
        nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight="bold")

        plt.title(
            f"Document Similarity Network (Threshold: {threshold})",
            fontsize=16,
            fontweight="bold",
        )
        plt.axis("off")
        plt.tight_layout()
        plt.savefig("similarity_network.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("   📁 네트워크 그래프가 'similarity_network.png'로 저장되었습니다.")
        print(f"   🔗 연결된 엣지 수: {len(G.edges())}")

    def create_dendrogram(self, df, top_n=20):
        """계층적 클러스터링 덴드로그램"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        print("\n🌳 계층적 클러스터링 덴드로그램 생성 중...")

        # 상위 N개 문서만 사용
        matrix_subset = self.similarity_matrix[:top_n, :top_n]
        titles_subset = [
            title[:25] + "..." if len(title) > 25 else title
            for title in df["Title"].head(top_n)
        ]

        # 거리 행렬로 변환 (1 - 유사도)
        distance_matrix = 1 - matrix_subset

        # 대각선을 0으로 설정 (자기 자신과의 거리)
        np.fill_diagonal(distance_matrix, 0)

        # 계층적 클러스터링
        condensed_distances = squareform(distance_matrix)
        linkage_matrix = linkage(condensed_distances, method="ward")

        # 덴드로그램 생성
        plt.figure(figsize=(15, 8))

        dendrogram(
            linkage_matrix,
            labels=titles_subset,
            orientation="top",
            distance_sort="descending",
            show_leaf_counts=True,
        )

        plt.title("Hierarchical Clustering Dendrogram", fontsize=16, fontweight="bold")
        plt.xlabel("Documents")
        plt.ylabel("Distance")
        plt.xticks(rotation=45, ha="right")
        plt.tight_layout()
        plt.savefig("similarity_dendrogram.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("   📁 덴드로그램이 'similarity_dendrogram.png'로 저장되었습니다.")

    def find_outliers(self, df, threshold=0.1):
        """유사도가 낮은 이상치 문서 찾기"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        print(f"\n🎯 이상치 문서 탐지 중... (임계값: {threshold})")

        # 각 문서의 평균 유사도 계산 (자기 자신 제외)
        avg_similarities = []
        for i in range(len(self.similarity_matrix)):
            similarities = self.similarity_matrix[i].copy()
            similarities[i] = 0  # 자기 자신 제외
            avg_sim = similarities.mean()
            avg_similarities.append(avg_sim)

        # 임계값 이하인 문서들 찾기
        outliers = []
        for i, avg_sim in enumerate(avg_similarities):
            if avg_sim < threshold:
                outliers.append(
                    {
                        "index": i,
                        "title": df.iloc[i]["Title"],
                        "avg_similarity": avg_sim,
                    }
                )

        if outliers:
            print(f"   📋 발견된 이상치 문서 ({len(outliers)}개):")
            for outlier in outliers:
                print(
                    f"   • {outlier['title'][:50]}... (평균 유사도: {outlier['avg_similarity']:.3f})"
                )
        else:
            print("   ✅ 임계값 이하의 이상치 문서가 없습니다.")

        return outliers


def save_similarity_results(analyzer, df, similar_pairs, outliers):
    """유사도 분석 결과 저장"""
    print("\n💾 유사도 분석 결과 저장 중...")

    # 1. 유사도 행렬 저장
    similarity_df = pd.DataFrame(
        analyzer.similarity_matrix,
        columns=[f"Doc_{i}" for i in range(len(df))],
        index=[f"Doc_{i}" for i in range(len(df))],
    )
    similarity_df.to_csv("similarity_matrix.csv", encoding="utf-8-sig")
    print("   📊 유사도 행렬: similarity_matrix.csv")

    # 2. 유사한 문서 쌍 저장
    if similar_pairs:
        pairs_data = []
        for pair in similar_pairs:
            pairs_data.append(
                {
                    "doc1_index": pair["doc1_idx"],
                    "doc1_title": pair["doc1_title"],
                    "doc2_index": pair["doc2_idx"],
                    "doc2_title": pair["doc2_title"],
                    "similarity": pair["similarity"],
                }
            )

        pairs_df = pd.DataFrame(pairs_data)
        pairs_df.to_csv("similar_document_pairs.csv", index=False, encoding="utf-8-sig")
        print("   🔗 유사 문서 쌍: similar_document_pairs.csv")

    # 3. 이상치 문서 저장
    if outliers:
        outliers_df = pd.DataFrame(outliers)
        outliers_df.to_csv("outlier_documents.csv", index=False, encoding="utf-8-sig")
        print("   🎯 이상치 문서: outlier_documents.csv")

    # 4. 문서별 평균 유사도 저장
    avg_similarities = []
    for i in range(len(analyzer.similarity_matrix)):
        similarities = analyzer.similarity_matrix[i].copy()
        similarities[i] = 0  # 자기 자신 제외
        avg_sim = similarities.mean()
        avg_similarities.append(
            {"doc_index": i, "title": df.iloc[i]["Title"], "avg_similarity": avg_sim}
        )

    avg_sim_df = pd.DataFrame(avg_similarities)
    avg_sim_df = avg_sim_df.sort_values("avg_similarity", ascending=False)
    avg_sim_df.to_csv(
        "document_avg_similarities.csv", index=False, encoding="utf-8-sig"
    )
    print("   📈 문서별 평균 유사도: document_avg_similarities.csv")


def find_top_similar_pairs(analyzer, df, top_k=5):
    """상위 유사 문서 쌍 찾기"""
    if analyzer.similarity_matrix is None:
        analyzer.compute_similarity_matrix()

    # 상삼각 행렬에서 유사도 추출 (중복 제거)
    mask = np.triu(np.ones_like(analyzer.similarity_matrix, dtype=bool), k=1)
    similarities = analyzer.similarity_matrix[mask]

    # 상위 k개 유사도 인덱스 찾기
    top_indices = similarities.argsort()[-top_k:][::-1]

    # 2D 인덱스로 변환
    mask_indices = np.where(mask)
    top_pairs = []

    for idx in top_indices:
        i, j = mask_indices[0][idx], mask_indices[1][idx]
        similarity = analyzer.similarity_matrix[i, j]

        top_pairs.append(
            {
                "doc1_idx": i,
                "doc1_title": df.iloc[i]["Title"],
                "doc2_idx": j,
                "doc2_title": df.iloc[j]["Title"],
                "similarity": similarity,
            }
        )

    print(f"\n🏆 상위 {top_k}개 유사 문서 쌍:")
    for idx, pair in enumerate(top_pairs, 1):
        print(f"   {idx}. 유사도 {pair['similarity']:.3f}")
        print(f"      • {pair['doc1_title'][:40]}...")
        print(f"      • {pair['doc2_title'][:40]}...")

    return top_pairs


def main():
    print("🔍 문서 유사도 분석 시스템을 시작합니다!")
    print("=" * 50)

    # 1. 데이터 로드 또는 생성
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv("it_news_articles_processed.csv")
    print(f"   ✅ news_data.csv 파일에서 {len(df)}개 기사 로드")

    # 데이터 기본 정보 출력
    print("\n📋 데이터 정보:")
    print(f"   • 총 기사 수: {len(df)}")

    # 2. TF-IDF 벡터화
    vectorizer = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
    )

    tfidf_matrix = vectorizer.fit_transform(df["processed_content"])

    print("   ✅ TF-IDF 벡터화 완료")
    print(f"   • 특성 수: {tfidf_matrix.shape[1]}")
    print(f"   • 문서 수: {tfidf_matrix.shape[0]}")

    # 3. 유사도 분석기 초기화
    print("\n🎯 유사도 분석 시작...")
    analyzer = SimilarityAnalyzer(tfidf_matrix)

    # 4. 전체 유사도 분석
    overall_stats = analyzer.analyze_overall_similarity(df)

    # 5. 특정 기사와 유사한 기사 찾기 (예시)
    if len(df) > 1:
        print("\n🔍 특정 기사 유사도 분석 예시:")
        sample_article_idx = 0
        similar_articles = analyzer.find_similar_articles(
            sample_article_idx, df, n_similar=3
        )
        print(similar_articles)

    # 6. 상위 유사 문서 쌍 찾기
    top_pairs = find_top_similar_pairs(analyzer, df, top_k=5)

    # 7. 이상치 문서 탐지
    outliers = analyzer.find_outliers(df, threshold=0.1)

    # 8. 시각화 생성
    if len(df) > 1:
        print("\n📊 시각화 생성 중...")

        # 유사도 행렬 히트맵
        analyzer.visualize_similarity_matrix(df, top_n=min(15, len(df)))

        # 유사도 분포 분석
        analyzer.create_similarity_distribution()

        # 네트워크 그래프 (문서가 충분한 경우)
        if len(df) >= 5:
            analyzer.create_similarity_network(
                df, threshold=0.2, top_n=min(12, len(df))
            )

        # 덴드로그램 (문서가 충분한 경우)
        if len(df) >= 3:
            analyzer.create_dendrogram(df, top_n=min(15, len(df)))
    else:
        print("\n⚠️  문서 수가 부족하여 시각화를 생략합니다.")

    # 9. 결과 저장
    save_similarity_results(analyzer, df, top_pairs, outliers)

    # 10. 개별 문서 분석 도구 (사용자 입력)
    print("\n🔧 개별 문서 분석 도구")
    print("-" * 25)
    print("사용 가능한 명령어:")
    print("1. 'similar <숫자>' - 특정 인덱스 문서의 유사 문서 찾기")
    print("2. 'compare <숫자1> <숫자2>' - 두 문서 직접 비교")
    print("3. 'list' - 모든 문서 목록 보기")
    print("4. 'stats' - 전체 통계 다시 보기")
    print("5. 'quit' - 종료")

    while True:
        try:
            command = input("\n명령어 입력 (또는 'quit'): ").strip().lower()

            if command == "quit":
                break
            elif command == "list":
                print("\n📋 문서 목록:")
                for i, title in enumerate(df["Title"]):
                    print(f"   {i}: {title[:60]}...")
            elif command == "stats":
                print("\n📊 전체 유사도 통계:")
                print(f"   평균 유사도: {overall_stats['mean_similarity']:.3f}")
                print(f"   최대 유사도: {overall_stats['max_similarity']:.3f}")
                print(f"   최소 유사도: {overall_stats['min_similarity']:.3f}")
            elif command.startswith("similar "):
                try:
                    idx = int(command.split()[1])
                    if 0 <= idx < len(df):
                        analyzer.find_similar_articles(idx, df, n_similar=5)
                    else:
                        print(
                            f"   ❌ 유효하지 않은 인덱스입니다. (0-{len(df) - 1} 범위)"
                        )
                except (ValueError, IndexError):
                    print("   ❌ 올바른 형식: 'similar <숫자>'")
            elif command.startswith("compare "):
                try:
                    parts = command.split()
                    idx1, idx2 = int(parts[1]), int(parts[2])
                    if 0 <= idx1 < len(df) and 0 <= idx2 < len(df):
                        similarity = analyzer.similarity_matrix[idx1, idx2]
                        print("\n🔍 문서 비교:")
                        print(f"   문서 {idx1}: {df.iloc[idx1]['Title'][:50]}...")
                        print(f"   문서 {idx2}: {df.iloc[idx2]['Title'][:50]}...")
                        print(f"   유사도: {similarity:.3f}")
                    else:
                        print(
                            f"   ❌ 유효하지 않은 인덱스입니다. (0-{len(df) - 1} 범위)"
                        )
                except (ValueError, IndexError):
                    print("   ❌ 올바른 형식: 'compare <숫자1> <숫자2>'")
            else:
                print(
                    "   ❌ 알 수 없는 명령어입니다. 'list', 'similar <숫자>', 'compare <숫자1> <숫자2>', 'stats', 'quit' 중 하나를 입력하세요."
                )

        except KeyboardInterrupt:
            print("\n\n👋 분석을 종료합니다.")
            break
        except Exception as e:
            print(f"   ❌ 오류 발생: {e}")

    # 11. 최종 요약 리포트
    print("\n📊 최종 분석 요약")
    print("=" * 30)
    print(f"• 총 문서 수: {len(df)}")
    print(f"• 평균 유사도: {overall_stats['mean_similarity']:.3f}")
    print(f"• 최대 유사도: {overall_stats['max_similarity']:.3f}")
    print(f"• 상위 유사 쌍 수: {len(top_pairs)}")
    print(f"• 이상치 문서 수: {len(outliers)}")

    # 카테고리별 유사도 분석
    high_similarity_pairs = len([p for p in top_pairs if p["similarity"] > 0.5])
    medium_similarity_pairs = len(
        [p for p in top_pairs if 0.2 <= p["similarity"] <= 0.5]
    )
    low_similarity_pairs = len([p for p in top_pairs if p["similarity"] < 0.2])

    print("\n📈 유사도 분포:")
    print(f"  • 높은 유사도 (>0.5): {high_similarity_pairs}쌍")
    print(f"  • 중간 유사도 (0.2-0.5): {medium_similarity_pairs}쌍")
    print(f"  • 낮은 유사도 (<0.2): {low_similarity_pairs}쌍")

    print("\n📁 생성된 파일:")
    print("  • similarity_matrix.csv - 전체 유사도 행렬")
    print("  • similar_document_pairs.csv - 상위 유사 문서 쌍")
    print("  • document_avg_similarities.csv - 문서별 평균 유사도")
    if outliers:
        print("  • outlier_documents.csv - 이상치 문서 목록")
    if len(df) > 1:
        print("  • similarity_matrix_heatmap.png - 유사도 히트맵")
        print("  • similarity_distribution_analysis.png - 유사도 분포 분석")
        if len(df) >= 5:
            print("  • similarity_network.png - 유사도 네트워크")
        if len(df) >= 3:
            print("  • similarity_dendrogram.png - 계층적 클러스터링")

    print("\n🎉 문서 유사도 분석이 완료되었습니다!")

    return analyzer, overall_stats, top_pairs, outliers


if __name__ == "__main__":
    results = main()
    analyzer, overall_stats, top_pairs, outliers = results
