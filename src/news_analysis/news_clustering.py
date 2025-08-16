# flake8: noqa: E501
import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import silhouette_score

warnings.filterwarnings("ignore")


class NewsClustering:
    """뉴스 기사 클러스터링"""

    def __init__(self, n_clusters=3):
        self.n_clusters = n_clusters
        self.kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        self.labels = None

    def fit_predict(self, tfidf_matrix):
        """클러스터링 수행"""
        self.labels = self.kmeans.fit_predict(tfidf_matrix)

        print(f"✅ K-Means 클러스터링 완료 ({self.n_clusters}개 클러스터)")

        return self.labels

    def analyze_clusters(self, df, feature_names, tfidf_matrix):
        """클러스터 분석"""
        print("\n🎯 클러스터 분석 결과")
        print("-" * 40)

        df_with_clusters = df.copy()
        df_with_clusters["cluster"] = self.labels

        for cluster_id in range(self.n_clusters):
            cluster_docs = df_with_clusters[df_with_clusters["cluster"] == cluster_id]
            print(f"\n📊 클러스터 {cluster_id} ({len(cluster_docs)}개 기사):")

            # 클러스터 내 기사 제목들
            for title in cluster_docs["Title"].head(3):
                print(f"   • {title[:50]}...")

            # 클러스터 중심 키워드
            cluster_center = self.kmeans.cluster_centers_[cluster_id]
            top_indices = cluster_center.argsort()[-10:][::-1]

            print("   🔑 주요 키워드:")
            for i in top_indices:
                if cluster_center[i] > 0.01:
                    print(f"      - {feature_names[i]} ({cluster_center[i]:.3f})")

        return df_with_clusters

    def visualize_clusters(self, tfidf_matrix, df_with_clusters):
        """클러스터 시각화"""
        print("\n📈 클러스터 시각화 생성 중...")

        # PCA로 차원 축소 (2D)
        pca = PCA(n_components=2, random_state=42)
        coords_2d = pca.fit_transform(tfidf_matrix.toarray())

        # 그래프 설정
        plt.figure(figsize=(15, 5))

        # 1. 클러스터 분포 시각화
        plt.subplot(1, 3, 1)
        colors = ["red", "blue", "green", "purple", "orange", "brown", "pink", "gray"]

        for i in range(self.n_clusters):
            cluster_points = coords_2d[self.labels == i]
            plt.scatter(
                cluster_points[:, 0],
                cluster_points[:, 1],
                c=colors[i % len(colors)],
                label=f"Cluster {i}",
                alpha=0.6,
            )

        plt.title("News Clusters (PCA 2D)")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend()
        plt.grid(True, alpha=0.3)

        # 2. 클러스터 크기 분포
        plt.subplot(1, 3, 2)
        cluster_counts = df_with_clusters["cluster"].value_counts().sort_index()
        plt.bar(
            range(self.n_clusters),
            cluster_counts.values,
            color=[colors[i % len(colors)] for i in range(self.n_clusters)],
        )
        plt.title("Cluster Size Distribution")
        plt.xlabel("Cluster ID")
        plt.ylabel("Number of Articles")
        plt.grid(True, alpha=0.3)

        # 각 막대 위에 숫자 표시
        for i, count in enumerate(cluster_counts.values):
            plt.text(i, count + 0.5, str(count), ha="center", va="bottom")

        # 3. 실루엣 점수 (클러스터 품질 평가)
        plt.subplot(1, 3, 3)
        silhouette_avg = silhouette_score(tfidf_matrix, self.labels)

        # 클러스터 개수별 실루엣 점수 계산
        cluster_range = range(2, min(8, len(df_with_clusters) // 2))
        silhouette_scores = []

        for n in cluster_range:
            kmeans_temp = KMeans(n_clusters=n, random_state=42)
            temp_labels = kmeans_temp.fit_predict(tfidf_matrix)
            score = silhouette_score(tfidf_matrix, temp_labels)
            silhouette_scores.append(score)

        plt.plot(cluster_range, silhouette_scores, "bo-")
        plt.axvline(
            x=self.n_clusters,
            color="red",
            linestyle="--",
            label=f"Current ({self.n_clusters} clusters)",
        )
        plt.title("Silhouette Score by Cluster Count")
        plt.xlabel("Number of Clusters")
        plt.ylabel("Silhouette Score")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig("news_clustering_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print(f"   현재 클러스터링 실루엣 점수: {silhouette_avg:.3f}")
        print("   📁 시각화 결과가 'news_clustering_analysis.png'로 저장되었습니다.")


def find_optimal_clusters(tfidf_matrix, max_clusters=8):
    """최적 클러스터 수 찾기"""
    print("\n🔍 최적 클러스터 수 탐색 중...")

    silhouette_scores = []
    inertias = []
    cluster_range = range(2, min(max_clusters + 1, len(tfidf_matrix.toarray()) // 2))

    for n_clusters in cluster_range:
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(tfidf_matrix)

        silhouette_avg = silhouette_score(tfidf_matrix, labels)
        silhouette_scores.append(silhouette_avg)
        inertias.append(kmeans.inertia_)

        print(f"   클러스터 수 {n_clusters}: 실루엣 점수 = {silhouette_avg:.3f}")

    # 최적 클러스터 수 (실루엣 점수 기준)
    optimal_idx = np.argmax(silhouette_scores)
    optimal_clusters = cluster_range[optimal_idx]

    print(
        f"\n✅ 최적 클러스터 수: {optimal_clusters} (실루엣 점수: {silhouette_scores[optimal_idx]:.3f})"
    )

    return optimal_clusters, silhouette_scores, inertias


def save_results(df_with_clusters, feature_names, clustering_model):
    """결과 저장"""
    print("\n💾 결과 저장 중...")

    # 1. 클러스터링 결과 CSV 저장
    df_with_clusters.to_csv(
        "news_clustering_results.csv", index=False, encoding="utf-8-sig"
    )
    print("   📄 클러스터링 결과: news_clustering_results.csv")

    # 2. 클러스터별 키워드 저장
    cluster_keywords = {}
    for cluster_id in range(clustering_model.n_clusters):
        cluster_center = clustering_model.kmeans.cluster_centers_[cluster_id]
        top_indices = cluster_center.argsort()[-10:][::-1]

        keywords = []
        for i in top_indices:
            if cluster_center[i] > 0.01:
                keywords.append(
                    {"keyword": feature_names[i], "score": cluster_center[i]}
                )

        cluster_keywords[f"cluster_{cluster_id}"] = keywords

    # 키워드 결과를 JSON으로 저장
    import json

    with open("cluster_keywords.json", "w", encoding="utf-8") as f:
        json.dump(cluster_keywords, f, ensure_ascii=False, indent=2)
    print("   🔑 클러스터 키워드: cluster_keywords.json")

    # 3. 클러스터별 기사 요약 저장
    cluster_summary = []
    for cluster_id in range(clustering_model.n_clusters):
        cluster_docs = df_with_clusters[df_with_clusters["cluster"] == cluster_id]

        summary = {
            "cluster_id": cluster_id,
            "article_count": len(cluster_docs),
            "sample_titles": cluster_docs["Title"].head(5).tolist(),
            "top_keywords": [
                kw["keyword"] for kw in cluster_keywords[f"cluster_{cluster_id}"][:5]
            ],
        }
        cluster_summary.append(summary)

    summary_df = pd.DataFrame(cluster_summary)
    summary_df.to_csv("cluster_summary.csv", index=False, encoding="utf-8-sig")
    print("   📊 클러스터 요약: cluster_summary.csv")

    return cluster_keywords


def main():
    """메인 함수"""
    print("📰 뉴스 클러스터링 시스템을 시작합니다!")
    print("=" * 50)

    df = pd.read_csv("it_news_articles_processed.csv")

    vectorizer = TfidfVectorizer(
        max_features=1000,  # 최대 특성 수
        stop_words="english",  # 영어 불용어 제거
        ngram_range=(1, 2),  # 1-gram과 2-gram 사용
        min_df=2,  # 최소 문서 빈도
        max_df=0.8,  # 최대 문서 빈도
    )

    tfidf_matrix = vectorizer.fit_transform(df["processed_content"])
    feature_names = vectorizer.get_feature_names_out()

    print("   ✅ TF-IDF 벡터화 완료")
    print(f"   • 특성 수: {len(feature_names)}")
    print(f"   • 행렬 크기: {tfidf_matrix.shape}")

    # 최적 클러스터 수 찾기
    if len(df) > 4:  # 클러스터 수 최적화는 데이터가 충분할 때만
        optimal_k, silhouette_scores, inertias = find_optimal_clusters(tfidf_matrix)
    else:
        optimal_k = min(3, len(df) - 1)
        print(f"\n⚠️  데이터 수가 적어 클러스터 수를 {optimal_k}개로 설정합니다.")

    # 클러스터링 수행
    print(f"\n🎯 K-Means 클러스터링 수행 중... (k={optimal_k})")

    clustering = NewsClustering(n_clusters=optimal_k)
    labels = clustering.fit_predict(tfidf_matrix)

    # 클러스터 분석
    df_with_clusters = clustering.analyze_clusters(df, feature_names, tfidf_matrix)

    # 시각화
    if len(df) > 2:  # 시각화는 최소 3개 이상의 데이터가 있을 때
        clustering.visualize_clusters(tfidf_matrix, df_with_clusters)
    else:
        print("\n⚠️  데이터 수가 적어 시각화를 생략합니다.")

    # 결과 저장
    cluster_keywords = save_results(df_with_clusters, feature_names, clustering)

    # 요약 리포트
    print("\n📊 최종 결과 요약")
    print("=" * 30)
    print(f"• 총 기사 수: {len(df)}")
    print(f"• 클러스터 수: {optimal_k}")

    if len(df) > 4:
        silhouette_avg = silhouette_score(tfidf_matrix, labels)
        print(f"• 실루엣 점수: {silhouette_avg:.3f}")

    print("• 클러스터별 기사 수:")
    for i in range(optimal_k):
        count = sum(labels == i)
        print(f"  - 클러스터 {i}: {count}개")

    print("\n📁 생성된 파일:")
    print("  • news_clustering_results.csv")
    print("  • cluster_keywords.json")
    print("  • cluster_summary.csv")
    if len(df) > 2:
        print("  • news_clustering_analysis.png")

    print("\n🎉 뉴스 클러스터링이 완료되었습니다!")

    return df_with_clusters, clustering, cluster_keywords


if __name__ == "__main__":
    # 실행
    results = main()

    # 결과 사용 예시
    df_with_clusters, clustering_model, keywords = results

    # 특정 클러스터의 기사 확인
    print("\n🔍 클러스터 0의 기사들:")
    cluster_0_articles = df_with_clusters[df_with_clusters["cluster"] == 0]
    for idx, article in cluster_0_articles.iterrows():
        print(f"  • {article['Title']}")
