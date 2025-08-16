# flake8: noqa: E501
import re

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud


class NewsTextAnalyzer:
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.tfidf_vectorizer = None
        self.tfidf_matrix = None
        self.processed_texts = []

    def preprocess_text(self, text):
        """텍스트 전처리"""
        if pd.isna(text) or text == "":
            return ""

        # 소문자 변환
        text = text.lower()
        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", "", text)
        # 특수문자 제거 (영문자, 숫자, 공백만 유지)
        text = re.sub(r"[^a-zA-Z\s]", "", text)
        # 여러 공백을 하나로
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def prepare_texts(self):
        """분석할 텍스트 준비"""
        # Parsed Content와 Description 결합하여 분석
        combined_texts = []
        for _, row in self.df.iterrows():
            content = str(row.get("Parsed Content", ""))
            description = str(row.get("Description", ""))
            title = str(row.get("Title", ""))

            combined_text = f"{title} {description} {content}"
            processed = self.preprocess_text(combined_text)
            combined_texts.append(processed)

        # 빈 텍스트 필터링
        self.processed_texts = [text for text in combined_texts if text.strip()]
        return self.processed_texts

    def perform_tfidf_analysis(self, max_features=100):
        """TF-IDF 분석 수행"""
        if not self.processed_texts:
            self.prepare_texts()

        self.tfidf_vectorizer = TfidfVectorizer(
            max_features=max_features, stop_words="english", ngram_range=(1, 2)
        )

        self.tfidf_matrix = self.tfidf_vectorizer.fit_transform(self.processed_texts)

        # 특성명과 TF-IDF 점수
        feature_names = self.tfidf_vectorizer.get_feature_names_out()
        tfidf_scores = self.tfidf_matrix.sum(axis=0).A1

        # 상위 키워드 추출
        top_keywords = sorted(
            zip(feature_names, tfidf_scores), key=lambda x: x[1], reverse=True
        )[:20]

        return top_keywords

    def perform_clustering(self, n_clusters=3):
        """K-means 클러스터링"""
        if self.tfidf_matrix is None:
            self.perform_tfidf_analysis()

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        clusters = kmeans.fit_predict(self.tfidf_matrix)

        # DataFrame에 클러스터 정보 추가
        valid_indices = [
            i for i, text in enumerate(self.prepare_texts()) if text.strip()
        ]
        cluster_df = self.df.iloc[valid_indices].copy()
        cluster_df["Cluster"] = clusters

        return cluster_df, kmeans

    def calculate_similarity_matrix(self):
        """문서간 유사도 계산"""
        if self.tfidf_matrix is None:
            self.perform_tfidf_analysis()

        similarity_matrix = cosine_similarity(self.tfidf_matrix)
        return similarity_matrix

    def visualize_top_keywords(self, top_keywords=None):
        """상위 키워드 시각화"""
        if top_keywords is None:
            top_keywords = self.perform_tfidf_analysis()

        keywords, scores = zip(*top_keywords)

        plt.figure(figsize=(12, 8))
        plt.barh(range(len(keywords)), scores)
        plt.yticks(range(len(keywords)), keywords)
        plt.xlabel("TF-IDF Score")
        plt.title("Top Keywords in IT News")
        plt.gca().invert_yaxis()
        plt.tight_layout()
        plt.show()

    def visualize_clusters(self, cluster_df=None):
        """클러스터 시각화"""
        if cluster_df is None:
            cluster_df, _ = self.perform_clustering()

        if self.tfidf_matrix is None:
            self.perform_tfidf_analysis()

        # PCA를 사용한 차원 축소
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(self.tfidf_matrix.toarray())

        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(
            reduced_data[:, 0],
            reduced_data[:, 1],
            c=cluster_df["Cluster"],
            cmap="viridis",
            alpha=0.6,
        )
        plt.colorbar(scatter)
        plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.2%} variance)")
        plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.2%} variance)")
        plt.title("News Articles Clustering (PCA)")

        # 각 점에 기사 제목 일부 표시
        for i, title in enumerate(cluster_df["Title"]):
            plt.annotate(
                title[:30] + "...",
                (reduced_data[i, 0], reduced_data[i, 1]),
                fontsize=8,
                alpha=0.7,
            )

        plt.tight_layout()
        plt.show()

    def create_wordcloud(self):
        """워드클라우드 생성"""
        if not self.processed_texts:
            self.prepare_texts()

        # 모든 텍스트 결합
        all_text = " ".join(self.processed_texts)

        wordcloud = WordCloud(
            width=800,
            height=400,
            background_color="white",
            max_words=100,
            colormap="viridis",
        ).generate(all_text)

        plt.figure(figsize=(12, 6))
        plt.imshow(wordcloud, interpolation="bilinear")
        plt.axis("off")
        plt.title("IT News Word Cloud")
        plt.tight_layout()
        plt.show()

    def visualize_similarity_heatmap(self):
        """문서 유사도 히트맵"""
        similarity_matrix = self.calculate_similarity_matrix()

        plt.figure(figsize=(10, 8))

        # 제목을 라벨로 사용 (처음 30자만)
        valid_indices = [
            i for i, text in enumerate(self.prepare_texts()) if text.strip()
        ]
        labels = [self.df.iloc[i]["Title"][:30] + "..." for i in valid_indices]

        sns.heatmap(
            similarity_matrix,
            annot=False,
            cmap="coolwarm",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.title("Document Similarity Matrix")
        plt.xticks(rotation=45, ha="right")
        plt.yticks(rotation=0)
        plt.tight_layout()
        plt.show()

    def analyze_article_stats(self):
        """기사 통계 분석"""
        stats_data = []

        for _, row in self.df.iterrows():
            word_count = row.get("Word Count", 0)
            reading_time = row.get("Reading Time (min)", 0)
            source = row.get("Source", "Unknown")

            stats_data.append(
                {
                    "Source": source,
                    "Word Count": word_count,
                    "Reading Time": reading_time,
                    "Title": row.get("Title", "")[:40] + "...",
                }
            )

        stats_df = pd.DataFrame(stats_data)

        # 통계 시각화
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))

        # 1. 소스별 기사 수
        source_counts = stats_df["Source"].value_counts()
        axes[0, 0].bar(source_counts.index, source_counts.values)
        axes[0, 0].set_title("Articles by Source")
        axes[0, 0].set_xlabel("Source")
        axes[0, 0].set_ylabel("Count")
        axes[0, 0].tick_params(axis="x", rotation=45)

        # 2. 단어 수 분포
        axes[0, 1].hist(stats_df["Word Count"], bins=10, alpha=0.7)
        axes[0, 1].set_title("Word Count Distribution")
        axes[0, 1].set_xlabel("Word Count")
        axes[0, 1].set_ylabel("Frequency")

        # 3. 읽기 시간 분포
        axes[1, 0].hist(stats_df["Reading Time"], bins=10, alpha=0.7, color="orange")
        axes[1, 0].set_title("Reading Time Distribution")
        axes[1, 0].set_xlabel("Reading Time (minutes)")
        axes[1, 0].set_ylabel("Frequency")

        # 4. 단어 수 vs 읽기 시간
        axes[1, 1].scatter(stats_df["Word Count"], stats_df["Reading Time"], alpha=0.6)
        axes[1, 1].set_title("Word Count vs Reading Time")
        axes[1, 1].set_xlabel("Word Count")
        axes[1, 1].set_ylabel("Reading Time (minutes)")

        plt.tight_layout()
        plt.show()

        return stats_df

    def generate_full_analysis_report(self):
        """전체 분석 보고서 생성"""
        print("🔍 IT 뉴스 텍스트 분석 보고서")
        print("=" * 50)

        # 기본 정보
        print(f"📊 총 기사 수: {len(self.df)}")
        print(f"📊 유효한 텍스트가 있는 기사 수: {len(self.processed_texts)}")

        # TF-IDF 분석
        print("\n🔑 상위 키워드 (TF-IDF):")
        top_keywords = self.perform_tfidf_analysis()
        for i, (keyword, score) in enumerate(top_keywords[:10], 1):
            print(f"{i:2d}. {keyword:<20} (점수: {score:.3f})")

        # 클러스터링
        print("\n🎯 클러스터 분석:")
        cluster_df, _ = self.perform_clustering()
        for cluster_id in sorted(cluster_df["Cluster"].unique()):
            cluster_articles = cluster_df[cluster_df["Cluster"] == cluster_id]
            print(f"\n클러스터 {cluster_id} ({len(cluster_articles)}개 기사):")
            for _, article in cluster_articles.head(3).iterrows():
                print(f"  - {article['Title'][:60]}...")

        # 통계 정보
        print("\n📈 기사 통계:")
        stats_df = self.analyze_article_stats()
        print(f"평균 단어 수: {stats_df['Word Count'].mean():.1f}")
        print(f"평균 읽기 시간: {stats_df['Reading Time'].mean():.1f}분")
        print(
            f"주요 뉴스 소스: {', '.join(stats_df['Source'].value_counts().head(3).index.tolist())}"
        )

        return {
            "top_keywords": top_keywords,
            "cluster_df": cluster_df,
            "stats_df": stats_df,
        }


def main():
    """메인 실행 함수"""
    print("🚀 IT 뉴스 텍스트 분석을 시작합니다...")

    df = pd.read_csv("it_news_articles.csv")  # CSV 파일에서 데이터 로드

    # 분석기 초기화
    analyzer = NewsTextAnalyzer(df)

    # 전체 분석 실행
    results = analyzer.generate_full_analysis_report()

    print("\n📊 시각화를 생성합니다...")

    # 각종 시각화 실행
    analyzer.visualize_top_keywords()
    analyzer.visualize_clusters()
    analyzer.create_wordcloud()
    analyzer.visualize_similarity_heatmap()
    analyzer.analyze_article_stats()

    print("✅ 분석 완료!")
    return analyzer, results


if __name__ == "__main__":
    analyzer, results = main()
