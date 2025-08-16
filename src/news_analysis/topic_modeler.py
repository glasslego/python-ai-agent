# flake8: noqa: E501

import warnings

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import WordCloud

warnings.filterwarnings("ignore")

# 한글 폰트 설정
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "AppleGothic",
    "Malgun Gothic",
    "gulim",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False  # 마이너스 기호 깨짐 방지


class TopicModeler:
    """LDA 토픽 모델링"""

    def __init__(self, n_topics=3, max_features=1000):
        self.n_topics = n_topics
        self.vectorizer = CountVectorizer(
            max_features=max_features, min_df=2, max_df=0.95, ngram_range=(1, 2)
        )
        self.lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=100
        )
        self.count_matrix = None
        self.doc_topic_probs = None

    def fit_transform(self, texts):
        """토픽 모델링 수행"""
        # CountVectorizer로 변환
        self.count_matrix = self.vectorizer.fit_transform(texts)

        # LDA 모델 학습
        self.lda.fit(self.count_matrix)

        # 문서-토픽 확률 분포
        doc_topic_probs = self.lda.transform(self.count_matrix)

        print(f"✅ LDA 토픽 모델링 완료 ({self.n_topics}개 토픽)")

        return doc_topic_probs

    def get_top_words_per_topic(self, n_words=10):
        """각 토픽의 상위 단어들 추출"""
        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)

        return topics

    def analyze_topics(self, df, doc_topic_probs):
        """토픽 분석 결과 출력"""
        print("\n🎭 토픽 모델링 결과")
        print("-" * 40)

        # 각 토픽의 상위 단어들
        topics = self.get_top_words_per_topic()

        for i, topic_words in enumerate(topics):
            print(f"\n📚 토픽 {i + 1}:")
            print(f"   주요 단어: {', '.join(topic_words[:8])}")

        # 각 문서의 주요 토픽 할당
        df_with_topics = df.copy()
        df_with_topics["main_topic"] = doc_topic_probs.argmax(axis=1)
        df_with_topics["topic_confidence"] = doc_topic_probs.max(axis=1)

        print("\n📰 문서별 토픽 할당:")
        for _, row in df_with_topics.head(5).iterrows():
            title = row["Title"][:40]
            topic = row["main_topic"]
            confidence = row["topic_confidence"]
            print(f"   • {title}... → 토픽 {topic + 1} ({confidence:.2f})")

        return df_with_topics

    def visualize_topics(self, df_with_topics):
        """토픽 시각화"""
        print("\n📊 토픽 시각화 생성 중...")

        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 토픽별 문서 분포
        topic_counts = df_with_topics["main_topic"].value_counts().sort_index()
        colors = ["#FF6B6B", "#4ECDC4", "#45B7D1", "#96CEB4", "#FFEAA7", "#DDA0DD"]

        axes[0, 0].pie(
            topic_counts.values,
            labels=[f"Topic {i + 1}" for i in topic_counts.index],
            autopct="%1.1f%%",
            colors=colors[: len(topic_counts)],
        )
        axes[0, 0].set_title(
            "Document Distribution by Topic", fontsize=14, fontweight="bold"
        )

        # 2. 토픽별 신뢰도 분포
        for topic_id in range(self.n_topics):
            topic_data = df_with_topics[df_with_topics["main_topic"] == topic_id]
            if len(topic_data) > 0:
                axes[0, 1].hist(
                    topic_data["topic_confidence"],
                    alpha=0.7,
                    label=f"Topic {topic_id + 1}",
                    color=colors[topic_id % len(colors)],
                    bins=10,
                )

        axes[0, 1].set_title(
            "Topic Confidence Distribution", fontsize=14, fontweight="bold"
        )
        axes[0, 1].set_xlabel("Confidence Score")
        axes[0, 1].set_ylabel("Frequency")
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)

        # 3. 토픽-단어 히트맵
        feature_names = self.vectorizer.get_feature_names_out()
        top_words_per_topic = []
        word_scores_per_topic = []

        for topic_idx in range(self.n_topics):
            topic = self.lda.components_[topic_idx]
            top_words_idx = topic.argsort()[-10:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            top_scores = [topic[i] for i in top_words_idx]

            top_words_per_topic.extend(top_words)
            word_scores_per_topic.append(top_scores)

        # 중복 제거하여 상위 단어들만 선택
        unique_words = list(set(top_words_per_topic))[:15]
        heatmap_data = []

        for topic_idx in range(self.n_topics):
            topic = self.lda.components_[topic_idx]
            topic_scores = []
            for word in unique_words:
                if word in feature_names:
                    word_idx = list(feature_names).index(word)
                    topic_scores.append(topic[word_idx])
                else:
                    topic_scores.append(0)
            heatmap_data.append(topic_scores)

        im = axes[1, 0].imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        axes[1, 0].set_xticks(range(len(unique_words)))
        axes[1, 0].set_xticklabels(unique_words, rotation=45, ha="right")
        axes[1, 0].set_yticks(range(self.n_topics))
        axes[1, 0].set_yticklabels([f"Topic {i + 1}" for i in range(self.n_topics)])
        axes[1, 0].set_title("Topic-Word Heatmap", fontsize=14, fontweight="bold")

        # 컬러바 추가
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

        # 4. 토픽별 평균 신뢰도
        avg_confidence = df_with_topics.groupby("main_topic")["topic_confidence"].mean()
        bars = axes[1, 1].bar(
            range(self.n_topics), avg_confidence.values, color=colors[: self.n_topics]
        )
        axes[1, 1].set_title("Average Topic Confidence", fontsize=14, fontweight="bold")
        axes[1, 1].set_xlabel("Topic")
        axes[1, 1].set_ylabel("Average Confidence")
        axes[1, 1].set_xticks(range(self.n_topics))
        axes[1, 1].set_xticklabels([f"Topic {i + 1}" for i in range(self.n_topics)])
        axes[1, 1].grid(True, alpha=0.3)

        # 막대 위에 값 표시
        for bar, value in zip(bars, avg_confidence.values):
            axes[1, 1].text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{value:.3f}",
                ha="center",
                va="bottom",
            )

        plt.tight_layout()
        plt.savefig("topic_modeling_analysis.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("   📁 시각화 결과가 'topic_modeling_analysis.png'로 저장되었습니다.")

    def generate_wordclouds(self):
        """토픽별 워드클라우드 생성"""
        print("\n☁️  토픽별 워드클라우드 생성 중...")

        feature_names = self.vectorizer.get_feature_names_out()

        fig, axes = plt.subplots(1, self.n_topics, figsize=(5 * self.n_topics, 5))
        if self.n_topics == 1:
            axes = [axes]

        for topic_idx in range(self.n_topics):
            topic = self.lda.components_[topic_idx]

            # 토픽의 단어-확률 딕셔너리 생성
            word_prob_dict = {}
            for word_idx, prob in enumerate(topic):
                if prob > 0.001:  # 최소 임계값
                    word_prob_dict[feature_names[word_idx]] = prob

            if word_prob_dict:  # 단어가 있는 경우에만 워드클라우드 생성
                wordcloud = WordCloud(
                    width=400,
                    height=400,
                    background_color="white",
                    max_words=50,
                    colormap="viridis",
                ).generate_from_frequencies(word_prob_dict)

                axes[topic_idx].imshow(wordcloud, interpolation="bilinear")
                axes[topic_idx].set_title(
                    f"Topic {topic_idx + 1}", fontsize=14, fontweight="bold"
                )
            else:
                axes[topic_idx].text(
                    0.5,
                    0.5,
                    "No significant words",
                    ha="center",
                    va="center",
                    transform=axes[topic_idx].transAxes,
                )
                axes[topic_idx].set_title(
                    f"Topic {topic_idx + 1}", fontsize=14, fontweight="bold"
                )

            axes[topic_idx].axis("off")

        plt.tight_layout()
        plt.savefig("topic_wordclouds.png", dpi=300, bbox_inches="tight")
        plt.show()

        print("   📁 워드클라우드가 'topic_wordclouds.png'로 저장되었습니다.")

    def get_topic_coherence(self):
        """토픽 일관성 점수 계산 (간단한 버전)"""
        topics = self.get_top_words_per_topic(10)
        feature_names = self.vectorizer.get_feature_names_out()

        coherence_scores = []
        for topic_words in topics:
            # 토픽 내 단어들의 동시 출현 빈도 계산
            word_indices = [
                list(feature_names).index(word)
                for word in topic_words
                if word in feature_names
            ]

            if len(word_indices) < 2:
                coherence_scores.append(0)
                continue

            # 간단한 일관성 점수 (실제로는 더 복잡한 계산 필요)
            cooccurrence_sum = 0
            count = 0

            for i in range(len(word_indices)):
                for j in range(i + 1, len(word_indices)):
                    word1_counts = (
                        self.count_matrix[:, word_indices[i]].toarray().flatten()
                    )
                    word2_counts = (
                        self.count_matrix[:, word_indices[j]].toarray().flatten()
                    )

                    # 두 단어가 모두 나타나는 문서 수
                    cooccurrence = np.sum((word1_counts > 0) & (word2_counts > 0))
                    cooccurrence_sum += cooccurrence
                    count += 1

            coherence_scores.append(cooccurrence_sum / count if count > 0 else 0)

        return coherence_scores


def find_optimal_topics(texts, max_topics=10):
    """최적 토픽 수 찾기"""
    print("\n🔍 최적 토픽 수 탐색 중...")

    # 데이터 수에 따라 최대 토픽 수 조정
    max_topics = min(max_topics, len(texts) // 3)

    if max_topics < 2:
        print("   ⚠️  데이터가 너무 적어 토픽 수 최적화를 건너뜁니다.")
        return 2

    perplexities = []
    topic_range = range(2, max_topics + 1)

    for n_topics in topic_range:
        vectorizer = CountVectorizer(
            max_features=1000, min_df=2, max_df=0.95, ngram_range=(1, 2)
        )
        count_matrix = vectorizer.fit_transform(texts)

        lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=42, max_iter=50
        )
        lda.fit(count_matrix)

        perplexity = lda.perplexity(count_matrix)
        perplexities.append(perplexity)

        print(f"   토픽 수 {n_topics}: Perplexity = {perplexity:.2f}")

    # 가장 낮은 perplexity를 가진 토픽 수 선택
    optimal_idx = np.argmin(perplexities)
    optimal_topics = topic_range[optimal_idx]

    print(
        f"\n✅ 최적 토픽 수: {optimal_topics} (Perplexity: {perplexities[optimal_idx]:.2f})"
    )

    return optimal_topics


def save_topic_results(df_with_topics, topic_modeler):
    """토픽 모델링 결과 저장"""
    print("\n💾 결과 저장 중...")

    # 1. 문서별 토픽 결과 저장
    df_with_topics.to_csv(
        "topic_modeling_results.csv", index=False, encoding="utf-8-sig"
    )
    print("   📄 토픽 모델링 결과: topic_modeling_results.csv")

    # 2. 토픽별 상위 단어 저장
    topics = topic_modeler.get_top_words_per_topic(15)
    topic_words_data = []

    for i, topic_words in enumerate(topics):
        topic_words_data.append(
            {
                "topic_id": i + 1,
                "top_words": ", ".join(topic_words),
                "word_list": topic_words,
            }
        )

    topic_df = pd.DataFrame(topic_words_data)
    topic_df.to_csv("topic_words.csv", index=False, encoding="utf-8-sig")
    print("   🔤 토픽별 단어: topic_words.csv")

    # 3. 토픽별 문서 요약 저장
    topic_summary = []
    for topic_id in range(topic_modeler.n_topics):
        topic_docs = df_with_topics[df_with_topics["main_topic"] == topic_id]

        summary = {
            "topic_id": topic_id + 1,
            "document_count": len(topic_docs),
            "avg_confidence": (
                topic_docs["topic_confidence"].mean() if len(topic_docs) > 0 else 0
            ),
            "sample_titles": (
                topic_docs["Title"].head(3).tolist() if len(topic_docs) > 0 else []
            ),
            "top_words": ", ".join(topics[topic_id][:5]),
        }
        topic_summary.append(summary)

    summary_df = pd.DataFrame(topic_summary)
    summary_df.to_csv("topic_summary.csv", index=False, encoding="utf-8-sig")
    print("   📊 토픽 요약: topic_summary.csv")

    # 4. 문서-토픽 확률 행렬 저장
    prob_df = pd.DataFrame(
        topic_modeler.doc_topic_probs,
        columns=[f"Topic_{i + 1}_Prob" for i in range(topic_modeler.n_topics)],
    )
    prob_df["Title"] = df_with_topics["Title"]
    prob_df.to_csv(
        "document_topic_probabilities.csv", index=False, encoding="utf-8-sig"
    )
    print("   📈 문서-토픽 확률: document_topic_probabilities.csv")


def main():
    print("🎭 LDA 토픽 모델링 시스템을 시작합니다!")
    print("=" * 50)

    # 데이터 로드 또는 생성
    print("\n📂 데이터 로드 중...")
    df = pd.read_csv("it_news_articles_processed.csv")

    # 최적 토픽 수 찾기
    # if len(df) > 6:  # 토픽 수 최적화는 데이터가 충분할 때만
    #     optimal_topics = find_optimal_topics(df["processed_content"])
    # else:
    #     optimal_topics = min(3, len(df) - 1)
    #     print(f"\n⚠️  데이터 수가 적어 토픽 수를 {optimal_topics}개로 설정합니다.")
    #
    # # 토픽 모델링 수행
    # print(f"\n🎯 LDA 토픽 모델링 수행 중... (토픽 수: {optimal_topics})")
    optimal_topics = 5

    topic_modeler = TopicModeler(n_topics=optimal_topics, max_features=1000)
    doc_topic_probs = topic_modeler.fit_transform(df["processed_content"])

    # 토픽 분석
    df_with_topics = topic_modeler.analyze_topics(df, doc_topic_probs)

    # 토픽 일관성 점수 계산
    print("\n📏 토픽 일관성 평가 중...")
    coherence_scores = topic_modeler.get_topic_coherence()

    print("   토픽별 일관성 점수:")
    for i, score in enumerate(coherence_scores):
        print(f"   • 토픽 {i + 1}: {score:.3f}")

    avg_coherence = np.mean(coherence_scores)
    print(f"   평균 일관성 점수: {avg_coherence:.3f}")

    # 시각화
    if len(df) > 2:  # 시각화는 최소 3개 이상의 데이터가 있을 때
        topic_modeler.visualize_topics(df_with_topics)

        # 워드클라우드 생성 (옵션)
        try:
            topic_modeler.generate_wordclouds()
        except ImportError:
            print(
                "   ⚠️  WordCloud 라이브러리가 설치되지 않아 워드클라우드를 생략합니다."
            )
            print("   💡 'pip install wordcloud' 명령어로 설치 후 다시 실행해보세요.")
    else:
        print("\n⚠️  데이터 수가 적어 시각화를 생략합니다.")

    # 결과 저장
    save_topic_results(df_with_topics, topic_modeler)

    # 상세 토픽 분석
    print("\n🔍 상세 토픽 분석")
    print("=" * 30)

    topics = topic_modeler.get_top_words_per_topic(10)
    for i, topic_words in enumerate(topics):
        topic_docs = df_with_topics[df_with_topics["main_topic"] == i]

        print(f"\n📚 토픽 {i + 1} 상세 분석:")
        print(f"   • 문서 수: {len(topic_docs)}개")
        print(f"   • 평균 신뢰도: {topic_docs['topic_confidence'].mean():.3f}")
        print(f"   • 주요 키워드: {', '.join(topic_words[:5])}")
        print(f"   • 일관성 점수: {coherence_scores[i]:.3f}")

        if len(topic_docs) > 0:
            print("   • 대표 기사:")
            top_confident = topic_docs.nlargest(2, "topic_confidence")
            for _, article in top_confident.iterrows():
                print(
                    f"     - {article['Title'][:50]}... ({article['topic_confidence']:.3f})"
                )

    # 토픽 트렌드 분석 (날짜가 있는 경우)
    if "Date" in df.columns:
        print("\n📈 토픽 트렌드 분석")
        try:
            df_with_topics["Date"] = pd.to_datetime(df_with_topics["Date"])
            df_with_topics["Month"] = df_with_topics["Date"].dt.to_period("M")

            trend_data = (
                df_with_topics.groupby(["Month", "main_topic"])
                .size()
                .unstack(fill_value=0)
            )
            trend_data.to_csv("topic_trends.csv", encoding="utf-8-sig")
            print("   📊 토픽 트렌드 데이터: topic_trends.csv")
        except Exception as e:
            print("error during trend analysis:", e)

    # 최종 요약 리포트
    print("\n📊 최종 결과 요약")
    print("=" * 30)
    print(f"• 총 기사 수: {len(df)}")
    print(f"• 추출된 토픽 수: {optimal_topics}")
    print(f"• 평균 토픽 일관성: {avg_coherence:.3f}")
    print("• 토픽별 문서 분포:")

    for topic_id in range(optimal_topics):
        count = len(df_with_topics[df_with_topics["main_topic"] == topic_id])
        percentage = (count / len(df)) * 100
        print(f"  - 토픽 {topic_id + 1}: {count}개 ({percentage:.1f}%)")

    print("\n📁 생성된 파일:")
    print("  • topic_modeling_results.csv - 문서별 토픽 할당 결과")
    print("  • topic_words.csv - 토픽별 주요 단어")
    print("  • topic_summary.csv - 토픽별 요약 정보")
    # TODO 수정하기
    print("  • document_topic_probabilities.csv - 문서-토픽 확률 행렬")
    if len(df) > 2:
        print("  • topic_modeling_analysis.png - 토픽 분석 시각화")
        try:
            print("  • topic_wordclouds.png - 토픽별 워드클라우드")
        except ImportError:
            pass

    # 토픽 라벨링 제안
    print("\n🏷️  토픽 라벨링 제안")
    print("-" * 20)

    topic_labels = []
    for i, topic_words in enumerate(topics):
        # 상위 3개 단어를 조합하여 라벨 제안
        label = " & ".join(topic_words[:3]).title()
        topic_labels.append(label)
        print(f'   토픽 {i + 1}: "{label}"')

    # 라벨이 적용된 결과 저장
    df_labeled = df_with_topics.copy()
    df_labeled["topic_label"] = df_labeled["main_topic"].map(
        {i: topic_labels[i] for i in range(len(topic_labels))}
    )
    df_labeled.to_csv("topic_modeling_labeled.csv", index=False, encoding="utf-8-sig")
    print("  • topic_modeling_labeled.csv - 라벨이 적용된 결과")

    print("\n🎉 토픽 모델링이 완료되었습니다!")

    return df_with_topics, topic_modeler, topic_labels


def export_to_json(
    df_with_topics, topic_modeler, filename="topic_modeling_export.json"
):
    """JSON 형식으로 결과 내보내기"""
    import json

    topics = topic_modeler.get_top_words_per_topic(15)

    export_data = {
        "model_info": {
            "n_topics": topic_modeler.n_topics,
            "n_documents": len(df_with_topics),
            "timestamp": pd.Timestamp.now().isoformat(),
        },
        "topics": [],
        "documents": [],
    }

    # 토픽 정보
    for i, topic_words in enumerate(topics):
        topic_docs = df_with_topics[df_with_topics["main_topic"] == i]

        export_data["topics"].append(
            {
                "id": i,
                "words": topic_words,
                "document_count": len(topic_docs),
                "avg_confidence": (
                    float(topic_docs["topic_confidence"].mean())
                    if len(topic_docs) > 0
                    else 0
                ),
            }
        )

    # 문서 정보
    for idx, row in df_with_topics.iterrows():
        doc_info = {
            "id": idx,
            "title": row["Title"],
            "main_topic": int(row["main_topic"]),
            "topic_confidence": float(row["topic_confidence"]),
            "topic_probabilities": (
                topic_modeler.doc_topic_probs[idx].tolist()
                if topic_modeler.doc_topic_probs is not None
                else []
            ),
        }
        export_data["documents"].append(doc_info)

    with open(filename, "w", encoding="utf-8") as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)

    print(f"   📄 JSON 내보내기 완료: {filename}")


if __name__ == "__main__":
    # 실행
    results = main()

    # 결과 사용 예시
    df_with_topics, topic_modeler, topic_labels = results

    # JSON으로 내보내기
    export_to_json(df_with_topics, topic_modeler)
