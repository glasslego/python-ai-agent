"""
LDA 토픽 모델링 모듈
"""

import warnings
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")

# 플롯 설정
plt.rcParams["font.family"] = [
    "DejaVu Sans",
    "AppleGothic",
    "Malgun Gothic",
    "gulim",
    "Arial Unicode MS",
]
plt.rcParams["axes.unicode_minus"] = False


class TopicModeler:
    """LDA(Latent Dirichlet Allocation) 토픽 모델링 클래스"""

    def __init__(
        self,
        n_topics: int = 3,
        max_features: int = 1000,
        min_df: int = 2,
        max_df: float = 0.95,
        ngram_range: Tuple[int, int] = (1, 2),
        random_state: int = 42,
    ):
        """
        Args:
            n_topics: 추출할 토픽 수
            max_features: 최대 특성 수
            min_df: 최소 문서 빈도
            max_df: 최대 문서 빈도 비율
            ngram_range: n-gram 범위
            random_state: 랜덤 시드
        """
        self.n_topics = n_topics
        self.random_state = random_state

        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=min_df,
            max_df=max_df,
            ngram_range=ngram_range,
            lowercase=False,  # 전처리에서 이미 처리
            stop_words=None,  # 전처리에서 이미 처리
        )

        self.lda = LatentDirichletAllocation(
            n_components=n_topics,
            random_state=random_state,
            max_iter=100,
            learning_method="batch",
            doc_topic_prior=None,
            topic_word_prior=None,
        )

        self.count_matrix = None
        self.doc_topic_probs = None
        self.is_fitted = False

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """
        토픽 모델링 수행

        Args:
            texts: 입력 텍스트 리스트

        Returns:
            문서-토픽 확률 분포 행렬
        """
        if not texts:
            raise ValueError("텍스트 리스트가 비어있습니다.")

        # 빈 텍스트 제거
        valid_texts = [text for text in texts if text and text.strip()]

        if not valid_texts:
            raise ValueError("유효한 텍스트가 없습니다.")

        # CountVectorizer로 변환
        self.count_matrix = self.vectorizer.fit_transform(valid_texts)

        # LDA 모델 학습
        self.lda.fit(self.count_matrix)

        # 문서-토픽 확률 분포
        self.doc_topic_probs = self.lda.transform(self.count_matrix)
        self.is_fitted = True

        logger.success(f" LDA 토픽 모델링 완료 ({self.n_topics}개 토픽)")
        logger.info(f"   문서 수: {self.count_matrix.shape[0]}")
        logger.info(f"   특성 수: {self.count_matrix.shape[1]}")
        logger.info(f"   Perplexity: {self.lda.perplexity(self.count_matrix):.2f}")

        return self.doc_topic_probs

    def get_top_words_per_topic(self, n_words: int = 10) -> List[List[str]]:
        """
        각 토픽의 상위 단어들 추출

        Args:
            n_words: 추출할 단어 수

        Returns:
            토픽별 상위 단어 리스트
        """
        if not self.is_fitted:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic_idx, topic in enumerate(self.lda.components_):
            top_words_idx = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_words_idx]
            topics.append(top_words)

        return topics

    def get_topic_word_probabilities(
        self, topic_id: int, n_words: int = 15
    ) -> List[Tuple[str, float]]:
        """
        특정 토픽의 단어별 확률 반환

        Args:
            topic_id: 토픽 ID
            n_words: 반환할 단어 수

        Returns:
            (단어, 확률) 튜플 리스트
        """
        if not self.is_fitted or topic_id >= self.n_topics:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        topic = self.lda.components_[topic_id]

        # 확률로 정규화
        topic_probs = topic / topic.sum()

        top_indices = topic_probs.argsort()[-n_words:][::-1]
        word_probs = [(feature_names[i], topic_probs[i]) for i in top_indices]

        return word_probs

    def analyze_topics(
        self, df: pd.DataFrame, show_sample_docs: int = 3, show_top_words: int = 8
    ) -> pd.DataFrame:
        """
        토픽 분석 수행

        Args:
            df: 원본 DataFrame
            show_sample_docs: 표시할 샘플 문서 수
            show_top_words: 표시할 상위 단어 수

        Returns:
            토픽 정보가 추가된 DataFrame
        """
        if not self.is_fitted:
            raise ValueError("토픽 모델링이 수행되지 않았습니다.")

        # 각 문서의 주요 토픽 할당
        df_with_topics = df.copy()
        df_with_topics["main_topic"] = self.doc_topic_probs.argmax(axis=1)
        df_with_topics["topic_confidence"] = self.doc_topic_probs.max(axis=1)

        # 모든 토픽 확률 추가
        for i in range(self.n_topics):
            df_with_topics[f"topic_{i}_prob"] = self.doc_topic_probs[:, i]

        logger.info("\n🎭 토픽 모델링 분석 결과")
        logger.info("-" * 50)

        # 각 토픽의 상위 단어들
        topics = self.get_top_words_per_topic()

        self.topic_info = {}

        for i, topic_words in enumerate(topics):
            topic_docs = df_with_topics[df_with_topics["main_topic"] == i]

            logger.info(
                f"\n📚 토픽 {i + 1} ({len(topic_docs)}개 문서, {len(topic_docs) / len(df_with_topics) * 100:.1f}%):"  # noqa: E501
            )
            logger.info(f"   주요 단어: {', '.join(topic_words[:show_top_words])}")

            # 샘플 문서 제목들
            if len(topic_docs) > 0:
                logger.info("   대표 문서:")
                sample_docs = topic_docs.nlargest(show_sample_docs, "topic_confidence")
                for _, doc in sample_docs.iterrows():
                    title = doc.get("Title", "제목 없음")[:60]
                    confidence = doc["topic_confidence"]
                    logger.info(f"     • {title}... ({confidence:.3f})")

            # 토픽 정보 저장
            self.topic_info[i] = {
                "words": topic_words,
                "document_count": len(topic_docs),
                "avg_confidence": (
                    topic_docs["topic_confidence"].mean() if len(topic_docs) > 0 else 0
                ),
            }

        return df_with_topics

    def find_optimal_topics(
        self, texts: List[str], max_topics: int = 10, min_topics: int = 2
    ) -> Tuple[int, List[float]]:
        """
        최적 토픽 수 찾기 (Perplexity 기준)

        Args:
            texts: 입력 텍스트 리스트
            max_topics: 최대 토픽 수
            min_topics: 최소 토픽 수

        Returns:
            (최적 토픽 수, perplexity 점수 리스트)
        """
        # 데이터 수에 따라 최대 토픽 수 조정
        n_docs = len([t for t in texts if t and t.strip()])
        max_topics = min(max_topics, n_docs // 3)

        if max_topics < min_topics:
            return min_topics, []

        logger.info(f"\n🔍 최적 토픽 수 탐색 중... (범위: {min_topics}-{max_topics})")

        perplexities = []
        topic_range = range(min_topics, max_topics + 1)

        for n_topics in topic_range:
            # 임시 모델 생성
            temp_vectorizer = CountVectorizer(
                max_features=self.vectorizer.max_features,
                min_df=self.vectorizer.min_df,
                max_df=self.vectorizer.max_df,
                ngram_range=self.vectorizer.ngram_range,
            )
            temp_count_matrix = temp_vectorizer.fit_transform(texts)

            temp_lda = LatentDirichletAllocation(
                n_components=n_topics, random_state=self.random_state, max_iter=50
            )
            temp_lda.fit(temp_count_matrix)

            perplexity = temp_lda.perplexity(temp_count_matrix)
            perplexities.append(perplexity)

            logger.info(f"   토픽 수 {n_topics}: Perplexity = {perplexity:.2f}")

        # 가장 낮은 perplexity를 가진 토픽 수 선택
        optimal_idx = np.argmin(perplexities)
        optimal_topics = topic_range[optimal_idx]

        logger.info(
            f"\n✅ 최적 토픽 수: {optimal_topics} (Perplexity: {perplexities[optimal_idx]:.2f})"  # noqa: E501
        )

        return optimal_topics, perplexities

    def visualize_topics(
        self,
        df_with_topics: pd.DataFrame,
        save_path: str = "topic_modeling_analysis.png",
    ) -> None:
        """
        토픽 시각화

        Args:
            df_with_topics: 토픽 정보가 포함된 DataFrame
            save_path: 저장할 파일 경로
        """
        if not self.is_fitted:
            raise ValueError("토픽 모델링이 수행되지 않았습니다.")

        logger.info("\n📊 토픽 시각화 생성 중...")

        colors = [
            "#FF6B6B",
            "#4ECDC4",
            "#45B7D1",
            "#96CEB4",
            "#FFEAA7",
            "#DDA0DD",
            "#FFB347",
            "#98FB98",
        ]

        # 그래프 설정
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # 1. 토픽별 문서 분포 (원형차트)
        topic_counts = df_with_topics["main_topic"].value_counts().sort_index()
        axes[0, 0].pie(
            topic_counts.values,
            labels=[f"Topic {i + 1}" for i in topic_counts.index],
            autopct="%1.1f%%",
            colors=colors[: len(topic_counts)],
            startangle=90,
        )
        axes[0, 0].set_title(
            "Document Distribution by Topic", fontsize=14, fontweight="bold"
        )

        # 2. 토픽별 신뢰도 분포 (히스토그램)
        for topic_id in range(self.n_topics):
            topic_data = df_with_topics[df_with_topics["main_topic"] == topic_id]
            if len(topic_data) > 0:
                axes[0, 1].hist(
                    topic_data["topic_confidence"],
                    alpha=0.7,
                    label=f"Topic {topic_id + 1}",
                    color=colors[topic_id % len(colors)],
                    bins=15,
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

        # 상위 단어들 선택
        all_top_words = set()
        for topic_idx in range(self.n_topics):
            topic_words = self.get_top_words_per_topic(8)[topic_idx]
            all_top_words.update(topic_words[:5])

        top_words = list(all_top_words)[:12]  # 최대 12개 단어

        # 히트맵 데이터 생성
        heatmap_data = []
        for topic_idx in range(self.n_topics):
            topic = self.lda.components_[topic_idx]
            topic_normalized = topic / topic.sum()  # 정규화

            topic_scores = []
            for word in top_words:
                if word in feature_names:
                    word_idx = list(feature_names).index(word)
                    topic_scores.append(topic_normalized[word_idx])
                else:
                    topic_scores.append(0)
            heatmap_data.append(topic_scores)

        im = axes[1, 0].imshow(heatmap_data, cmap="YlOrRd", aspect="auto")
        axes[1, 0].set_xticks(range(len(top_words)))
        axes[1, 0].set_xticklabels(top_words, rotation=45, ha="right")
        axes[1, 0].set_yticks(range(self.n_topics))
        axes[1, 0].set_yticklabels([f"Topic {i + 1}" for i in range(self.n_topics)])
        axes[1, 0].set_title(
            "Topic-Word Intensity Heatmap", fontsize=14, fontweight="bold"
        )

        # 컬러바 추가
        plt.colorbar(im, ax=axes[1, 0], shrink=0.8)

        # 4. 토픽별 평균 신뢰도
        avg_confidence = df_with_topics.groupby("main_topic")["topic_confidence"].mean()
        bars = axes[1, 1].bar(
            range(self.n_topics),
            avg_confidence.values,
            color=colors[: self.n_topics],
            alpha=0.8,
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
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.show()

        logger.info(f"   📁 시각화 결과가 '{save_path}'로 저장되었습니다.")

    def export_results(
        self, df_with_topics: pd.DataFrame, output_prefix: str = "topic_modeling"
    ) -> Dict[str, str]:
        """
        토픽 모델링 결과 내보내기

        Args:
            df_with_topics: 토픽 정보가 포함된 DataFrame
            output_prefix: 출력 파일 접두사

        Returns:
            저장된 파일 경로들
        """
        if not self.is_fitted:
            raise ValueError("토픽 모델링이 수행되지 않았습니다.")

        saved_files = {}

        logger.info("\n💾 토픽 모델링 결과 저장 중...")

        # 1. 문서별 토픽 결과 저장
        results_file = f"{output_prefix}_results.csv"
        df_with_topics.to_csv(results_file, index=False, encoding="utf-8-sig")
        saved_files["results"] = results_file

        # 2. 토픽별 상위 단어 저장
        topics = self.get_top_words_per_topic(15)
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
        words_file = f"{output_prefix}_words.csv"
        topic_df.to_csv(words_file, index=False, encoding="utf-8-sig")
        saved_files["words"] = words_file

        # 3. 토픽별 요약 통계 저장
        summary_data = []
        for topic_id in range(self.n_topics):
            topic_docs = df_with_topics[df_with_topics["main_topic"] == topic_id]

            summary = {
                "topic_id": topic_id + 1,
                "document_count": len(topic_docs),
                "percentage": len(topic_docs) / len(df_with_topics) * 100,
                "avg_confidence": (
                    topic_docs["topic_confidence"].mean() if len(topic_docs) > 0 else 0
                ),
                "sample_titles": (
                    topic_docs["Title"].head(3).tolist() if len(topic_docs) > 0 else []
                ),
                "top_words": ", ".join(topics[topic_id][:5]),
            }
            summary_data.append(summary)

        summary_file = f"{output_prefix}_summary.csv"
        summary_df = pd.DataFrame(summary_data)
        summary_df.to_csv(summary_file, index=False, encoding="utf-8-sig")
        saved_files["summary"] = summary_file

        # 4. 문서-토픽 확률 행렬 저장
        prob_df = pd.DataFrame(
            self.doc_topic_probs,
            columns=[f"Topic_{i + 1}_Prob" for i in range(self.n_topics)],
        )
        prob_df["Title"] = df_with_topics["Title"].reset_index(drop=True)
        prob_file = f"{output_prefix}_probabilities.csv"
        prob_df.to_csv(prob_file, index=False, encoding="utf-8-sig")
        saved_files["probabilities"] = prob_file

        logger.info(f"   📄 토픽 결과: {results_file}")
        logger.info(f"   🔤 토픽 단어: {words_file}")
        logger.info(f"   📊 토픽 요약: {summary_file}")
        logger.info(f"   📈 확률 행렬: {prob_file}")

        return saved_files

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self.is_fitted:
            return {}

        return {
            "n_topics": self.n_topics,
            "n_documents": (
                self.count_matrix.shape[0] if self.count_matrix is not None else 0
            ),
            "n_features": (
                self.count_matrix.shape[1] if self.count_matrix is not None else 0
            ),
            "perplexity": (
                self.lda.perplexity(self.count_matrix)
                if self.count_matrix is not None
                else 0
            ),
            "log_likelihood": (
                self.lda.score(self.count_matrix)
                if self.count_matrix is not None
                else 0
            ),
        }


if __name__ == "__main__":
    logger.info("🎭 토픽 모델링 모듈 테스트")

    df = pd.read_csv("processed_news.csv")
    sample_texts = df["processed_text"].dropna().tolist()

    # 토픽 모델링 수행
    modeler = TopicModeler(n_topics=3, max_features=50)
    doc_topic_probs = modeler.fit_transform(sample_texts)

    # 분석 결과
    df_with_topics = modeler.analyze_topics(df)

    # 모델 정보
    info = modeler.get_model_info()
    logger.info(f"\n모델 정보: {info}")
