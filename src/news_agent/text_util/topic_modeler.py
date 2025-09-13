"""
간단한 LDA 토픽 모델링 모듈
핵심 기능만 포함
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer


class TopicModeler:
    """LDA 토픽 모델링 클래스"""

    def __init__(
        self, n_topics: int = 3, max_features: int = 500, random_state: int = 42
    ):
        """
        Args:
            n_topics: 추출할 토픽 수
            max_features: 최대 특성 수
            random_state: 랜덤 시드
        """
        self.n_topics = n_topics
        self.random_state = random_state

        self.vectorizer = CountVectorizer(
            max_features=max_features,
            min_df=1,
            max_df=0.95,
            lowercase=False,
            stop_words=None,
        )

        self.lda = LatentDirichletAllocation(
            n_components=n_topics, random_state=random_state, max_iter=50
        )

        self.is_fitted = False
        self.doc_topic_probs = None

    def fit_transform(self, texts: List[str]) -> np.ndarray:
        """토픽 모델링 수행"""
        if not texts:
            raise ValueError("텍스트 리스트가 비어있습니다")

        valid_texts = [text for text in texts if text and text.strip()]
        if not valid_texts:
            raise ValueError("유효한 텍스트가 없습니다")

        # 벡터화 및 LDA 학습
        count_matrix = self.vectorizer.fit_transform(valid_texts)
        self.doc_topic_probs = self.lda.fit_transform(count_matrix)
        self.is_fitted = True

        logger.success(
            f"토픽 모델링 완료: {self.n_topics}개 토픽, {len(valid_texts)}개 문서"
        )
        return self.doc_topic_probs

    def get_top_words(self, n_words: int = 5) -> List[List[str]]:
        """각 토픽의 상위 단어 추출"""
        if not self.is_fitted:
            return []

        feature_names = self.vectorizer.get_feature_names_out()
        topics = []

        for topic in self.lda.components_:
            top_indices = topic.argsort()[-n_words:][::-1]
            top_words = [feature_names[i] for i in top_indices]
            topics.append(top_words)

        return topics

    def analyze(self, df: pd.DataFrame) -> pd.DataFrame:
        """토픽 분석 및 결과 반환"""
        if not self.is_fitted:
            raise ValueError("모델이 학습되지 않았습니다")

        df_result = df.copy()
        df_result["main_topic"] = self.doc_topic_probs.argmax(axis=1)
        df_result["topic_confidence"] = self.doc_topic_probs.max(axis=1)

        # 결과 출력
        logger.info("\n🎭 토픽 분석 결과:")
        topics = self.get_top_words(5)

        for i, topic_words in enumerate(topics):
            topic_docs = df_result[df_result["main_topic"] == i]
            logger.info(
                f"토픽 {i + 1} ({len(topic_docs)}개 문서): {', '.join(topic_words)}"
            )

        return df_result

    def save_results(self, df_result: pd.DataFrame, prefix: str = "topics") -> str:
        """결과 저장"""
        filename = f"{prefix}_results.csv"
        df_result.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.success(f"토픽 분석 결과 저장: {filename}")
        return filename

    def get_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        if not self.is_fitted:
            return {}

        return {
            "n_topics": self.n_topics,
            "n_documents": self.doc_topic_probs.shape[0],
            "is_fitted": self.is_fitted,
        }


if __name__ == "__main__":
    # 테스트 코드
    sample_texts = [
        "machine learning artificial intelligence deep neural networks",
        "cloud computing aws azure google servers infrastructure",
        "cybersecurity privacy data protection encryption firewall",
        "mobile apps ios android development programming",
        "blockchain cryptocurrency bitcoin ethereum smart contracts",
    ]

    modeler = TopicModeler(n_topics=3)
    doc_topics = modeler.fit_transform(sample_texts)

    topics = modeler.get_top_words(3)
    for i, topic in enumerate(topics):
        print(f"Topic {i + 1}: {', '.join(topic)}")
