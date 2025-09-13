"""
간단한 문서 유사도 분석 모듈
핵심 기능만 포함
"""

from typing import Any, Dict, List

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.metrics.pairwise import cosine_similarity


class SimilarityAnalyzer:
    """문서 유사도 분석 클래스"""

    def __init__(self):
        self.tfidf_matrix = None
        self.similarity_matrix = None

    def set_tfidf_matrix(self, tfidf_matrix):
        """TF-IDF 행렬 설정"""
        self.tfidf_matrix = tfidf_matrix
        self.similarity_matrix = None

    def compute_similarity_matrix(self):
        """코사인 유사도 행렬 계산"""
        if self.tfidf_matrix is None:
            raise ValueError("TF-IDF 행렬이 설정되지 않았습니다")

        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)
        logger.success(f"유사도 행렬 계산 완료: {self.similarity_matrix.shape}")
        return self.similarity_matrix

    def get_statistics(self) -> Dict[str, float]:
        """유사도 통계 반환"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # 대각선 제외한 상삼각 행렬
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]

        return {
            "mean": float(similarities.mean()),
            "max": float(similarities.max()),
            "min": float(similarities.min()),
            "std": float(similarities.std()),
        }

    def find_similar_pairs(
        self, df: pd.DataFrame, top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """가장 유사한 문서 쌍 찾기"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        # 상삼각 행렬에서 유사도 추출
        mask = np.triu(np.ones_like(self.similarity_matrix, dtype=bool), k=1)
        similarities = self.similarity_matrix[mask]

        # 상위 k개 찾기
        top_indices = similarities.argsort()[-top_k:][::-1]
        mask_indices = np.where(mask)

        similar_pairs = []
        logger.info(f"\n🔗 가장 유사한 {top_k}개 문서 쌍:")

        for rank, idx in enumerate(top_indices, 1):
            i, j = mask_indices[0][idx], mask_indices[1][idx]
            similarity = self.similarity_matrix[i, j]

            doc1_title = df.iloc[i].get("Title", f"문서 {i}")
            doc2_title = df.iloc[j].get("Title", f"문서 {j}")

            logger.info(
                f"{rank}. {similarity:.3f}: {doc1_title[:40]}... / {doc2_title[:40]}..."
            )

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

    def find_similar_to_doc(
        self, doc_index: int, df: pd.DataFrame, n_similar: int = 3
    ) -> List[Dict[str, Any]]:
        """특정 문서와 유사한 문서들 찾기"""
        if self.similarity_matrix is None:
            self.compute_similarity_matrix()

        if doc_index >= len(df):
            raise ValueError(f"유효하지 않은 문서 인덱스: {doc_index}")

        similarities = self.similarity_matrix[doc_index].copy()
        similarities[doc_index] = -1  # 자기 자신 제외

        similar_indices = similarities.argsort()[-n_similar:][::-1]

        similar_docs = []
        for idx in similar_indices:
            if similarities[idx] > 0:
                similar_docs.append(
                    {
                        "index": int(idx),
                        "title": df.iloc[idx].get("Title", f"문서 {idx}"),
                        "similarity": float(similarities[idx]),
                    }
                )

        return similar_docs

    def save_results(self, df: pd.DataFrame, prefix: str = "similarity") -> str:
        """유사도 분석 결과 저장"""
        if self.similarity_matrix is None:
            raise ValueError("유사도 행렬이 계산되지 않았습니다")

        # 통계 정보
        stats = self.get_statistics()
        print(stats)

        # 유사한 쌍들
        similar_pairs = self.find_similar_pairs(df, 10)

        # 결과를 DataFrame으로 저장
        results_df = pd.DataFrame(similar_pairs)
        filename = f"{prefix}_results.csv"
        results_df.to_csv(filename, index=False, encoding="utf-8-sig")

        logger.success(f"유사도 분석 결과 저장: {filename}")
        return filename

    def get_info(self) -> Dict[str, Any]:
        """분석기 정보 반환"""
        if self.similarity_matrix is None:
            return {"computed": False}

        return {
            "computed": True,
            "matrix_shape": self.similarity_matrix.shape,
            "statistics": self.get_statistics(),
        }


if __name__ == "__main__":
    # 테스트 코드
    from sklearn.feature_extraction.text import TfidfVectorizer

    sample_texts = [
        "machine learning artificial intelligence",
        "deep learning neural networks",
        "cloud computing infrastructure",
        "data science analytics",
        "machine learning algorithms",
    ]

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(sample_texts)

    # 유사도 분석
    analyzer = SimilarityAnalyzer()
    analyzer.set_tfidf_matrix(tfidf_matrix)

    stats = analyzer.get_statistics()
    print(f"평균 유사도: {stats['mean']:.3f}")

    # 샘플 DataFrame
    df = pd.DataFrame(
        {
            "Title": [
                f"Article {i + 1}: {text[:20]}..."
                for i, text in enumerate(sample_texts)
            ]
        }
    )

    pairs = analyzer.find_similar_pairs(df, 3)
    for pair in pairs:
        print(
            f"유사도 {pair['similarity']:.3f}: {pair['doc1_title']} / {pair['doc2_title']}"
        )
