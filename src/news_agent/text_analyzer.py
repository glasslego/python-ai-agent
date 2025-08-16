"""
TF-IDF 기반 텍스트 분석 모듈
"""

from typing import List, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.feature_extraction.text import TfidfVectorizer


class TextAnalyzer:
    """TF-IDF 기반 텍스트 분석 클래스"""

    def __init__(
        self,
        max_features: int = 1000,
        ngram_range: Tuple[int, int] = (1, 2),
        min_df: int = 2,
        max_df: float = 0.95,
    ):
        """
        Args:
            max_features: 최대 특성 수
            ngram_range: n-gram 범위 (1-gram, 2-gram 등)
            min_df: 최소 문서 빈도
            max_df: 최대 문서 빈도 비율
        """
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            min_df=min_df,
            max_df=max_df,
            lowercase=False,  # 전처리에서 이미 소문자 변환
            stop_words=None,  # 전처리에서 이미 불용어 제거
        )
        self.tfidf_matrix = None
        self.feature_names = None
        self.is_fitted = False

    def fit_transform(self, texts: List[str]):
        """
        TF-IDF 벡터화 수행

        Args:
            texts: 텍스트 리스트

        Returns:
            TF-IDF 행렬
        """
        if not texts:
            raise ValueError("텍스트 리스트가 비어있습니다.")

        # 빈 텍스트 제거
        valid_texts = [text for text in texts if text and text.strip()]

        if not valid_texts:
            raise ValueError("유효한 텍스트가 없습니다.")

        self.tfidf_matrix = self.vectorizer.fit_transform(valid_texts)
        self.feature_names = self.vectorizer.get_feature_names_out()
        self.is_fitted = True

        logger.success(" TF-IDF 벡터화 완료")
        logger.info(f"   문서 수: {self.tfidf_matrix.shape[0]}")
        logger.info(f"   특성 수: {self.tfidf_matrix.shape[1]}")

        return self.tfidf_matrix

    def transform(self, texts: List[str]):
        """
        새로운 텍스트에 대해 TF-IDF 변환 수행

        Args:
            texts: 변환할 텍스트 리스트

        Returns:
            TF-IDF 행렬
        """
        if not self.is_fitted:
            raise ValueError("먼저 fit_transform을 호출해야 합니다.")

        return self.vectorizer.transform(texts)

    def get_matrix_info(self) -> dict:
        """TF-IDF 행렬 정보 반환"""
        if not self.is_fitted:
            return {}

        tfidf_matrix = self.tfidf_matrix
        total_elements = tfidf_matrix.shape[0] * tfidf_matrix.shape[1]
        non_zero_elements = tfidf_matrix.nnz
        sparsity = (1 - non_zero_elements / total_elements) * 100

        return {
            "shape": tfidf_matrix.shape,
            "total_elements": total_elements,
            "non_zero_elements": non_zero_elements,
            "sparsity_percentage": sparsity,
            "dtype": str(tfidf_matrix.dtype),
        }

    def get_top_keywords_global(self, n_keywords: int = 20) -> List[Tuple[str, float]]:
        """
        전체 코퍼스에서 상위 키워드 추출

        Args:
            n_keywords: 추출할 키워드 수

        Returns:
            (키워드, 점수) 튜플 리스트
        """
        if not self.is_fitted:
            return []

        # 각 특성의 평균 TF-IDF 점수 계산
        mean_scores = np.array(self.tfidf_matrix.mean(axis=0)).flatten()

        # 상위 키워드 추출
        top_indices = mean_scores.argsort()[-n_keywords:][::-1]
        top_keywords = [(self.feature_names[i], mean_scores[i]) for i in top_indices]

        return top_keywords

    def get_document_keywords(
        self, doc_index: int, n_keywords: int = 10
    ) -> List[Tuple[str, float]]:
        """
        특정 문서의 상위 키워드 추출

        Args:
            doc_index: 문서 인덱스
            n_keywords: 추출할 키워드 수

        Returns:
            (키워드, 점수) 튜플 리스트
        """
        if not self.is_fitted:
            return []

        if doc_index >= self.tfidf_matrix.shape[0]:
            return []

        doc_scores = self.tfidf_matrix[doc_index].toarray().flatten()
        top_indices = doc_scores.argsort()[-n_keywords:][::-1]

        doc_keywords = [
            (self.feature_names[i], doc_scores[i])
            for i in top_indices
            if doc_scores[i] > 0
        ]

        return doc_keywords

    def analyze_keywords_comprehensive(
        self,
        df: pd.DataFrame,
        show_top_global: int = 20,
        show_top_per_doc: int = 5,
        show_sample_docs: int = 3,
    ) -> dict:
        """
        종합적인 키워드 분석 수행

        Args:
            df: 원본 DataFrame
            show_top_global: 전역 상위 키워드 수
            show_top_per_doc: 문서별 상위 키워드 수
            show_sample_docs: 샘플로 보여줄 문서 수

        Returns:
            분석 결과 딕셔너리
        """
        if not self.is_fitted:
            return {}

        results = {}

        # 전체 상위 키워드
        top_keywords = self.get_top_keywords_global(show_top_global)
        results["global_keywords"] = top_keywords

        logger.info("\n🔍 TF-IDF 키워드 분석")
        logger.info("-" * 50)
        logger.info(f" 전체 상위 {show_top_global}개 키워드:")
        for i, (keyword, score) in enumerate(top_keywords, 1):
            logger.info(f"   {i:2d}. {keyword:<20} ({score:.4f})")

        # 문서별 키워드 분석
        document_keywords = {}
        logger.info(
            f"\n📰 문서별 상위 {show_top_per_doc}개 키워드 (샘플 {show_sample_docs}개):"
        )

        for i in range(min(show_sample_docs, len(df), self.tfidf_matrix.shape[0])):
            title = df.iloc[i].get("Title", f"문서 {i}")[:50]
            doc_keywords = self.get_document_keywords(i, show_top_per_doc)
            document_keywords[i] = doc_keywords

            logger.info(f"\n   📄 {title}...")
            for keyword, score in doc_keywords:
                logger.info(f"      • {keyword} ({score:.3f})")

        results["document_keywords"] = document_keywords
        return results

    def export_analysis_results(
        self, df: pd.DataFrame, output_prefix: str = "tfidf_analysis"
    ) -> dict:
        """
        분석 결과를 CSV 파일로 저장

        Args:
            df: 원본 DataFrame
            output_prefix: 출력 파일 접두사

        Returns:
            저장된 파일 정보
        """
        if not self.is_fitted:
            return {}

        saved_files = {}

        # 1. 전체 상위 키워드 저장
        top_keywords = self.get_top_keywords_global(50)
        keywords_df = pd.DataFrame(top_keywords, columns=["Keyword", "TF-IDF_Score"])
        keywords_file = f"{output_prefix}_top_keywords.csv"
        keywords_df.to_csv(keywords_file, index=False)
        saved_files["top_keywords"] = keywords_file

        # 2. 문서별 키워드 분석 결과 저장
        document_results = []

        for i in range(min(len(df), self.tfidf_matrix.shape[0])):
            doc_keywords = self.get_document_keywords(i, 10)
            keywords = [kw[0] for kw in doc_keywords]
            scores = [kw[1] for kw in doc_keywords]

            result = {
                "Document_Index": i,
                "Title": df.iloc[i].get("Title", f"문서 {i}"),
                "Top_Keywords": ", ".join(keywords[:5]),
                "Top_Keyword_Scores": ", ".join(
                    [f"{score:.3f}" for score in scores[:5]]
                ),
                "All_Keywords": ", ".join(keywords),
                "All_Scores": ", ".join([f"{score:.3f}" for score in scores]),
            }
            document_results.append(result)

        doc_results_df = pd.DataFrame(document_results)
        doc_file = f"{output_prefix}_document_keywords.csv"
        doc_results_df.to_csv(doc_file, index=False)
        saved_files["document_keywords"] = doc_file

        # 3. 행렬 정보 저장
        matrix_info = self.get_matrix_info()
        info_df = pd.DataFrame([matrix_info])
        info_file = f"{output_prefix}_matrix_info.csv"
        info_df.to_csv(info_file, index=False)
        saved_files["matrix_info"] = info_file

        logger.info("\n💾 분석 결과 저장 완료:")
        for key, filename in saved_files.items():
            logger.info(f"   • {key}: {filename}")

        return saved_files

    def get_keyword_statistics(self) -> dict:
        """키워드 통계 정보 반환"""
        if not self.is_fitted:
            return {}

        # 문서별 통계
        doc_scores = np.array(self.tfidf_matrix.sum(axis=1)).flatten()

        # 단어별 통계
        word_scores = np.array(self.tfidf_matrix.sum(axis=0)).flatten()

        return {
            "document_stats": {
                "mean_tfidf_sum": float(doc_scores.mean()),
                "max_tfidf_sum": float(doc_scores.max()),
                "min_tfidf_sum": float(doc_scores.min()),
                "std_tfidf_sum": float(doc_scores.std()),
            },
            "word_stats": {
                "mean_tfidf_sum": float(word_scores.mean()),
                "max_tfidf_sum": float(word_scores.max()),
                "min_tfidf_sum": float(word_scores.min()),
                "std_tfidf_sum": float(word_scores.std()),
                "top_word": self.feature_names[word_scores.argmax()],
                "bottom_word": self.feature_names[word_scores.argmin()],
            },
        }

    def print_statistics(self):
        """통계 정보 출력"""
        if not self.is_fitted:
            logger.error(" TF-IDF 모델이 학습되지 않았습니다.")
            return

        matrix_info = self.get_matrix_info()
        stats = self.get_keyword_statistics()

        logger.info("\n📊 TF-IDF 분석 통계")
        logger.info("=" * 40)

        logger.info("📄 매트릭스 정보:")
        logger.info(f"   • 크기: {matrix_info['shape']}")
        logger.info(f"   • 희소성: {matrix_info['sparsity_percentage']:.2f}%")
        logger.info(f"   • 0이 아닌 원소: {matrix_info['non_zero_elements']:,}")

        logger.info("\n📊 문서별 통계:")
        doc_stats = stats["document_stats"]
        logger.info(f"   • 평균 TF-IDF 합: {doc_stats['mean_tfidf_sum']:.3f}")
        logger.info(f"   • 최대값: {doc_stats['max_tfidf_sum']:.3f}")
        logger.info(f"   • 최소값: {doc_stats['min_tfidf_sum']:.3f}")

        logger.info("\n🔤 단어별 통계:")
        word_stats = stats["word_stats"]
        logger.info(f"   • 평균 TF-IDF 합: {word_stats['mean_tfidf_sum']:.3f}")
        logger.info(f"   • 가장 중요한 단어: {word_stats['top_word']}")


if __name__ == "__main__":
    # 사용 예제
    analyzer = TextAnalyzer(max_features=1500, ngram_range=(1, 3))

    df = pd.read_csv("processed_news.csv")
    sample_texts = df["processed_text"].dropna().tolist()

    try:
        # TF-IDF 분석 수행
        analyzer.fit_transform(sample_texts)

        # 통계 출력
        analyzer.print_statistics()

        # 키워드 추출
        keywords = analyzer.get_top_keywords_global(10)
        logger.info("\n상위 키워드:")
        for keyword, score in keywords:
            logger.info(f"  • {keyword}: {score:.3f}")

    except Exception as e:
        logger.error(f" 에러 발생: {e}")
