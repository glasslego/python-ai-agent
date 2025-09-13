"""
뉴스 텍스트 전처리 모듈
"""

import re
from typing import List, Optional

import nltk
import pandas as pd
from loguru import logger
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize


class NewsPreprocessor:
    """뉴스 텍스트 전처리 클래스"""

    def __init__(self, download_nltk_data: bool = True):
        """
        Args:
            download_nltk_data: NLTK 데이터 자동 다운로드 여부
        """
        if download_nltk_data:
            self._download_nltk_requirements()

        self.stop_words = set(stopwords.words("english"))
        self._add_custom_stopwords()

    def _download_nltk_requirements(self):
        """필요한 NLTK 데이터 다운로드"""
        try:
            nltk.download("punkt_tab", quiet=True)
            nltk.download("punkt", quiet=True)
            nltk.download("stopwords", quiet=True)
            nltk.download("wordnet", quiet=True)
            nltk.download("averaged_perceptron_tagger_eng", quiet=True)
        except Exception as e:
            logger.info(f"NLTK 데이터 다운로드 중 에러: {e}")

    def _add_custom_stopwords(self):
        """커스텀 불용어 추가"""
        custom_stopwords = {
            "said",
            "says",
            "also",
            "would",
            "could",
            "one",
            "two",
            "new",
            "first",
            "last",
            "year",
            "years",
            "time",
            "way",
            "company",
            "companies",
            "people",
            "get",
            "make",
            "go",
            "see",
            "know",
            "take",
            "use",
            "work",
            "day",
            "part",
            "number",
            "system",
            "reuters",
            "ap",
            "associated",
            "press",
            "news",
            "report",
            "reports",
            "according",
            "sources",
            "source",
        }
        self.stop_words.update(custom_stopwords)

    def clean_text(self, text: str) -> str:
        """
        텍스트 정리

        Args:
            text: 원본 텍스트

        Returns:
            정리된 텍스트
        """
        if not isinstance(text, str) or not text.strip():
            return ""

        # 소문자 변환
        text = text.lower()

        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", "", text)

        # URL 제거
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # 이메일 제거
        text = re.sub(r"\S+@\S+", "", text)

        # 특수문자 제거 (알파벳, 숫자, 공백만 유지)
        text = re.sub(r"[^a-zA-Z0-9\s]", " ", text)

        # 연속된 공백을 하나로 통합
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_filter(self, text: str, min_length: int = 2) -> List[str]:
        """
        토큰화 및 불용어 제거

        Args:
            text: 입력 텍스트
            min_length: 최소 토큰 길이

        Returns:
            필터링된 토큰 리스트
        """
        if not text:
            return []

        try:
            tokens = word_tokenize(text)
        except Exception:
            # 토큰화 실패 시 공백으로 분할
            tokens = text.split()

        # 불용어 제거 및 길이 필터링
        filtered_tokens = [
            token
            for token in tokens
            if (
                token not in self.stop_words
                and len(token) >= min_length
                and token.isalpha()
            )  # 알파벳만 포함
        ]

        return filtered_tokens

    def preprocess_single_text(self, text: str) -> List[str]:
        """
        단일 텍스트 전처리

        Args:
            text: 입력 텍스트

        Returns:
            전처리된 토큰 리스트
        """
        cleaned_text = self.clean_text(text)
        return self.tokenize_and_filter(cleaned_text)

    def preprocess_dataframe(
        self, df: pd.DataFrame, text_columns: Optional[List[str]] = None
    ) -> pd.DataFrame:
        """
        DataFrame 전체 전처리

        Args:
            df: 입력 DataFrame
            text_columns: 처리할 텍스트 컬럼 리스트
                         (None인 경우 기본 컬럼들 사용)

        Returns:
            전처리가 완료된 DataFrame
        """
        if df.empty:
            return df

        df_copy = df.copy()

        # 기본 텍스트 컬럼 설정
        if text_columns is None:
            text_columns = ["Title", "Description", "Parsed Content", "Content"]

        def combine_and_process_row(row):
            """행의 여러 텍스트 컬럼을 결합하여 전처리"""
            texts = []

            for col in text_columns:
                if col in row and pd.notna(row[col]):
                    texts.append(str(row[col]))

            combined_text = " ".join(texts)
            return self.preprocess_single_text(combined_text)

        # 전처리 적용
        logger.info("🔄 텍스트 전처리를 진행합니다...")
        df_copy["processed_tokens"] = df_copy.apply(combine_and_process_row, axis=1)

        # 전처리된 텍스트를 문자열로 변환
        df_copy["processed_text"] = df_copy["processed_tokens"].apply(
            lambda tokens: " ".join(tokens) if tokens else ""
        )

        # 빈 내용 제거
        initial_count = len(df_copy)
        df_copy = df_copy[df_copy["processed_tokens"].str.len() > 0]
        final_count = len(df_copy)

        removed_count = initial_count - final_count
        if removed_count > 0:
            logger.warning(
                f"  빈 내용으로 인해 {removed_count}개 기사가 제거되었습니다."
            )

        logger.success(f" 전처리 완료: {final_count}개 기사")
        return df_copy

    def get_vocabulary(self, df: pd.DataFrame) -> dict:
        """
        전처리된 DataFrame에서 어휘 통계 추출

        Args:
            df: 전처리된 DataFrame

        Returns:
            어휘 통계 딕셔너리
        """
        if "processed_tokens" not in df.columns:
            return {}

        all_tokens = []
        for tokens in df["processed_tokens"]:
            all_tokens.extend(tokens)

        vocabulary = {
            "total_tokens": len(all_tokens),
            "unique_tokens": len(set(all_tokens)),
            "avg_tokens_per_doc": len(all_tokens) / len(df) if len(df) > 0 else 0,
        }

        return vocabulary

    def filter_by_frequency(
        self, df: pd.DataFrame, min_freq: int = 2, max_freq_ratio: float = 0.8
    ) -> pd.DataFrame:
        """
        빈도수 기반 토큰 필터링

        Args:
            df: 전처리된 DataFrame
            min_freq: 최소 출현 빈도
            max_freq_ratio: 최대 출현 비율 (문서 대비)

        Returns:
            필터링된 DataFrame
        """
        if "processed_tokens" not in df.columns:
            return df

        # 토큰 빈도 계산
        token_freq = {}
        for tokens in df["processed_tokens"]:
            for token in tokens:
                token_freq[token] = token_freq.get(token, 0) + 1

        # 문서 빈도 계산
        doc_freq = {}
        for tokens in df["processed_tokens"]:
            unique_tokens = set(tokens)
            for token in unique_tokens:
                doc_freq[token] = doc_freq.get(token, 0) + 1

        total_docs = len(df)
        max_doc_freq = int(total_docs * max_freq_ratio)

        # 필터링 조건 적용
        def filter_tokens(tokens):
            return [
                token
                for token in tokens
                if (
                    token_freq.get(token, 0) >= min_freq
                    and doc_freq.get(token, 0) <= max_doc_freq
                )
            ]

        df_copy = df.copy()
        df_copy["processed_tokens"] = df_copy["processed_tokens"].apply(filter_tokens)
        df_copy["processed_text"] = df_copy["processed_tokens"].apply(
            lambda tokens: " ".join(tokens)
        )

        # 빈 문서 제거
        df_copy = df_copy[df_copy["processed_tokens"].str.len() > 0]

        logger.success(f" 빈도 필터링 완료: {len(df_copy)}개 기사")
        return df_copy


if __name__ == "__main__":
    # 사용 예제
    preprocessor = NewsPreprocessor()

    # 단일 텍스트 전처리 예제
    sample_text = "This is a sample news article about technology and AI."
    processed_tokens = preprocessor.preprocess_single_text(sample_text)
    logger.info("전처리 결과:", processed_tokens)

    # DataFrame 전처리 예제 (CSV 파일이 있는 경우)
    try:
        df = pd.read_csv("../collected_news.csv")
        processed_df = preprocessor.preprocess_dataframe(df)

        if processed_df.empty:
            logger.warning(
                "전처리 결과 DataFrame이 비어 있습니다. CSV/전처리 단계를 확인하세요."
            )
        else:
            vocab_stats = preprocessor.get_vocabulary(processed_df)
            logger.info(
                "어휘 통계 | total={total_tokens}, unique={unique_tokens}, avg_per_doc={avg_tokens_per_doc:.2f}",  # noqa: E501
                **vocab_stats,
            )

        # 결과 저장
        processed_df.to_csv("processed_news.csv", index=False)
        logger.info("💾 processed_news.csv로 저장되었습니다.")

    except FileNotFoundError:
        logger.info("📄 뉴스 데이터 파일을 먼저 수집해주세요.")
