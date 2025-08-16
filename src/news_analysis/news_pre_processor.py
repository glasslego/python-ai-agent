import re

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

# import nltk
# print(nltk.data.path)  # NLTK 데이터 경로 출력

# 'punkt_tab' 토크나이저 다운로드 (사용하지 않는 경우 삭제 가능)
nltk.download("punkt_tab")
# 단어 토크나이저 다운로드
nltk.download("punkt")
# 영어 불용어(stopwords) 다운로드
nltk.download("stopwords")
# 워드넷(WordNet) 사전 다운로드 (어휘 정보 활용 시 필요)
nltk.download("wordnet")
# 품사 태깅을 위한 모델 다운로드
nltk.download("averaged_perceptron_tagger_eng")


class NewsTextPreprocessor:
    """뉴스 텍스트 전처리 클래스"""

    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.stop_words.update(
            [
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
            ]
        )

    def clean_text(self, text):
        """텍스트 정리"""
        if not isinstance(text, str):
            return ""

        # 소문자 변환
        text = text.lower()

        # HTML 태그 제거
        text = re.sub(r"<[^>]+>", "", text)

        # URL 제거
        text = re.sub(r"http\S+|www\S+|https\S+", "", text)

        # 특수문자 제거 (알파벳, 숫자, 공백만 유지)
        text = re.sub(r"[^a-zA-Z0-9\s]", "", text)

        # 연속된 공백 제거
        text = re.sub(r"\s+", " ", text).strip()

        return text

    def tokenize_and_filter(self, text):
        """토큰화 및 불용어 제거"""
        tokens = word_tokenize(text)

        # 불용어 제거 및 길이 필터링
        filtered_tokens = [
            token for token in tokens if token not in self.stop_words and len(token) > 2
        ]

        return filtered_tokens

    def preprocess_dataframe(self, df):
        """DataFrame 전체 전처리"""
        df_copy = df.copy()

        def process_row(row):
            content = str(row.get("Parsed Content", ""))
            description = str(row.get("Description", ""))
            title = str(row.get("Title", ""))

            combined_text = f"{title} {description} {content}"
            cleaned_text = self.clean_text(combined_text)
            return self.tokenize_and_filter(cleaned_text)

        df_copy["processed_content"] = df_copy.apply(process_row, axis=1)

        # 빈 내용 제거
        df_copy = df_copy[df_copy["processed_content"].str.len() > 0]

        print(f"✅ 전처리 완료: {len(df_copy)}개 기사")
        return df_copy


if __name__ == "__main__":
    df = pd.read_csv("it_news_articles.csv")
    processed_df = NewsTextPreprocessor().preprocess_dataframe(df)
    processed_df.to_csv("it_news_articles_processed.csv", index=False)
    print(processed_df.head())
