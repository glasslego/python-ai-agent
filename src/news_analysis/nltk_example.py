# flake8: noqa: E501
import nltk
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

# 필수 모듈 다운로드
nltk.download("punkt_tab")
nltk.download("punkt")
nltk.download("stopwords")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")


def preprocess_text(text):
    """완전한 텍스트 전처리 함수"""

    # 1. 문장 토큰화
    sentences = sent_tokenize(text)

    # 2. 단어 토큰화
    words1 = word_tokenize(text.lower())

    # 3. 불용어 제거 (알파벳이고 불용어가 아닌 단어만 선택)
    stop_words = set(stopwords.words("english"))
    words2 = [w for w in words1 if w.isalpha() and w not in stop_words]

    # 4. 어간 추출
    lemmatizer = WordNetLemmatizer()
    words3 = [lemmatizer.lemmatize(w) for w in words2]

    # 5. 품사 태깅
    pos_tags = pos_tag(words3)

    return {
        "sentences": sentences,
        "raw_words": words1,
        "filtered_words": words2,
        "processed_words": words3,
        "pos_tags": pos_tags,
    }


if __name__ == "__main__":
    # 사용 예시
    text = "Natural Language Processing is fascinating. It helps computers understand human language."
    result = preprocess_text(text)
    print(result)
