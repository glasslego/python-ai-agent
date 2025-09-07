# Word Embedding 예시 (리팩토링 버전)
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
from dotenv import load_dotenv
from gensim.models import Word2Vec
from gensim.models.keyedvectors import KeyedVectors
from openai import APIError, OpenAI

# --- 1. 설정 (Configuration) ---
# 설정값들을 상단에 상수로 정의하여 관리 용이성을 높입니다.
GLOVE_FILE_PATH = "glove.6B.100d.txt"
OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"
W2V_VECTOR_SIZE = 100
W2V_WINDOW = 5
W2V_MIN_COUNT = 1
W2V_WORKERS = 4

load_dotenv()


# --- 2. 유틸리티 함수 ---
# 독립적인 기능은 별도의 함수로 분리합니다.
def cosine_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """두 벡터 간 코사인 유사도를 계산합니다."""
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)

    # 0으로 나누는 것을 방지
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0

    return dot_product / (norm_vec1 * norm_vec2)


# --- 3. 임베딩 모델 클래스 ---
# 각 임베딩 방식을 클래스로 캡슐화합니다.


class Word2VecModel:
    """Word2Vec 모델을 학습하고 사용하는 클래스"""

    def __init__(self, vector_size: int, window: int, min_count: int, workers: int):
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.workers = workers
        self.model: Optional[Word2Vec] = None
        self.wv: Optional[KeyedVectors] = None

    def train(self, sentences: List[List[str]]):
        """주어진 문장들로 Word2Vec 모델을 학습시킵니다."""
        print("Word2Vec 모델 학습 시작...")
        self.model = Word2Vec(
            sentences,
            vector_size=self.vector_size,
            window=self.window,
            min_count=self.min_count,
            workers=self.workers,
        )
        self.wv = self.model.wv
        print("학습 완료.")

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """단어에 대한 임베딩 벡터를 반환합니다."""
        if self.wv and word in self.wv:
            return self.wv[word]
        return None

    def get_similar_words(self, word: str, topn: int = 3) -> List[Tuple[str, float]]:
        """주어진 단어와 가장 유사한 단어들을 반환합니다."""
        if self.wv and word in self.wv:
            return self.wv.most_similar(word, topn=topn)
        return []


class GloVeModel:
    """사전 학습된 GloVe 파일을 로드하여 사용하는 클래스"""

    def __init__(self, file_path: str):
        self.file_path = file_path
        self.glove_model: Dict[str, np.ndarray] = {}

    def load(self):
        """GloVe 파일을 로드하여 모델을 메모리에 올립니다."""
        print(f"GloVe 모델 로드 시작: {self.file_path}")
        try:
            with open(self.file_path, "r", encoding="utf-8") as f:
                for line in f:
                    parts = line.split()
                    word = parts[0]
                    embedding = np.array([float(val) for val in parts[1:]])
                    self.glove_model[word] = embedding
            print("로드 완료.")
        except FileNotFoundError:
            print(f"오류: GloVe 파일을 찾을 수 없습니다: {self.file_path}")
            print("다운로드: https://nlp.stanford.edu/projects/glove/")

    def get_embedding(self, word: str) -> Optional[np.ndarray]:
        """단어에 대한 임베딩 벡터를 반환합니다."""
        return self.glove_model.get(word)


class OpenAIEmbeddingModel:
    """OpenAI Embedding API를 사용하는 클래스"""

    def __init__(self, api_key: Optional[str]):
        if not api_key:
            raise ValueError("OPENAI_API_KEY가 필요합니다.")
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))

    def get_embedding(
        self, text: str, model: str = OPENAI_EMBEDDING_MODEL
    ) -> Optional[List[float]]:
        """API를 호출하여 텍스트에 대한 임베딩을 반환합니다."""
        try:
            response = self.client.embeddings.create(input=text, model=model)
            return response.data[0].embedding
        except APIError as e:
            print(f"OpenAI API 오류 발생: {e}")
            return None


# --- 4. 예제 실행 함수 ---
# 핵심 로직(클래스)과 예제 실행 로직을 분리합니다.


def run_word2vec_example():
    print("\n=== Word2Vec 예시 ===")
    sentences = [
        ["나는", "파이썬을", "좋아한다"],
        ["머신러닝은", "재미있다"],
        ["자연어", "처리는", "유용하다"],
        ["파이썬을", "배우고", "있다"],
    ]

    model = Word2VecModel(W2V_VECTOR_SIZE, W2V_WINDOW, W2V_MIN_COUNT, W2V_WORKERS)
    model.train(sentences)

    vector = model.get_embedding("파이썬을")
    if vector is not None:
        print(f"Word2Vec - '파이썬을' 벡터 shape: {vector.shape}")
        print(f"벡터 샘플 (처음 5개 값): {vector[:5]}")

    similar_words = model.get_similar_words("파이썬을")
    print(f"'파이썬을'과 유사한 단어: {similar_words}")


def run_glove_example():
    print("\n=== GloVe 예시 ===")
    model = GloVeModel(GLOVE_FILE_PATH)
    model.load()

    if not model.glove_model:
        print("GloVe 모델이 로드되지 않아 예제를 건너뜁니다.")
        return

    test_word = "computer"
    vector = model.get_embedding(test_word)
    if vector is not None:
        print(f"GloVe - '{test_word}' 벡터 shape: {vector.shape}")
        print(f"벡터 샘플 (처음 5개 값): {vector[:5]}")
    else:
        print(f"'{test_word}'를 GloVe 모델에서 찾을 수 없습니다.")


def run_openai_example():
    print("\n=== OpenAI Embedding 예시 ===")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수가 설정되지 않아 예제를 건너뜁니다.")
        return

    model = OpenAIEmbeddingModel(api_key)
    test_text = "자연어 처리는 인공지능의 중요한 분야입니다"
    embedding = model.get_embedding(test_text)

    if embedding:
        print(f"텍스트: '{test_text[:30]}...'")
        print(f"OpenAI Embedding 차원: {len(embedding)}")
        print(f"벡터 샘플 (처음 5개 값): {embedding[:5]}")


def run_cosine_similarity_example():
    print("\n=== 코사인 유사도 예시 ===")
    np.random.seed(42)
    vec1 = np.random.randn(100)
    vec2 = vec1 + np.random.randn(100) * 0.1  # vec1과 유사한 벡터
    vec3 = np.random.randn(100)  # vec1과 다른 벡터

    sim_12 = cosine_similarity(vec1, vec2)
    sim_13 = cosine_similarity(vec1, vec3)

    print(f"유사한 벡터 간의 유사도 (vec1 vs vec2): {sim_12:.4f}")
    print(f"다른 벡터 간의 유사도 (vec1 vs vec3): {sim_13:.4f}")


if __name__ == "__main__":
    run_word2vec_example()
    run_glove_example()
    run_openai_example()
    run_cosine_similarity_example()
