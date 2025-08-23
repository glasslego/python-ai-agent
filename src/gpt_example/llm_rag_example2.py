import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# API 키 설정 (환경변수 사용 권장)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class AdvancedRAG:
    """OpenAI Embedding을 사용한 고급 RAG"""

    def __init__(self):
        self.documents = []
        self.embeddings = []

    def get_embedding(self, text: str) -> List[float]:
        """OpenAI Embedding API를 사용하여 텍스트 임베딩 생성"""
        try:
            response = client.embeddings.create(
                model="text-embedding-ada-002", input=text
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"임베딩 생성 오류: {e}")
            return []

    def add_documents(self, documents: List[str]):
        """문서 추가 및 임베딩 생성"""
        self.documents = documents
        print("🔄 문서 임베딩 생성 중...")

        self.embeddings = []
        for doc in documents:
            embedding = self.get_embedding(doc)
            if embedding:
                self.embeddings.append(embedding)

        print(f"✅ {len(self.embeddings)}개 문서의 임베딩이 생성되었습니다.")

    def cosine_similarity(self, a: List[float], b: List[float]) -> float:
        """코사인 유사도 계산"""
        dot_product = sum(x * y for x, y in zip(a, b))
        magnitude_a = sum(x * x for x in a) ** 0.5
        magnitude_b = sum(x * x for x in b) ** 0.5

        if magnitude_a == 0 or magnitude_b == 0:
            return 0

        return dot_product / (magnitude_a * magnitude_b)

    def retrieve_relevant_documents(self, query: str, top_k: int = 2) -> List[str]:
        """질문과 관련된 문서 검색 (OpenAI Embedding 사용)"""
        if not self.documents or not self.embeddings:
            return []

        query_embedding = self.get_embedding(query)
        if not query_embedding:
            return []

        # 유사도 계산
        similarities = []
        for doc_embedding in self.embeddings:
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)

        # 가장 유사한 문서들 선택
        top_indices = sorted(
            range(len(similarities)), key=lambda i: similarities[i], reverse=True
        )[:top_k]

        relevant_docs = [
            self.documents[i] for i in top_indices if similarities[i] > 0.7
        ]  # 임계값 설정

        return relevant_docs


def advanced_rag_example():
    """고급 RAG 예시"""
    print("고급 RAG (OpenAI Embedding 사용) 예시")

    advanced_rag = AdvancedRAG()

    # 기술 관련 문서들
    tech_documents = [
        "Python은 1991년 귀도 반 로섬에 의해 개발된 프로그래밍 언어입니다. 간결하고 읽기 쉬운 문법이 특징입니다.",
        "머신러닝은 인공지능의 한 분야로, 컴퓨터가 데이터를 통해 스스로 학습하는 기술입니다.",
        "Docker는 컨테이너 기반의 가상화 플랫폼으로, 애플리케이션을 쉽게 배포하고 관리할 수 있게 해줍니다.",
        "React는 페이스북에서 개발한 사용자 인터페이스 구축을 위한 JavaScript 라이브러리입니다.",
    ]

    advanced_rag.add_documents(tech_documents)

    tech_questions = [
        "Python의 특징이 뭐야?",
        "머신러닝에 대해 설명해줘",
        "Docker의 장점은?",
        "React를 왜 사용하나?",
    ]

    for question in tech_questions:
        print(f"\n🔍 질문: {question}")
        relevant_docs = advanced_rag.retrieve_relevant_documents(question)
        print(f"📄 관련 문서: {relevant_docs}")


if __name__ == "__main__":
    advanced_rag_example()
