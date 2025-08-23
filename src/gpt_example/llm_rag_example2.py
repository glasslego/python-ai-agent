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
            # OpenAI Embeddings API 호출
            # text-embedding-ada-002: OpenAI의 최신 임베딩 모델
            response = client.embeddings.create(
                model="text-embedding-ada-002",  # 임베딩 모델 지정
                input=text,  # 변환할 텍스트 입력
            )
            embedding_vector = response.data[0].embedding

            # 임베딩 벡터 정보 출력 (디버깅용)
            print(f"임베딩 생성 완료 - 차원: {len(embedding_vector)}")
            print(f"텍스트: '{text[:50]}...' (처음 50자)")
            print(
                f"벡터 예시: [{embedding_vector[0]:.6f}, {embedding_vector[1]:.6f}, ...]"
            )

            return embedding_vector
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
        """
        질문과 관련된 문서 검색 (OpenAI Embedding 사용)

        Args:
            query (str): 검색할 질문
            top_k (int): 반환할 상위 문서 개수

        Returns:
            List[str]: 관련성이 높은 문서들 (유사도 순서대로 정렬)
        """
        if not self.documents or not self.embeddings:
            print("⚠️  저장된 문서나 임베딩이 없습니다.")
            return []

        print(f"🔍 질문 임베딩 생성 중: '{query}'")

        # 1. 질문을 임베딩으로 변환
        query_embedding = self.get_embedding(query)
        if not query_embedding:
            print("❌ 질문 임베딩 생성 실패")
            return []

        print("📊 문서들과 유사도 계산 중...")

        # 2. 모든 문서와 질문 간의 유사도 계산
        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = self.cosine_similarity(query_embedding, doc_embedding)
            similarities.append(similarity)
            print(f"   문서 {i + 1}: 유사도 = {similarity:.4f}")

        # 3. 유사도 기준으로 상위 k개 문서 선택
        # enumerate로 인덱스와 유사도를 함께 정렬
        indexed_similarities = list(enumerate(similarities))
        # 유사도 기준 내림차순 정렬 후 상위 k개 선택
        top_indices = sorted(
            indexed_similarities,
            key=lambda x: x[1],  # 유사도(두 번째 요소)로 정렬
            reverse=True,  # 내림차순
        )[:top_k]

        print(f"🎯 상위 {top_k}개 문서 선택:")

        # 4. 임계값 적용하여 관련성이 높은 문서만 선택
        relevant_docs = []
        similarity_threshold = 0.7  # 유사도 임계값 (70% 이상)

        for idx, similarity in top_indices:
            print(
                f"   순위 {len(relevant_docs) + 1}: 문서 {idx + 1} (유사도: {similarity:.4f})"
            )

            if similarity > similarity_threshold:
                relevant_docs.append(self.documents[idx])
                print("     ✅ 임계값 통과 - 선택됨")
            else:
                print(
                    f"     ❌ 임계값 미달 ({similarity:.4f} <= {similarity_threshold}) - 제외"
                )

        if not relevant_docs:
            print(f"⚠️  임계값 {similarity_threshold} 이상인 관련 문서가 없습니다.")

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
