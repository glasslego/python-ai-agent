import os
from typing import List

from dotenv import load_dotenv
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()

# API 키 설정 (환경변수 사용 권장)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class SimpleRAG:
    """간단한 RAG 시스템 구현"""

    def __init__(self):
        self.documents = []
        self.vectorizer = TfidfVectorizer()
        self.document_vectors = None

    def add_documents(self, documents: List[str]):
        """문서를 추가하고 벡터화"""
        self.documents = documents
        # TF-IDF 벡터화 (실제로는 OpenAI embedding을 사용하는 것이 더 좋음)
        self.document_vectors = self.vectorizer.fit_transform(documents)
        print(f"📚 {len(documents)}개의 문서가 추가되었습니다.")

    def retrieve_relevant_documents(self, query: str, top_k: int = 2) -> List[str]:
        """질문과 관련된 문서 검색"""
        if not self.documents:
            return []

        # 질문을 벡터로 변환
        query_vector = self.vectorizer.transform([query])

        # 코사인 유사도 계산
        similarities = cosine_similarity(query_vector, self.document_vectors)[0]

        # 가장 유사한 문서들의 인덱스 찾기
        top_indices = similarities.argsort()[-top_k:][::-1]

        # 관련 문서 반환
        relevant_docs = [
            self.documents[i] for i in top_indices if similarities[i] > 0.1
        ]
        return relevant_docs

    def generate_answer(self, query: str) -> str:
        """RAG를 사용하여 답변 생성"""
        # 1단계: 관련 문서 검색
        relevant_docs = self.retrieve_relevant_documents(query)

        if not relevant_docs:
            context = "관련 정보를 찾을 수 없습니다."
        else:
            context = "\n\n".join(relevant_docs)

        # 2단계: 검색된 정보와 함께 LLM에 질문
        prompt = f"""
다음 정보를 바탕으로 질문에 답변해주세요.

참고 정보:
{context}

질문: {query}

답변:"""

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 주어진 정보를 바탕으로 정확하게 답변하는 AI 어시스턴트입니다.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=500,
                temperature=0.3,
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ 오류가 발생했습니다: {str(e)}"


def main():
    print("RAG (Retrieval-Augmented Generation) 예시")

    rag_system = SimpleRAG()

    # 회사 관련 문서들 추가
    company_documents = [
        "우리 회사의 근무시간은 오전 9시부터 오후 6시까지입니다. 점심시간은 오후 12시부터 1시까지입니다.",
        "연차는 입사 1년 후부터 15일이 주어지며, 매년 1일씩 추가로 증가합니다. 최대 25일까지 가능합니다.",
        "회사 식당은 지하 1층에 있으며, 조식은 8시-9시, 중식은 11시30분-1시30분, 석식은 5시30분-7시30분에 운영됩니다.",
        "주차장은 지하 2층과 3층에 있으며, 직원용 주차는 무료입니다. 방문자용 주차는 1시간당 2000원입니다.",
        "회사 내 흡연은 금지되어 있으며, 지정된 흡연구역은 건물 옥상에 있습니다.",
    ]

    rag_system.add_documents(company_documents)

    # RAG 질문 예시들
    rag_questions = [
        "회사 근무시간이 어떻게 되나요?",
        "연차는 몇 일까지 받을 수 있나요?",
        "회사 식당 운영시간을 알려주세요",
        "주차비는 얼마인가요?",
    ]

    for question in rag_questions:
        print(f"\n🔍 질문: {question}")
        answer = rag_system.generate_answer(question)
        print(f"💬 RAG 답변: {answer}")


if __name__ == "__main__":
    main()
