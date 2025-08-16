import os
from typing import Dict, List

import numpy as np
import pandas as pd
from openai import OpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")


class RAGMBTIAgent:
    def __init__(self, api_key: str = None, csv_path: str = "mbti_types.csv"):
        self.client = OpenAI(api_key=api_key or os.getenv("OPENAI_API_KEY"))
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words="english")

        # CSV 데이터 로드 및 벡터화
        self.mbti_data = pd.read_csv(csv_path)
        self.document_vectors = self._vectorize_documents()

    def _vectorize_documents(self):
        """MBTI 데이터를 TF-IDF 벡터로 변환"""
        documents = self.mbti_data["description"].fillna("").tolist()
        return self.vectorizer.fit_transform(documents)

    def _retrieve_relevant_docs(self, query: str, top_k: int = 3) -> List[Dict]:
        """쿼리와 유사한 MBTI 정보 검색"""
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.document_vectors).flatten()
        top_indices = np.argsort(similarities)[::-1][:top_k]

        relevant_docs = []
        for idx in top_indices:
            relevant_docs.append(
                {
                    "mbti_type": self.mbti_data.iloc[idx]["type"],
                    "description": self.mbti_data.iloc[idx]["description"],
                    "traits": self.mbti_data.iloc[idx]["traits"],
                    "similarity": similarities[idx],
                }
            )
        return relevant_docs

    def analyze_personality(self, user_text: str) -> str:
        """RAG 기반 성격 분석"""
        relevant_docs = self._retrieve_relevant_docs(user_text)

        context = "\n".join(
            [
                f"MBTI {doc['mbti_type']}: {doc['description'][:200]}..."
                for doc in relevant_docs
            ]
        )

        prompt = f"""
        참고 자료:
        {context}

        사용자 텍스트: {user_text}

        위 자료를 바탕으로 사용자의 MBTI 유형을 분석하고 다음 형식으로 답변:
        - 추정 MBTI:
        - 근거:
        - 확신도 (1-10):
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 MBTI 전문 분석가입니다. 제공된 자료를 참고하여 정확한 분석을 제공하세요.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )

        return response.choices[0].message.content

    def get_mbti_info(self, mbti_type: str) -> str:
        """특정 MBTI 유형 정보 검색"""
        mbti_type = mbti_type.upper()

        # 해당 MBTI 타입 데이터 검색
        mbti_row = self.mbti_data[self.mbti_data["type"] == mbti_type]

        if mbti_row.empty:
            return f"'{mbti_type}' 데이터를 찾을 수 없습니다."

        mbti_info = mbti_row.iloc[0]

        prompt = f"""
        MBTI 유형: {mbti_info["type"]}
        설명: {mbti_info["description"]}
        주요 특성: {mbti_info["traits"]}
        강점: {mbti_info.get("strengths", "N/A")}
        약점: {mbti_info.get("weaknesses", "N/A")}

        위 정보를 바탕으로 {mbti_type} 유형에 대해 친근하게 설명해주세요.
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "MBTI 정보를 이해하기 쉽고 친근하게 설명하는 전문가입니다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.4,
            max_tokens=500,
        )

        return response.choices[0].message.content

    def compatibility_analysis(self, type1: str, type2: str) -> str:
        """RAG 기반 궁합 분석"""
        type1, type2 = type1.upper(), type2.upper()

        # 두 타입의 정보 검색
        info1 = self.mbti_data[self.mbti_data["type"] == type1]
        info2 = self.mbti_data[self.mbti_data["type"] == type2]

        if info1.empty or info2.empty:
            return "올바른 MBTI 유형을 입력해주세요."

        context = f"""
        {type1} 정보:
        - 특성: {info1.iloc[0]["traits"]}
        - 강점: {info1.iloc[0].get("strengths", "N/A")}

        {type2} 정보:
        - 특성: {info2.iloc[0]["traits"]}
        - 강점: {info2.iloc[0].get("strengths", "N/A")}
        """

        prompt = f"""
        {context}

        위 정보를 바탕으로 {type1}과 {type2}의 궁합을 분석해주세요:
        - 궁합 점수 (1-10):
        - 잘 맞는 점:
        - 주의할 점:
        - 관계 개선 팁:
        """

        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "MBTI 궁합 전문가로서 객관적이고 균형잡힌 분석을 제공합니다.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.5,
            max_tokens=500,
        )

        return response.choices[0].message.content

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        """의미적 검색으로 관련 MBTI 정보 찾기"""
        relevant_docs = self._retrieve_relevant_docs(query, top_k)

        results = []
        for doc in relevant_docs:
            if doc["similarity"] > 0.1:  # 임계값 설정
                results.append(
                    {
                        "type": doc["mbti_type"],
                        "description": doc["description"][:150] + "...",
                        "relevance": round(doc["similarity"], 3),
                    }
                )

        return results


# 사용 예시
def main():
    # RAG MBTI Agent 초기화
    agent = RAGMBTIAgent(api_key=OPENAI_API_KEY)

    # 사용 예시
    user_input = "나는 사람들과 어울리는 것을 좋아하고 책임감이 강한편이고. 감정적인 편이야. 난 어떤 MBTI 유형일까?"

    # 성격 분석
    result = agent.analyze_personality(user_input)
    print("성격 분석 결과:")
    print(result)

    # MBTI 정보 검색
    mbti_info = agent.get_mbti_info("ESFJ")
    print("\nINTJ 정보:")
    print(mbti_info)

    # 궁합 분석
    compatibility = agent.compatibility_analysis("ISFJ", "ESFJ")
    print("\n궁합 분석:")
    print(compatibility)

    # 의미적 검색
    search_results = agent.semantic_search("창의적이고 예술적인 성격을 가진 사람")
    print("\n검색 결과:")
    for result in search_results:
        print(
            f"- {result['type']}: {result['description']} (관련도: {result['relevance']})"
        )


if __name__ == "__main__":
    main()
