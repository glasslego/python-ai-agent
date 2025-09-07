# flake8: noqa E501

import os

import numpy as np
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# 환경변수 로드
load_dotenv()


def rag_with_metadata():
    """
    메타데이터를 포함한 RAG 예제
    """

    # 메타데이터가 포함된 문서
    documents_with_metadata = [
        {
            "content": "파이썬 3.12가 2023년 10월에 릴리즈되었습니다. 성능이 크게 향상되었습니다.",
            "source": "Python.org",
            "date": "2023-10-02",
            "category": "Release Notes",
        },
        {
            "content": "Django 5.0이 2023년 12월에 출시되었습니다. 많은 새로운 기능이 추가되었습니다.",
            "source": "Django Docs",
            "date": "2023-12-04",
            "category": "Framework",
        },
        {
            "content": "ChatGPT가 2022년 11월에 공개되어 AI 혁명을 일으켰습니다.",
            "source": "OpenAI Blog",
            "date": "2022-11-30",
            "category": "AI",
        },
        {
            "content": "FastAPI는 현대적이고 빠른 웹 API 프레임워크입니다. 타입 힌트를 활용합니다.",
            "source": "FastAPI Docs",
            "date": "2023-08-15",
            "category": "Framework",
        },
        {
            "content": "NumPy 2.0이 2024년에 출시 예정입니다. 주요 성능 개선이 포함됩니다.",
            "source": "NumPy News",
            "date": "2023-11-20",
            "category": "Library",
        },
    ]

    # 카테고리별 필터링 함수
    def filter_by_category(docs, category):
        return [doc for doc in docs if doc["category"] == category]

    # 날짜별 정렬 함수
    def sort_by_date(docs):
        return sorted(docs, key=lambda x: x["date"], reverse=True)

    # 날짜 범위 필터링 함수
    def filter_by_date_range(docs, start_date, end_date):
        return [doc for doc in docs if start_date <= doc["date"] <= end_date]

    # 키워드 검색 함수
    def search_by_keyword(docs, keyword):
        keyword_lower = keyword.lower()
        return [doc for doc in docs if keyword_lower in doc["content"].lower()]

    # RAG 프롬프트
    rag_template = """
    참고 문서:
    {context}

    질문: {question}

    답변 시 출처와 날짜를 명시해주세요.
    """

    prompt = PromptTemplate(
        input_variables=["context", "question"], template=rag_template
    )

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 다양한 검색 시나리오
    scenarios = [
        {
            "question": "최근 AI 관련 소식은 무엇인가요?",
            "filter_type": "category",
            "filter_value": "AI",
        },
        {
            "question": "프레임워크 관련 업데이트를 알려주세요",
            "filter_type": "category",
            "filter_value": "Framework",
        },
        {
            "question": "2023년에 있었던 주요 릴리즈는?",
            "filter_type": "date_range",
            "filter_value": ("2023-01-01", "2023-12-31"),
        },
        {
            "question": "파이썬과 관련된 소식을 찾아주세요",
            "filter_type": "keyword",
            "filter_value": "파이썬",
        },
    ]

    for scenario in scenarios:
        print(f"\n{'=' * 60}")
        print(f"❓ 질문: {scenario['question']}")

        # 필터링 적용
        filtered_docs = documents_with_metadata.copy()

        if scenario["filter_type"] == "category":
            filtered_docs = filter_by_category(filtered_docs, scenario["filter_value"])
            print(f"🏷️  필터: {scenario['filter_value']} 카테고리")

        elif scenario["filter_type"] == "date_range":
            start, end = scenario["filter_value"]
            filtered_docs = filter_by_date_range(filtered_docs, start, end)
            print(f"📅 필터: {start} ~ {end}")

        elif scenario["filter_type"] == "keyword":
            filtered_docs = search_by_keyword(filtered_docs, scenario["filter_value"])
            print(f"🔍 키워드: {scenario['filter_value']}")

        # 날짜순 정렬
        sorted_docs = sort_by_date(filtered_docs)

        if sorted_docs:
            print(f"📚 검색된 문서: {len(sorted_docs)}개")

            # 컨텍스트 생성 (메타데이터 포함)
            context_parts = []
            for i, doc in enumerate(sorted_docs, 1):
                context_part = f"""
문서 {i}:
- 출처: {doc["source"]}
- 날짜: {doc["date"]}
- 카테고리: {doc["category"]}
- 내용: {doc["content"]}
"""
                context_parts.append(context_part)
                print(
                    f"  📄 [{doc['date']}] {doc['source']} - {doc['content'][:50]}..."
                )

            context = "\n".join(context_parts)

            # LLM으로 답변 생성
            formatted_prompt = prompt.format(
                context=context, question=scenario["question"]
            )

            response = llm.invoke(formatted_prompt)
            print(f"\n📝 답변: {response.content}")
        else:
            print("📝 해당하는 문서를 찾을 수 없습니다.")

    # 추가: 복합 필터링 예제
    print(f"\n{'=' * 60}")
    print("🔧 복합 필터링 예제")
    print("='*60")

    # Framework 카테고리 중 2023년 문서만
    framework_docs = filter_by_category(documents_with_metadata, "Framework")
    recent_framework_docs = filter_by_date_range(
        framework_docs, "2023-01-01", "2023-12-31"
    )

    if recent_framework_docs:
        print("조건: Framework 카테고리 AND 2023년")
        print(f"결과: {len(recent_framework_docs)}개 문서")
        for doc in recent_framework_docs:
            print(f"  - [{doc['date']}] {doc['source']}: {doc['content'][:50]}...")

    return documents_with_metadata


# 추가: RAG 성능 개선을 위한 고급 예제
def rag_with_reranking():
    """
    재순위화(Reranking)를 포함한 고급 RAG 예제
    """

    # 확장된 문서 데이터베이스
    documents = [
        "파이썬은 귀도 반 로섬이 1991년 개발한 고급 프로그래밍 언어입니다.",
        "파이썬은 간결한 문법과 높은 가독성으로 초보자에게 인기가 많습니다.",
        "파이썬은 데이터 과학, 웹 개발, 자동화 등 다양한 분야에서 사용됩니다.",
        "자바스크립트는 브렌던 아이크가 1995년 개발한 웹 프로그래밍 언어입니다.",
        "러스트는 메모리 안전성을 보장하는 시스템 프로그래밍 언어입니다.",
        "Go 언어는 구글에서 개발한 동시성 처리에 강한 언어입니다.",
        "파이썬의 주요 라이브러리로는 NumPy, Pandas, TensorFlow가 있습니다.",
        "파이썬 3.x 버전은 2.x와 호환되지 않는 변경사항이 있습니다.",
    ]

    # TF-IDF 벡터화
    vectorizer = TfidfVectorizer()
    doc_vectors = vectorizer.fit_transform(documents)

    def initial_retrieval(query, top_k=5):
        """초기 검색: TF-IDF 기반"""
        query_vector = vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, doc_vectors)[0]
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0:  # 관련성이 있는 문서만
                results.append(
                    {
                        "document": documents[idx],
                        "initial_score": similarities[idx],
                        "index": idx,
                    }
                )
        return results

    def rerank_documents(query, initial_results, llm):
        """LLM을 사용한 재순위화"""
        rerank_template = """
        질문: {query}
        문서: {document}

        위 문서가 질문에 대한 답변에 얼마나 관련이 있는지 0-10 점수로 평가하세요.
        점수만 숫자로 답하세요.
        """

        prompt = PromptTemplate(
            input_variables=["query", "document"], template=rerank_template
        )

        reranked_results = []
        for result in initial_results:
            formatted_prompt = prompt.format(query=query, document=result["document"])

            try:
                response = llm.invoke(formatted_prompt)
                relevance_score = float(response.content.strip())
                result["rerank_score"] = relevance_score
                reranked_results.append(result)
            except Exception as e:
                print(f"재순위화 오류: {e}")
                result["rerank_score"] = 0
                reranked_results.append(result)

        # 재순위화 점수로 정렬
        reranked_results.sort(key=lambda x: x["rerank_score"], reverse=True)
        return reranked_results

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo", temperature=0, openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    # 질문 처리
    questions = [
        "파이썬의 창시자와 개발 연도를 알려주세요",
        "파이썬이 인기 있는 이유는 무엇인가요?",
        "메모리 안전성이 중요한 언어는?",
    ]

    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"❓ 질문: {question}")

        # 1단계: 초기 검색
        initial_results = initial_retrieval(question, top_k=5)
        print("\n📊 초기 검색 결과 (TF-IDF):")
        for i, result in enumerate(initial_results[:3], 1):
            print(
                f"  {i}. [점수: {result['initial_score']:.3f}] {result['document'][:50]}..."
            )

        # 2단계: 재순위화
        reranked_results = rerank_documents(question, initial_results, llm)
        print("\n📊 재순위화 결과 (LLM 평가):")
        for i, result in enumerate(reranked_results[:3], 1):
            print(
                f"  {i}. [점수: {result['rerank_score']:.1f}] {result['document'][:50]}..."
            )

        # 3단계: 최종 답변 생성
        top_docs = [r["document"] for r in reranked_results[:2]]
        context = "\n".join(top_docs)

        answer_template = """
        참고 문서:
        {context}

        질문: {question}

        위 문서를 바탕으로 간결하게 답변하세요.
        """

        answer_prompt = PromptTemplate(
            input_variables=["context", "question"], template=answer_template
        )

        formatted_prompt = answer_prompt.format(context=context, question=question)

        response = llm.invoke(formatted_prompt)
        print(f"\n📝 최종 답변: {response.content}")

    return vectorizer


# 테스트 함수
if __name__ == "__main__":
    print("RAG with Metadata 예제 실행")
    print("=" * 60)
    rag_with_metadata()

    print("\n\nRAG with Reranking 예제 실행")
    print("=" * 60)
    rag_with_reranking()
