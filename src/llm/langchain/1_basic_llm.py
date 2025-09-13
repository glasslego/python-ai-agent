from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()


# ============================================================
# 1. 기본 LLM 사용하기
# ============================================================


def basic_llm_example():
    """
    가장 기본적인 LLM 호출 예제
    - 단순 질문/답변
    - 온도(temperature) 조절
    - 다양한 모델 사용법
    """

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # 단순 호출
    simple_response = llm.invoke(
        "2025년 현재 가장 인기있는 프로그래밍 언어 3개를 알려줘"
    )
    print("단순 질문 응답:")
    print(simple_response.content)

    return simple_response


def basic_llm_with_params():
    """
    파라미터를 조절한 LLM 호출 예제
    """

    # 창의적인 답변 (높은 temperature)
    creative_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.9,  # 0~1, 높을수록 창의적
        max_tokens=300,  # 최대 토큰 수
    )

    # 정확한 답변 (낮은 temperature)
    precise_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,  # 낮을수록 일관적
    )

    question = "AI의 미래는?"

    creative_answer = creative_llm.invoke(question)
    precise_answer = precise_llm.invoke(question)

    print(f"창의적 답변 (temp=0.9): {creative_answer.content[:300]}...")
    print(f"정확한 답변 (temp=0.1): {precise_answer.content[:300]}...")

    return creative_answer, precise_answer


def basic_llm_streaming():
    """
    스트리밍 방식으로 LLM 응답 받기
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        streaming=True,  # 스트리밍 활성화
    )

    print("스트리밍 응답:")
    for chunk in llm.stream("파이썬의 장점 3가지를 설명해줘"):
        print(chunk.content, end="", flush=True)
    print()


if __name__ == "__main__":
    basic_llm_example()
    basic_llm_with_params()
    basic_llm_streaming()
