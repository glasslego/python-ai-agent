from dotenv import load_dotenv
from langchain.prompts import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
)
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()


# ============================================================
# 2. 프롬프트 템플릿
# ============================================================


def prompt_template_basic():
    """
    기본 프롬프트 템플릿 사용 예제
    """

    # 템플릿 생성
    template = """
    당신은 {role}입니다.

    질문: {question}

    {style} 스타일로 답변해주세요.
    """

    prompt = PromptTemplate(
        input_variables=["role", "question", "style"], template=template
    )

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
    )

    # 다양한 조합으로 사용
    examples = [
        {
            "role": "과학 선생님",
            "question": "블랙홀이 뭐야?",
            "style": "초등학생도 이해할 수 있는",
        },
        {
            "role": "요리사",
            "question": "파스타 만드는 법",
            "style": "전문적이고 상세한",
        },
        {"role": "시인", "question": "봄의 아름다움", "style": "감성적이고 은유적인"},
    ]

    for example in examples:
        formatted = prompt.format(**example)
        response = llm.invoke(formatted)
        print(f"\n[{example['role']}의 {example['style']} 답변]")
        print(response.content[:200] + "...")

    return prompt


def prompt_template_chat():
    """
    대화형 프롬프트 템플릿 예제
    """

    # 시스템 메시지 템플릿
    system_template = "당신은 {expertise} 전문가입니다. 항상 {tone} 톤으로 대답하세요."
    system_prompt = SystemMessagePromptTemplate.from_template(system_template)

    # 사용자 메시지 템플릿
    human_template = "{question}"
    human_prompt = HumanMessagePromptTemplate.from_template(human_template)

    # 전체 대화 템플릿
    chat_prompt = ChatPromptTemplate.from_messages([system_prompt, human_prompt])

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
    )

    # 사용 예시
    messages = chat_prompt.format_messages(
        expertise="Python 프로그래밍",
        tone="친근하고 격려하는",
        question="코딩을 처음 시작하는데 어떻게 해야 할까요?",
    )

    response = llm.invoke(messages)
    print("대화형 템플릿 응답:")
    print(response.content)

    return chat_prompt


def prompt_template_few_shot():
    """
    Few-shot 프롬프트 템플릿 예제 (예시를 포함한 학습)
    """
    from langchain.prompts import FewShotPromptTemplate, PromptTemplate
    from langchain_openai import ChatOpenAI

    # 예시 데이터
    examples = [
        {"input": "happy", "output": "😊"},
        {"input": "sad", "output": "😢"},
        {"input": "angry", "output": "😠"},
        {"input": "love", "output": "❤️"},
    ]

    # 예시 템플릿
    example_template = """
    입력: {input}
    출력: {output}
    """

    example_prompt = PromptTemplate(
        input_variables=["input", "output"], template=example_template
    )

    # Few-shot 프롬프트
    few_shot_prompt = FewShotPromptTemplate(
        examples=examples,
        example_prompt=example_prompt,
        prefix="감정을 이모지로 변환하는 예시입니다:",
        suffix="입력: {emotion}\n출력:",
        input_variables=["emotion"],
    )

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0,
    )

    # 테스트
    test_emotions = ["excited", "tired", "surprised"]

    for emotion in test_emotions:
        prompt = few_shot_prompt.format(emotion=emotion)
        response = llm.invoke(prompt)
        print(f"{emotion} → {response.content}")

    return few_shot_prompt


if __name__ == "__main__":
    print("프롬프트 템플릿 예제 실행")
    prompt_template_basic()
    prompt_template_chat()
    prompt_template_few_shot()
