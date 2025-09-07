import uuid

from dotenv import load_dotenv
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()

# 세션별 메모리 저장소
store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    """세션 ID별로 메모리 히스토리를 관리"""
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]


# ============================================================
# 4. 메모리 (Memory) - 최신 방식
# ============================================================


def memory_conversation_buffer():
    """
    대화 버퍼 메모리 예제 - 전체 대화 내용 저장 (최신 방식)
    """

    # LLM 초기화
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # 프롬프트 템플릿 생성
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 도움이 되는 AI 어시스턴트입니다. 이전 대화 내용을 기억하고 맥락에 맞게 응답하세요.",
            ),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    # 체인 생성
    chain = prompt | llm

    # 메모리가 있는 실행 가능한 체인 생성
    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 세션 ID
    session_id = str(uuid.uuid4())

    # 대화 시뮬레이션
    conversations = [
        "안녕하세요! 저는 파이썬을 배우고 있는 학생입니다.",
        "제가 뭘 배우고 있다고 했죠?",
        "초보자에게 추천하는 프로젝트가 있나요?",
        "앞서 제가 누구라고 소개했나요?",
    ]

    print("=== 대화 버퍼 메모리 예제 ===")
    for user_input in conversations:
        print(f"\n👤 사용자: {user_input}")

        response = chain_with_history.invoke(
            {"input": user_input}, config={"configurable": {"session_id": session_id}}
        )

        print(f"🤖 AI: {response.content}")

    # 메모리 내용 확인
    print("\n📝 저장된 대화 내용:")
    history = get_session_history(session_id)
    for message in history.messages:
        if isinstance(message, HumanMessage):
            print(f"👤: {message.content}")
        elif isinstance(message, AIMessage):
            print(f"🤖: {message.content[:100]}...")

    return chain_with_history


def memory_conversation_summary():
    """
    대화 요약 메모리 예제 - 대화를 요약해서 저장 (수동 구현)
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # 요약을 위한 별도 체인
    summary_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "다음 대화 내용을 간결하게 요약해주세요. 중요한 정보만 포함하세요.",
            ),
            ("human", "{conversation}"),
        ]
    )
    summary_chain = summary_prompt | llm

    # 메인 대화 프롬프트
    conversation_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 도움이 되는 AI 어시스턴트입니다. 다음은 이전 대화의 요약입니다: {summary}",
            ),
            ("human", "{input}"),
        ]
    )
    conversation_chain = conversation_prompt | llm

    # 대화 히스토리와 요약 저장
    conversation_history = []
    summary = "아직 대화가 없습니다."

    # 긴 대화 진행
    long_conversation = [
        "저는 웹 개발자이고, 최근에 React를 공부하기 시작했습니다.",
        "React에서 상태 관리는 어떻게 하나요?",
        "Redux와 Context API의 차이점은 무엇인가요?",
        "제가 처음에 무엇을 공부한다고 했었나요?",
    ]

    print("=== 대화 요약 메모리 예제 ===")
    for i, user_input in enumerate(long_conversation):
        print(f"\n👤 사용자: {user_input}")

        # 대화 실행
        response = conversation_chain.invoke({"input": user_input, "summary": summary})

        print(f"🤖 AI: {response.content[:200]}...")

        # 대화 히스토리에 추가
        conversation_history.append(f"Human: {user_input}")
        conversation_history.append(f"AI: {response.content}")

        # 3번째 대화부터 요약 시작
        if i >= 2:
            full_conversation = "\n".join(conversation_history)
            summary_response = summary_chain.invoke({"conversation": full_conversation})
            summary = summary_response.content
            print(f"📄 현재 요약: {summary[:100]}...")

    return conversation_chain


def memory_window_buffer():
    """
    윈도우 버퍼 메모리 예제 - 최근 K개의 대화만 기억 (수동 구현)
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    # 프롬프트 템플릿
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "당신은 도움이 되는 AI 어시스턴트입니다. 다음은 최근 대화 내용입니다: {recent_history}",
            ),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    # 최근 대화만 저장 (k=2, 즉 최근 2번의 대화쌍)
    k = 2
    recent_messages = []

    # 여러 대화 진행
    conversations = [
        "제 이름은 김철수입니다.",
        "저는 서울에 살고 있어요.",
        "파이썬을 배우고 있습니다.",
        "제 이름이 뭐라고 했죠?",  # 이미 메모리에서 사라짐
        "제가 어디 산다고 했죠?",  # 아직 메모리에 있음
    ]

    print("=== 윈도우 버퍼 메모리 예제 (최근 2개 대화만 기억) ===")
    for i, user_input in enumerate(conversations, 1):
        print(f"\n대화 {i}")
        print(f"👤 사용자: {user_input}")

        # 최근 대화 히스토리 구성
        recent_history = (
            "\n".join(recent_messages[-k * 2 :])
            if recent_messages
            else "이전 대화가 없습니다."
        )

        response = chain.invoke({"input": user_input, "recent_history": recent_history})

        print(f"🤖 AI: {response.content[:150]}...")

        # 메시지 히스토리에 추가
        recent_messages.append(f"Human: {user_input}")
        recent_messages.append(f"AI: {response.content}")

        # 윈도우 크기만큼만 유지 (k=2이므로 4개 메시지: 2개 대화쌍)
        if len(recent_messages) > k * 2:
            recent_messages = recent_messages[-k * 2 :]

        print(f"💾 현재 메모리: {len(recent_messages) // 2}개 대화쌍 저장됨")
        print(f"📝 저장된 내용: {[msg[:50] + '...' for msg in recent_messages[-4:]]}")

    return chain


def memory_custom_session():
    """
    커스텀 세션 관리 예제 - 여러 사용자의 대화를 별도로 관리
    """

    llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.7,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "당신은 도움이 되는 AI 어시스턴트입니다."),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ]
    )

    chain = prompt | llm

    chain_with_history = RunnableWithMessageHistory(
        chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="history",
    )

    # 서로 다른 사용자 시뮬레이션
    users = {
        "user1": ["안녕하세요, 저는 김철수입니다.", "제 이름을 기억하고 계신가요?"],
        "user2": ["안녕하세요, 저는 이영희입니다.", "제 이름은 무엇인가요?"],
    }

    print("=== 멀티 세션 메모리 예제 ===")
    for user_id, messages in users.items():
        print(f"\n--- {user_id} 대화 ---")
        for message in messages:
            print(f"👤 {user_id}: {message}")

            response = chain_with_history.invoke(
                {"input": message}, config={"configurable": {"session_id": user_id}}
            )

            print(f"🤖 AI: {response.content}")

    # 각 사용자별 메모리 확인
    print("\n📝 저장된 세션들:")
    for session_id in store.keys():
        print(f"{session_id}: {len(store[session_id].messages)}개 메시지")

    return chain_with_history


if __name__ == "__main__":
    memory_conversation_buffer()
    print("\n" + "=" * 50 + "\n")
    memory_conversation_summary()
    print("\n" + "=" * 50 + "\n")
    memory_window_buffer()
    print("\n" + "=" * 50 + "\n")
    memory_custom_session()
