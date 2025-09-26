# flake8: noqa E501
"""
LangGraph 상태 관리 모듈

이 모듈은 스마트 어시스턴트의 그래프 실행 중 상태를 정의합니다.
상태는 각 노드 간에 전달되며, 전체 대화 흐름을 추적합니다.
"""

from typing import List, Optional, TypedDict

from langchain_core.messages import BaseMessage


class AssistantState(TypedDict):
    """
    LangGraph에서 사용하는 상태 정의 클래스

    각 노드가 실행될 때마다 이 상태 객체가 업데이트되며,
    노드 간에 정보를 전달하는 역할을 합니다.

    Attributes:
        messages (List[BaseMessage]): 대화 기록을 저장하는 메시지 리스트
            - 사용자와 어시스턴트 간의 모든 대화 내용
            - LangChain의 BaseMessage 형태로 저장

        user_input (str): 사용자가 입력한 원본 텍스트
            - 의도 분석과 엔티티 추출의 기본 데이터
            - 전체 처리 과정에서 참조되는 원본 입력

        intent (Optional[str]): 분석된 사용자 의도
            - "news", "weather", "exchange", "stock", "schedule", "general" 중 하나
            - analyze_intent_node에서 설정됨
            - None인 경우 의도 분석이 아직 완료되지 않음을 의미

        entities (dict): 추출된 엔티티 정보
            - 도시명, 날짜, 종목코드, 통화 등의 구조화된 정보
            - 예: {"city": "Seoul", "date": "2024-03-25", "currencies": ["USD", "KRW"]}
            - 각 도구 실행에 필요한 매개변수로 사용

        tool_results (List[str]): 도구 실행 결과 리스트
            - 각 도구가 반환한 결과를 문자열 형태로 저장
            - 여러 도구가 순차적으로 실행될 경우 결과가 누적됨
            - 최종 응답 생성의 기본 데이터가 됨

        final_response (Optional[str]): 사용자에게 전달될 최종 응답
            - 모든 처리가 완료된 후 생성되는 최종 답변
            - None인 경우 아직 응답이 생성되지 않음을 의미
            - 이 값이 설정되면 그래프 실행이 종료됨

        next_action (Optional[str]): 다음에 실행할 노드 또는 액션
            - 조건부 라우팅에서 사용되는 제어 플래그
            - "tool_selection", "tool_execution", "llm_response", "response_generation", "end"
            - 각 노드에서 다음 단계를 결정하는 데 사용

    사용 예시:
        initial_state: AssistantState = {
            "messages": [HumanMessage(content="서울 날씨 어때?")],
            "user_input": "서울 날씨 어때?",
            "intent": None,
            "entities": {},
            "tool_results": [],
            "final_response": None,
            "next_action": None
        }

    상태 흐름:
        1. 초기 상태: user_input만 설정
        2. 의도 분석 후: intent, entities 설정
        3. 도구 실행 후: tool_results 추가
        4. 응답 생성 후: final_response 설정
    """

    # 대화 기록 - LangChain 메시지 객체들의 리스트
    messages: List[BaseMessage]

    # 사용자 원본 입력 - 모든 분석의 기준이 되는 텍스트
    user_input: str

    # 분석된 의도 - 어떤 종류의 요청인지 분류
    intent: Optional[str]

    # 추출된 엔티티 - 의도 실행에 필요한 구조화된 정보
    entities: dict

    # 도구 실행 결과들 - API 호출 등의 결과물들
    tool_results: List[str]

    # 최종 응답 - 사용자에게 전달될 완성된 답변
    final_response: Optional[str]

    # 다음 액션 - 그래프 라우팅을 위한 제어 변수
    next_action: Optional[str]
