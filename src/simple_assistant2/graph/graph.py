"""
LangGraph 기반 스마트 어시스턴트 메인 그래프 모듈

이 모듈은 전체 대화 처리 워크플로우를 LangGraph로 구현합니다.
상태 기반 그래프를 통해 사용자 입력을 체계적으로 처리하고 응답을 생성합니다.

주요 구성 요소:
- SmartAssistantGraph: 메인 그래프 관리 클래스
- 조건부 라우팅: 의도에 따른 동적 워크플로우 제어
- 상태 관리: 각 단계별 정보 전달 및 추적

워크플로우:
사용자 입력 → 의도 분석 → 도구 선택 → 도구 실행 → 응답 생성
            ↘ 일반 대화 → LLM 응답
"""

import os
from typing import Literal

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import END, StateGraph

from src.simple_assistant2.graph.nodes import GraphNodes
from src.simple_assistant2.graph.state import AssistantState

# .env 파일에서 환경 변수 로드 (API 키 등)
load_dotenv()


class SmartAssistantGraph:
    """
    LangGraph를 사용한 스마트 어시스턴트 메인 그래프 클래스

    이 클래스는 전체 대화 처리 워크플로우를 관리합니다.
    상태 기반 그래프를 통해 사용자 입력을 단계별로 처리하고,
    각 단계의 결과에 따라 동적으로 다음 단계를 결정합니다.

    주요 기능:
    - 그래프 구조 정의 및 관리
    - 노드 간 조건부 라우팅
    - 상태 전달 및 추적
    - 오류 처리 및 복구

    Attributes:
        llm (ChatOpenAI): OpenAI GPT 모델 인스턴스
        nodes (GraphNodes): 모든 처리 노드들을 포함하는 클래스
        graph (CompiledGraph): 컴파일된 LangGraph 실행 그래프
    """

    def __init__(self):
        """
        SmartAssistantGraph 초기화

        1. OpenAI API 키 검증
        2. LLM 모델 설정
        3. 노드 클래스 초기화
        4. 그래프 구조 생성 및 컴파일

        Raises:
            ValueError: OPENAI_API_KEY 환경변수가 설정되지 않은 경우
        """
        # OpenAI API 키 확인 및 검증
        # .env 파일이나 시스템 환경변수에서 로드됨
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

        # OpenAI ChatGPT 모델 초기화
        # - model: gpt-3.5-turbo (비용 효율적이고 충분한 성능)
        # - temperature: 0.1 (일관되고 예측 가능한 응답을 위해 낮은 값)
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.1)

        # 모든 그래프 노드들을 포함하는 클래스 초기화
        # LLM 인스턴스를 전달하여 필요시 사용할 수 있도록 함
        self.nodes = GraphNodes(self.llm)

        # LangGraph 워크플로우 생성 및 컴파일
        # 모든 노드와 엣지가 정의된 실행 가능한 그래프 생성
        self.graph = self._create_graph()

    def _create_graph(self) -> StateGraph:
        """
        LangGraph 그래프를 생성하고 구성하는 내부 메서드

        전체 워크플로우를 정의하고 노드 간의 연결 관계를 설정합니다.
        조건부 라우팅을 통해 사용자 입력에 따라 동적으로 경로를 결정합니다.

        그래프 구조:
        1. analyze_intent (시작점) - 사용자 입력 의도 분석
        2. tool_selection - 의도에 따른 도구 선택
        3. tool_execution - 선택된 도구 실행
        4. response_generation - 도구 결과 기반 응답 생성
        5. llm_response - LLM 직접 응답 (일반 대화용)

        Returns:
            StateGraph: 컴파일된 실행 가능한 그래프
        """
        # AssistantState를 상태 타입으로 하는 상태 그래프 초기화
        workflow = StateGraph(AssistantState)

        # 각 처리 단계별 노드 추가
        # 노드명과 실제 처리 함수를 연결
        workflow.add_node("analyze_intent", self.nodes.analyze_intent_node)  # 의도 분석
        workflow.add_node("tool_selection", self.nodes.tool_selection_node)  # 도구 선택
        workflow.add_node("tool_execution", self.nodes.tool_execution_node)  # 도구 실행
        workflow.add_node("llm_response", self.nodes.llm_response_node)  # LLM 응답
        workflow.add_node(
            "response_generation", self.nodes.response_generation_node
        )  # 응답 생성

        # 그래프의 시작점 설정
        # 모든 사용자 입력은 의도 분석부터 시작
        workflow.set_entry_point("analyze_intent")

        # 조건부 엣지 설정 1: 의도 분석 후 라우팅
        # 의도에 따라 도구 사용 또는 직접 LLM 응답으로 분기
        workflow.add_conditional_edges(
            "analyze_intent",  # 시작 노드
            self._route_after_intent,  # 라우팅 결정 함수
            {
                "tool_selection": "tool_selection",  # 도구가 필요한 경우
                "llm_response": "llm_response",  # 일반 대화인 경우
            },
        )

        # 조건부 엣지 설정 2: 도구 선택 후 라우팅
        # 선택된 처리 방식에 따라 도구 실행 또는 LLM 응답으로 분기
        workflow.add_conditional_edges(
            "tool_selection",  # 시작 노드
            self._route_after_tool_selection,  # 라우팅 결정 함수
            {
                "tool_execution": "tool_execution",  # 도구 실행 필요
                "llm_response": "llm_response",  # LLM 직접 응답
            },
        )

        # 일반(고정) 엣지 설정: 항상 같은 다음 노드로 이동
        workflow.add_edge(
            "tool_execution", "response_generation"
        )  # 도구 실행 → 응답 생성
        workflow.add_edge("response_generation", END)  # 응답 생성 → 종료
        workflow.add_edge("llm_response", END)  # LLM 응답 → 종료

        # 그래프 컴파일: 정의된 구조를 실행 가능한 형태로 변환
        # 컴파일 과정에서 순환 참조 검사, 연결성 확인 등이 수행됨
        return workflow.compile()

    def _route_after_intent(
        self, state: AssistantState
    ) -> Literal["tool_selection", "llm_response"]:
        """
        의도 분석 후 다음 노드를 결정하는 라우팅 함수

        분석된 사용자 의도에 따라 처리 경로를 결정합니다:
        - 일반 대화: LLM으로 직접 응답 생성
        - 특정 기능: 도구 선택 단계로 진행

        Args:
            state (AssistantState): 의도 분석이 완료된 현재 상태

        Returns:
            Literal: "tool_selection" 또는 "llm_response"
        """
        # 상태에서 분석된 의도 추출, 기본값은 "general"
        intent = state.get("intent", "general")

        # 일반 대화인 경우 도구 없이 LLM으로 직접 응답
        if intent == "general":
            return "llm_response"

        # 특정 기능(뉴스, 날씨, 주식 등)인 경우 도구 선택으로 진행
        return "tool_selection"

    def _route_after_tool_selection(
        self, state: AssistantState
    ) -> Literal["tool_execution", "llm_response"]:
        """
        도구 선택 후 다음 노드를 결정하는 라우팅 함수

        tool_selection_node에서 설정한 next_action 값에 따라
        실제 도구 실행 또는 LLM 응답으로 분기합니다.

        Args:
            state (AssistantState): 도구 선택이 완료된 현재 상태

        Returns:
            Literal: "tool_execution" 또는 "llm_response"
        """
        # tool_selection_node에서 설정한 다음 액션 확인
        next_action = state.get("next_action", "llm_response")

        # 도구 실행이 필요한 경우
        if next_action == "tool_execution":
            return "tool_execution"

        # 그 외의 경우는 LLM 응답으로 처리
        return "llm_response"

    def chat(self, user_input: str) -> str:
        """
        사용자 입력을 처리하고 응답을 반환하는 메인 인터페이스 메서드

        이 메서드는 외부에서 호출되는 주요 진입점입니다.
        사용자의 텍스트 입력을 받아 LangGraph를 통해 처리하고
        최종 응답을 문자열로 반환합니다.

        처리 과정:
        1. 초기 상태 설정
        2. LangGraph 실행 (여러 노드를 거쳐 처리)
        3. 최종 결과 추출 및 반환
        4. 오류 발생 시 안전한 에러 메시지 반환

        Args:
            user_input (str): 사용자가 입력한 텍스트

        Returns:
            str: 처리된 최종 응답 문자열

        Examples:
            >>> assistant = SmartAssistantGraph()
            >>> response = assistant.chat("서울 날씨 어때?")
            >>> print(response)
            "서울 현재 날씨: clear sky, 온도: 15°C, 습도: 60%"
        """
        # LangGraph 실행을 위한 초기 상태 설정
        # 모든 필드를 명시적으로 초기화하여 예상 가능한 동작 보장
        initial_state: AssistantState = {
            "messages": [HumanMessage(content=user_input)],  # 대화 기록 시작점
            "user_input": user_input,  # 원본 사용자 입력
            "intent": None,  # 의도 분석 결과 (아직 미분석)
            "entities": {},  # 추출된 엔티티들 (아직 미추출)
            "tool_results": [],  # 도구 실행 결과들 (아직 없음)
            "final_response": None,  # 최종 응답 (아직 생성 안됨)
            "next_action": None,  # 다음 액션 지시자 (초기값)
        }

        try:
            # LangGraph 실행: 초기 상태를 입력으로 전달하여 전체 워크플로우 실행
            # 그래프는 정의된 노드들을 순서대로 (또는 조건에 따라) 실행
            # 각 노드는 상태를 업데이트하며 다음 노드로 전달
            final_state = self.graph.invoke(initial_state)

            # 최종 상태에서 응답 추출
            # get() 메서드로 안전하게 접근하여 키가 없을 때 기본값 반환
            return final_state.get("final_response", "응답을 생성할 수 없습니다.")

        except Exception as e:
            # 그래프 실행 중 발생할 수 있는 모든 예외 처리
            # API 오류, 네트워크 문제, 구성 오류 등을 포함
            # 사용자에게는 친화적인 오류 메시지 제공
            return f"처리 중 오류가 발생했습니다: {str(e)}"
