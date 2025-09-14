# =============================================================================
# LangChain vs LangGraph 비교 예시
# =============================================================================

from typing import Any, Dict, List, TypedDict

from langgraph.graph import END, StateGraph

# =============================================================================
# 1. LangChain 방식 - 선형적 체인
# =============================================================================


class LangChainExample:
    """LangChain의 선형적 접근 방식"""

    def simple_weather_chain(self, user_input: str) -> str:
        """간단한 날씨 조회 체인"""
        # A → B → C 순서로 고정

        # 1단계: 의도 파악
        intent = self.analyze_intent(user_input)

        # 2단계: API 호출
        weather_data = self.call_weather_api(intent["city"])

        # 3단계: 응답 생성
        response = self.generate_response(weather_data)

        return response

    def analyze_intent(self, text: str) -> Dict:
        """의도 분석 (Mock)"""
        return {"intent": "weather", "city": "서울"}

    def call_weather_api(self, city: str) -> Dict:
        """날씨 API 호출 (Mock)"""
        return {"city": city, "weather": "맑음", "temp": 22}

    def generate_response(self, data: Dict) -> str:
        """응답 생성"""
        return f"{data['city']}의 날씨는 {data['weather']}, {data['temp']}°C입니다."


# LangChain의 문제점들을 보여주는 복잡한 시나리오
class ComplexLangChainExample:
    """복잡한 시나리오에서 LangChain의 한계"""

    def complex_task_chain(self, user_input: str) -> str:
        """복잡한 작업: 날씨 확인 후 조건부 일정 추가"""

        # 문제 1: 선형적 구조로만 처리 가능
        intent = self.analyze_intent(user_input)
        weather = self.get_weather(intent["city"])

        # 문제 2: 조건부 로직이 복잡해짐
        if weather["condition"] == "비":
            # 우산 일정 추가
            schedule_result = self.add_umbrella_schedule()
            if not schedule_result["success"]:
                # 문제 3: 실패 시 재시도 로직이 어려움
                return "일정 추가에 실패했습니다"
        else:
            # 산책 일정 추가
            self.add_walk_schedule()

        # 문제 4: 사용자 확인 후 수정이 어려움
        return self.generate_final_response(weather)

    def analyze_intent(self, text: str) -> Dict:
        return {"intent": "weather_and_schedule", "city": "서울"}

    def get_weather(self, city: str) -> Dict:
        return {"city": city, "condition": "비", "temp": 15}

    def add_umbrella_schedule(self) -> Dict:
        return {"success": False, "reason": "시간 충돌"}

    def add_walk_schedule(self) -> Dict:
        return {"success": True}

    def generate_final_response(self, weather: Dict) -> str:
        return "날씨 확인 및 일정 추가 완료"


# =============================================================================
# 2. LangGraph 방식 - 상태 기반 그래프
# =============================================================================


class ConversationState(TypedDict):
    """대화 상태 정의"""

    messages: List[str]
    user_input: str
    intent: str
    city: str
    weather_data: Dict[str, Any]
    schedule_attempts: int
    final_response: str
    next_action: str


class LangGraphExample:
    """LangGraph를 사용한 상태 기반 워크플로우"""

    def __init__(self):
        self.graph = self.build_graph()

    def build_graph(self) -> StateGraph:
        """복잡한 워크플로우 그래프 구성"""

        # 그래프 생성
        workflow = StateGraph(ConversationState)

        # 노드들 추가
        workflow.add_node("analyze_intent", self.analyze_intent_node)
        workflow.add_node("get_weather", self.get_weather_node)
        workflow.add_node("decide_action", self.decide_action_node)
        workflow.add_node("add_umbrella_schedule", self.add_umbrella_schedule_node)
        workflow.add_node("add_walk_schedule", self.add_walk_schedule_node)
        workflow.add_node("handle_schedule_failure", self.handle_schedule_failure_node)
        workflow.add_node("generate_response", self.generate_response_node)
        workflow.add_node("ask_user_confirmation", self.ask_user_confirmation_node)

        # 엣지(연결) 정의
        workflow.set_entry_point("analyze_intent")
        workflow.add_edge("analyze_intent", "get_weather")
        workflow.add_edge("get_weather", "decide_action")

        # 조건부 엣지
        workflow.add_conditional_edges(
            "decide_action",
            self.route_decision,
            {"rainy": "add_umbrella_schedule", "sunny": "add_walk_schedule"},
        )

        # 실패 처리 및 재시도 로직
        workflow.add_conditional_edges(
            "add_umbrella_schedule",
            self.check_schedule_success,
            {
                "success": "generate_response",
                "failure": "handle_schedule_failure",
                "retry": "add_umbrella_schedule",  # 자기 자신으로 루프
            },
        )

        workflow.add_edge("add_walk_schedule", "generate_response")
        workflow.add_edge("handle_schedule_failure", "ask_user_confirmation")

        # 사용자 확인 후 분기
        workflow.add_conditional_edges(
            "ask_user_confirmation",
            self.handle_user_response,
            {
                "retry": "add_umbrella_schedule",
                "skip": "generate_response",
                "change_time": "add_umbrella_schedule",
            },
        )

        workflow.add_edge("generate_response", END)

        return workflow.compile()

    # 각 노드 구현
    def analyze_intent_node(self, state: ConversationState) -> ConversationState:
        """의도 분석 노드"""
        print("🔍 의도 분석 중...")
        state["intent"] = "weather_and_schedule"
        state["city"] = "서울"
        state["messages"].append("의도 분석 완료")
        return state

    def get_weather_node(self, state: ConversationState) -> ConversationState:
        """날씨 조회 노드"""
        print(f"🌤️ {state['city']} 날씨 조회 중...")
        state["weather_data"] = {
            "city": state["city"],
            "condition": "비",
            "temp": 15,
            "humidity": 80,
        }
        state["messages"].append(
            f"날씨 조회 완료: {state['weather_data']['condition']}"
        )
        return state

    def decide_action_node(self, state: ConversationState) -> ConversationState:
        """다음 액션 결정"""
        print("🤔 다음 액션 결정 중...")
        condition = state["weather_data"]["condition"]
        state["next_action"] = "rainy" if condition == "비" else "sunny"
        return state

    def add_umbrella_schedule_node(self, state: ConversationState) -> ConversationState:
        """우산 일정 추가"""
        print("☔ 우산 일정 추가 중...")

        # 시도 횟수 증가
        state["schedule_attempts"] = state.get("schedule_attempts", 0) + 1

        # Mock: 2번째 시도에서 성공
        if state["schedule_attempts"] >= 2:
            state["messages"].append("우산 일정 추가 성공")
            return state
        else:
            state["messages"].append(
                f"우산 일정 추가 실패 (시도 {state['schedule_attempts']}/3)"
            )
            return state

    def add_walk_schedule_node(self, state: ConversationState) -> ConversationState:
        """산책 일정 추가"""
        print("🚶‍♂️ 산책 일정 추가 중...")
        state["messages"].append("산책 일정 추가 완료")
        return state

    def handle_schedule_failure_node(
        self, state: ConversationState
    ) -> ConversationState:
        """일정 추가 실패 처리"""
        print("❌ 일정 추가 실패 처리 중...")
        state["messages"].append("일정 추가 실패 - 사용자 확인 필요")
        return state

    def ask_user_confirmation_node(self, state: ConversationState) -> ConversationState:
        """사용자 확인 요청"""
        print("❓ 사용자에게 확인 요청 중...")
        # 실제로는 사용자 입력을 받아야 함
        state["messages"].append("사용자 확인 완료 - 재시도")
        return state

    def generate_response_node(self, state: ConversationState) -> ConversationState:
        """최종 응답 생성"""
        print("💬 최종 응답 생성 중...")
        weather = state["weather_data"]
        messages_summary = " → ".join(state["messages"])

        state[
            "final_response"
        ] = f"""
        {weather["city"]}의 날씨는 {weather["condition"]}, {weather["temp"]}°C입니다.
        처리 과정: {messages_summary}
        """
        return state

    # 라우팅 함수들
    def route_decision(self, state: ConversationState) -> str:
        """액션 라우팅"""
        return state["next_action"]

    def check_schedule_success(self, state: ConversationState) -> str:
        """일정 추가 성공 여부 확인"""
        if "성공" in state["messages"][-1]:
            return "success"
        elif state["schedule_attempts"] >= 3:
            return "failure"
        else:
            return "retry"

    def handle_user_response(self, state: ConversationState) -> str:
        """사용자 응답 처리"""
        # 실제로는 사용자 입력을 분석해야 함
        return "retry"  # Mock

    def run_workflow(self, user_input: str) -> str:
        """워크플로우 실행"""
        initial_state = ConversationState(
            messages=[],
            user_input=user_input,
            intent="",
            city="",
            weather_data={},
            schedule_attempts=0,
            final_response="",
            next_action="",
        )

        print("🕸️ LangGraph 워크플로우 시작")
        print("=" * 50)

        # 그래프 실행
        result = self.graph.invoke(initial_state)

        print("=" * 50)
        print("✅ 워크플로우 완료")

        return result["final_response"]


# =============================================================================
# 3. 핵심 차이점 비교
# =============================================================================


def compare_approaches():
    """두 접근 방식의 차이점"""

    print("=" * 60)
    print("📊 LangChain vs LangGraph 핵심 차이점")
    print("=" * 60)

    comparisons = [
        ("구조", "선형 체인 (A→B→C)", "그래프 (복잡한 분기와 루프)"),
        ("상태 관리", "체인 간 데이터 전달", "중앙화된 상태 관리"),
        ("조건부 로직", "if/else로 복잡해짐", "조건부 엣지로 깔끔하게"),
        ("에러 처리", "체인 중단", "실패 시 다른 경로로 라우팅"),
        ("재시도", "어려움", "자기 자신으로 루프 가능"),
        ("사용자 개입", "체인 재시작", "중간에 사용자 입력 받기 가능"),
        ("복잡도", "단순한 작업에 적합", "복잡한 워크플로우에 적합"),
        ("디버깅", "순차적", "노드별 상태 추적"),
    ]

    for category, langchain, langgraph in comparisons:
        print(f"\n🎯 {category}:")
        print(f"  LangChain: {langchain}")
        print(f"  LangGraph: {langgraph}")


# =============================================================================
# 4. 실제 사용 예시
# =============================================================================


def demonstrate_differences():
    """실제 동작 차이 보여주기"""

    user_input = "서울 날씨 확인하고 비 오면 우산 챙기라고 일정 추가해줘"

    print("\n" + "=" * 60)
    print("🔥 LangChain 방식 실행")
    print("=" * 60)

    # LangChain 방식
    langchain_example = ComplexLangChainExample()
    langchain_result = langchain_example.complex_task_chain(user_input)
    print(f"결과: {langchain_result}")

    print("\n" + "=" * 60)
    print("🕸️ LangGraph 방식 실행")
    print("=" * 60)

    # LangGraph 방식
    langgraph_example = LangGraphExample()
    langgraph_result = langgraph_example.run_workflow(user_input)
    print(f"결과: {langgraph_result}")


# 실행
if __name__ == "__main__":
    # 차이점 비교
    compare_approaches()

    # 실제 동작 시연
    demonstrate_differences()
