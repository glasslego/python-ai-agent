import os
from datetime import datetime

from dotenv import load_dotenv
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain.prompts import ChatPromptTemplate
from langchain.tools import tool
from langchain_openai import ChatOpenAI

# 환경 변수 로드
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


# =============================================================================
# 2. LangChain Tool Call 방식 (더 간단하고 추상화됨)
# =============================================================================


class LangChainToolCallExample:
    def __init__(self, api_key: str):
        self.llm = ChatOpenAI(model="gpt-3.5-turbo", openai_api_key=api_key)
        self.tools = self._create_tools()
        self.agent_executor = self._create_agent()

    def _create_tools(self):
        """LangChain @tool 데코레이터를 사용한 도구 정의"""

        @tool
        def get_current_time_tool() -> str:
            """현재 시각을 조회합니다."""
            return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        @tool
        def get_weather_tool(city: str) -> str:
            """특정 도시의 날씨를 조회합니다.

            Args:
                city: 날씨를 조회할 도시명
            """
            weather_data = {
                "서울": "맑음, 22°C",
                "부산": "흐림, 18°C",
                "대구": "비, 15°C",
            }
            return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")

        @tool
        def calculate_tool(expression: str) -> str:
            """수학 계산을 수행합니다.

            Args:
                expression: 계산할 수식 (예: 2+3, 10*5)
            """
            try:
                result = eval(expression)
                return f"{expression} = {result}"
            except Exception:
                return "계산할 수 없는 식입니다"

        return [get_current_time_tool, get_weather_tool, calculate_tool]

    def _create_agent(self):
        """LangChain 에이전트 생성"""
        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "당신은 도움이 되는 AI 어시스턴트입니다. 필요시 제공된 도구들을 사용하세요.",
                ),
                ("user", "{input}"),
                ("assistant", "{agent_scratchpad}"),
            ]
        )

        agent = create_openai_functions_agent(self.llm, self.tools, prompt)
        return AgentExecutor(agent=agent, tools=self.tools, verbose=True)

    def chat_with_tools(self, user_message: str) -> str:
        """LangChain Tool Call을 사용한 채팅"""
        print(f"👤 사용자: {user_message}")
        print("🔧 LangChain Tool Call 실행:")

        try:
            result = self.agent_executor.invoke({"input": user_message})
            answer = result["output"]
            print(f"🤖 최종 답변: {answer}")
            return answer
        except Exception as e:
            error_msg = f"오류가 발생했습니다: {str(e)}"
            print(f"❌ {error_msg}")
            return error_msg


if __name__ == "__main__":
    test_questions = [
        "지금 몇 시야?",
        "서울 날씨 어때?",
        "10 * 25는 얼마야?",
        "안녕하세요!",  # 함수/도구 호출 불필요
        "현재 시간이랑 서울 날씨 둘 다 알려줘",  # 멀티 호출
    ]

    langchain_client = LangChainToolCallExample(api_key)
    for question in test_questions:
        print(f"\n📝 테스트: {question}")
        langchain_client.chat_with_tools(question)
