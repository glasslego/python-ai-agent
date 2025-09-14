# flake8: noqa: E501
import os

from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI

from src.simple_assistant.utils.tools import (
    add_schedule,
    get_exchange_rate,
    get_schedule,
    get_stock_price,
    get_weather,
    search_news,
)

# 환경 변수 파일(.env)에서 API 키 등을 로드
load_dotenv()


class SmartAssistant:
    """
    스마트 개인 비서 클래스

    LangChain의 ReAct (Reasoning + Acting) 패턴을 사용하여
    다양한 도구들을 조합해 사용자 질문에 응답하는 AI 에이전트

    주요 기능:
    - 뉴스 검색
    - 날씨 조회
    - 환율 정보
    - 주식 가격
    - 일정 관리
    """

    def __init__(self):
        """
        SmartAssistant 초기화

        1. OpenAI LLM 모델 설정
        2. 사용 가능한 도구들 등록
        3. ReAct 프롬프트 템플릿 설정
        4. 에이전트 및 실행기 생성
        """

        # OpenAI API 키를 환경 변수에서 가져오기
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY 환경변수를 설정해주세요.")

        # OpenAI ChatGPT 모델 초기화
        # - model: gpt-3.5-turbo (비용 효율적)
        # - temperature: 0.1 (일관된 답변을 위해 낮은 값)
        self.llm = ChatOpenAI(
            model="gpt-3.5-turbo", temperature=0.1, openai_api_key=api_key
        )

        # 에이전트가 사용할 수 있는 도구들의 목록
        # 각 도구는 @tool 데코레이터로 정의된 함수들
        self.tools = [
            search_news,  # 뉴스 검색 도구
            get_weather,  # 날씨 조회 도구
            get_exchange_rate,  # 환율 조회 도구
            get_stock_price,  # 주식 가격 조회 도구
            add_schedule,  # 일정 추가 도구
            get_schedule,  # 일정 조회 도구
        ]

        # ReAct 프롬프트 템플릿을 LangChain Hub에서 가져오기
        # ReAct = Reasoning (추론) + Acting (행동) 패턴
        # LLM이 단계별로 생각하고 도구를 사용하는 방식
        try:
            # LangChain Hub에서 표준 ReAct 프롬프트 가져오기
            self.prompt = hub.pull("hwchase17/react")
        except Exception:
            # Hub 접속 실패 시 백업 프롬프트 템플릿 사용
            # ReAct 패턴: Question → Thought → Action → Observation → Final Answer
            template = """
            Answer the following questions as best you can. You have access to the following tools:

            {tools}

            Use the following format:

            Question: the input question you must answer
            Thought: you should always think about what to do
            Action: the action to take, should be one of [{tool_names}]
            Action Input: the input to the action
            Observation: the result of the action
            ... (this Thought/Action/Action Input/Observation can repeat N times)
            Thought: I now know the final answer
            Final Answer: the final answer to the original input question

            Begin!

            Question: {input}
            Thought:{agent_scratchpad}
            """
            self.prompt = PromptTemplate.from_template(template)

        # ReAct 에이전트 생성
        # - llm: 사용할 언어 모델
        # - tools: 에이전트가 사용할 도구들
        # - prompt: ReAct 패턴을 위한 프롬프트 템플릿
        self.agent = create_react_agent(self.llm, self.tools, self.prompt)

        # 에이전트 실행기 생성 (에이전트를 실제로 실행하는 래퍼)
        self.agent_executor = AgentExecutor(
            agent=self.agent,  # 생성된 ReAct 에이전트
            tools=self.tools,  # 도구 목록
            verbose=True,  # 상세 로그 출력 (디버깅용)
            handle_parsing_errors=True,  # 파싱 오류 자동 처리
            max_iterations=3,  # 최대 반복 횟수 (무한루프 방지)
        )

    def chat(self, user_input: str) -> str:
        """
        사용자 입력에 대한 응답 생성

        Args:
            user_input (str): 사용자의 질문이나 요청

        Returns:
            str: AI 에이전트의 응답

        동작 과정:
        1. 사용자 입력을 에이전트에 전달
        2. 에이전트가 ReAct 패턴으로 추론
        3. 필요시 적절한 도구들을 순차적으로 호출
        4. 최종 답변 반환
        """
        try:
            # 에이전트 실행기를 통해 사용자 입력 처리
            # {"input": user_input} 형태로 입력을 전달
            response = self.agent_executor.invoke({"input": user_input})

            # 응답에서 "output" 키의 값(최종 답변)을 반환
            return response["output"]
        except Exception as e:
            # 예외 발생 시 사용자 친화적 오류 메시지 반환
            return f"처리 중 오류가 발생했습니다: {str(e)}"


def main():
    """
    테스트 함수 - OpenAI API 키가 필요합니다

    두 가지 모드로 동작:
    1. API 키가 있을 때: 완전한 AI 에이전트 테스트
    2. API 키가 없을 때: 개별 도구들만 테스트 (더미 데이터)
    """
    print("🧪 SmartAssistant (LangChain ReAct) 테스트")

    # API 키 존재 여부 확인
    if not os.getenv("OPENAI_API_KEY"):
        # API 키가 없는 경우: 사용자에게 안내 후 개별 도구 테스트
        print("❌ OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
        print("   .env 파일에 OPENAI_API_KEY를 설정하거나")
        print("   export OPENAI_API_KEY='your-api-key' 로 설정해주세요.")
        print("\n📝 참고: API 키 없이는 더미 도구만 테스트됩니다.")

        # 도구별 개별 테스트 실행
        # 각 도구의 기본 동작을 확인 (실제 API 호출 없이 더미 데이터 반환)
        print("\n🔧 개별 도구 테스트:")

        # 뉴스 검색 도구 테스트
        print("\n📰 뉴스 검색 테스트:")
        news_result = search_news.invoke({"query": "AI"})
        print(news_result)

        # 날씨 조회 도구 테스트
        print("\n🌤️ 날씨 조회 테스트:")
        weather_result = get_weather.invoke({"city": "Seoul"})
        print(weather_result)

        # 환율 조회 도구 테스트
        print("\n💱 환율 조회 테스트:")
        exchange_result = get_exchange_rate.invoke(
            {"from_currency": "USD", "to_currency": "KRW"}
        )
        print(exchange_result)

        # 주식 가격 조회 도구 테스트
        print("\n📈 주식 조회 테스트:")
        stock_result = get_stock_price.invoke({"symbol": "005930"})
        print(stock_result)

        # 일정 추가 도구 테스트
        print("\n📅 일정 추가 테스트:")
        schedule_add_result = add_schedule.invoke(
            {"date": "2024-03-25", "task": "LangChain 테스트"}
        )
        print(schedule_add_result)

        # 일정 조회 도구 테스트
        print("\n📅 일정 조회 테스트:")
        schedule_get_result = get_schedule.invoke({"date": "2024-03-25"})
        print(schedule_get_result)

        return

    try:
        # API 키가 있는 경우: 완전한 SmartAssistant 테스트
        print("🤖 SmartAssistant 초기화 중...")
        assistant = SmartAssistant()
        print("✅ 초기화 완료!")

        # 다양한 시나리오의 테스트 케이스들
        # 각각 다른 도구들을 사용하는 질문들
        test_cases = [
            "안녕하세요!",  # 일반 대화 (도구 사용 안함)
            "서울 날씨 알려주세요",  # 날씨 도구 사용
            "삼성전자 주가 확인해주세요",  # 주식 도구 사용
            "미국 달러를 한국 원화로 환율 알려주세요",  # 환율 도구 사용
        ]

        print(f"\n🧪 {len(test_cases)}개 테스트 케이스 실행:")
        print("-" * 50)

        # 각 테스트 케이스를 순차적으로 실행
        for i, test_input in enumerate(test_cases, 1):
            print(f"\n[테스트 {i}/{len(test_cases)}]")
            print(f"👤 사용자: {test_input}")
            print("🤖 Assistant:")

            try:
                # 에이전트에게 테스트 입력 전달하고 응답 받기
                # verbose=True로 설정했기 때문에 중간 추론 과정도 출력됨
                response = assistant.chat(test_input)
                print(response)
            except Exception as e:
                # 개별 테스트 케이스에서 오류 발생 시 처리
                print(f"❌ 오류: {e}")

            print("-" * 30)

        print("\n🎉 테스트 완료!")

    except ValueError as e:
        # 설정 관련 오류 (주로 API 키 문제)
        print(f"❌ 설정 오류: {e}")
    except Exception as e:
        # 기타 예상치 못한 오류들
        print(f"❌ 예상치 못한 오류: {e}")


# 스크립트가 직접 실행될 때만 main() 함수 호출
# 모듈로 import될 때는 실행되지 않음
if __name__ == "__main__":
    main()
