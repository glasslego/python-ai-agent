"""
LangGraph 노드 정의 모듈

이 모듈은 스마트 어시스턴트의 LangGraph에서 사용되는 모든 노드들을 정의합니다.
각 노드는 특정 기능을 수행하고 상태를 업데이트하여 다음 노드로 전달합니다.

주요 노드들:
- analyze_intent_node: 사용자 입력 의도 분석
- tool_selection_node: 의도에 따른 도구 선택
- tool_execution_node: 선택된 도구 실행
- llm_response_node: LLM을 통한 직접 응답
- response_generation_node: 최종 응답 생성
"""

import re
from datetime import date
from typing import Dict, List

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from src.simple_assistant2.graph.state import AssistantState
from src.simple_assistant2.utils.tools import (
    add_schedule,
    get_exchange_rate,
    get_schedule,
    get_stock_price,
    get_weather,
    search_news,
)


class GraphNodes:
    """
    LangGraph 노드들을 정의하고 관리하는 클래스

    이 클래스는 스마트 어시스턴트의 모든 처리 단계를 노드로 구현합니다.
    각 노드는 AssistantState를 받아서 처리하고 업데이트된 상태를 반환합니다.

    Attributes:
        llm (ChatOpenAI): OpenAI GPT 모델 인스턴스
        tools (dict): 사용 가능한 도구들의 매핑
        intent_patterns (dict): 의도 분석을 위한 정규식 패턴들

    주요 워크플로우:
        사용자 입력 → 의도 분석 → 도구 선택 → 도구 실행 → 응답 생성
    """

    def __init__(self, llm: ChatOpenAI):
        """
        GraphNodes 초기화

        Args:
            llm (ChatOpenAI): 언어모델 인스턴스. 일반 대화 처리에 사용됨
        """
        # OpenAI GPT 모델 - 일반 대화 및 복잡한 추론에 사용
        self.llm = llm

        # 사용 가능한 도구들을 딕셔너리로 매핑
        # 각 도구는 @tool 데코레이터로 정의된 LangChain 도구
        self.tools = {
            "search_news": search_news,  # 뉴스 검색 도구
            "get_weather": get_weather,  # 날씨 조회 도구
            "get_exchange_rate": get_exchange_rate,  # 환율 조회 도구
            "get_stock_price": get_stock_price,  # 주식 가격 조회 도구
            "add_schedule": add_schedule,  # 일정 추가 도구
            "get_schedule": get_schedule,  # 일정 조회 도구
        }

        # 의도 분석을 위한 키워드 패턴 정의
        # 정규식을 사용하여 사용자 입력에서 의도를 추출
        self.intent_patterns = {
            # 뉴스 관련 키워드들
            "news": [r"뉴스", r"news", r"기사", r"소식", r"헤드라인"],
            # 날씨 관련 키워드들
            "weather": [
                r"날씨",
                r"weather",
                r"기온",
                r"온도",
                r"비",
                r"눈",
                r"바람",
                r"옷차림",
            ],
            # 환율 관련 키워드들
            "exchange": [
                r"환율",
                r"exchange",
                r"달러",
                r"원",
                r"엔",
                r"유로",
                r"USD",
                r"KRW",
            ],
            # 주식 관련 키워드들
            "stock": [
                r"주식",
                r"주가",
                r"stock",
                r"종목",
                r"삼성전자",
                r"애플",
                r"AAPL",
            ],
            # 일정 관련 키워드들
            "schedule": [
                r"일정",
                r"schedule",
                r"약속",
                r"미팅",
                r"회의",
                r"할일",
                r"todo",
            ],
        }

    def analyze_intent_node(self, state: AssistantState) -> AssistantState:
        """
        사용자 입력의 의도를 분석하는 노드

        이 노드는 LangGraph의 첫 번째 처리 단계로,
        사용자의 입력을 분석하여 어떤 종류의 요청인지 파악합니다.

        처리 과정:
        1. 입력 텍스트를 소문자로 변환
        2. 사전 정의된 패턴과 매칭하여 의도 탐지
        3. 관련 엔티티(도시명, 날짜, 종목 등) 추출
        4. 상태에 의도와 엔티티 정보 저장

        Args:
            state (AssistantState): 현재 그래프 상태

        Returns:
            AssistantState: 의도와 엔티티가 추가된 업데이트된 상태
        """
        # 사용자 입력을 소문자로 변환하여 패턴 매칭 준비
        user_input = state["user_input"].lower()
        detected_intents = []

        # 각 의도별 패턴을 순차적으로 확인
        for intent, patterns in self.intent_patterns.items():
            # 각 패턴에 대해 정규식 매칭 수행
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    detected_intents.append(intent)
                    break  # 하나의 패턴이 매칭되면 해당 의도로 간주

        # 주 의도 결정: 첫 번째로 발견된 의도를 사용
        # 감지된 의도가 없으면 "general"(일반 대화)로 분류
        main_intent = detected_intents[0] if detected_intents else "general"

        # 의도 실행에 필요한 구체적인 정보(엔티티) 추출
        # 예: 도시명, 날짜, 종목코드, 통화 등
        entities = self._extract_entities(user_input, detected_intents)

        # 상태 업데이트: 분석 결과를 다음 노드에서 사용할 수 있도록 저장
        state["intent"] = main_intent  # 주 의도
        state["entities"] = entities  # 추출된 엔티티들
        state["next_action"] = "tool_selection"  # 다음 단계 지정

        return state

    def tool_selection_node(self, state: AssistantState) -> AssistantState:
        """
        의도에 따라 사용할 도구를 선택하는 노드

        분석된 의도를 바탕으로 다음 처리 방식을 결정합니다.
        - 일반 대화: LLM으로 직접 응답
        - 특정 기능: 해당 도구 실행

        이 노드는 그래프의 라우팅 역할을 담당하며,
        조건부 엣지를 통해 적절한 다음 노드로 분기시킵니다.

        Args:
            state (AssistantState): 의도 분석이 완료된 상태

        Returns:
            AssistantState: next_action이 설정된 상태
        """
        # 현재 상태에서 필요한 정보 추출
        intent = state["intent"]  # 분석된 주 의도
        entities = state["entities"]  # 추출된 엔티티들
        print(f"entities: {entities}")
        user_input = state["user_input"]  # 원본 사용자 입력
        print(f"user_input: {user_input}")

        # 의도에 따른 처리 방식 결정
        if intent == "general":
            # 일반 대화인 경우: 도구 없이 LLM으로 직접 응답
            # 예: "안녕하세요", "감사합니다" 등의 일상적 대화
            state["next_action"] = "llm_response"
        elif intent in ["news", "weather", "exchange", "stock", "schedule"]:
            # 특정 기능이 필요한 경우: 해당 도구 실행
            # API 호출이나 데이터 처리가 필요한 요청들
            state["next_action"] = "tool_execution"
        else:
            # 예상치 못한 의도인 경우: 안전하게 LLM 응답으로 처리
            state["next_action"] = "llm_response"

        return state

    def tool_execution_node(self, state: AssistantState) -> AssistantState:
        """
        선택된 도구를 실행하는 노드

        의도에 따라 적절한 도구를 실행하고 결과를 수집합니다.
        각 도구는 외부 API나 데이터 소스에 접근하여 정보를 가져옵니다.

        지원하는 도구들:
        - news: 뉴스 검색 (NewsAPI 등)
        - weather: 날씨 조회 (OpenWeatherMap 등)
        - exchange: 환율 조회 (ExchangeRate API 등)
        - stock: 주식 가격 조회 (Yahoo Finance 등)
        - schedule: 일정 관리 (로컬 JSON 파일)

        Args:
            state (AssistantState): 도구 실행이 필요한 상태

        Returns:
            AssistantState: 도구 실행 결과가 포함된 상태
        """
        # 상태에서 도구 실행에 필요한 정보 추출
        intent = state["intent"]  # 실행할 도구를 결정하는 의도
        entities = state["entities"]  # 도구 실행에 필요한 매개변수들
        user_input = state["user_input"]  # 추가 분석이 필요한 원본 입력

        try:
            # 의도별 도구 실행 및 결과 처리
            if intent == "news":
                # 뉴스 검색: 입력에서 검색 키워드 추출 후 뉴스 API 호출
                query = self._extract_news_query(user_input)
                result = search_news.invoke({"query": query})

            elif intent == "weather":
                # 날씨 조회: 도시명이 있으면 사용, 없으면 기본값(Seoul) 사용
                city = entities.get("city", "Seoul")
                result = get_weather.invoke({"city": city})

            elif intent == "exchange":
                # 환율 조회: 통화 쌍 설정, 기본값은 USD->KRW
                currencies = entities.get("currencies", ["USD", "KRW"])
                from_curr = currencies[0] if len(currencies) > 0 else "USD"
                to_curr = currencies[1] if len(currencies) > 1 else "KRW"
                result = get_exchange_rate.invoke(
                    {"from_currency": from_curr, "to_currency": to_curr}
                )

            elif intent == "stock":
                # 주식 조회: 종목코드 사용, 기본값은 삼성전자(005930)
                symbol = entities.get("stock_symbol", "005930")
                result = get_stock_price.invoke({"symbol": symbol})

            elif intent == "schedule":
                # 일정 관리: 추가/조회 여부를 입력 텍스트로 판단
                if "추가" in user_input or "등록" in user_input:
                    # 일정 추가 모드
                    task_date = entities.get("date", str(date.today()))
                    task_content = self._extract_task_content(user_input)
                    result = add_schedule.invoke(
                        {"date": task_date, "task": task_content}
                    )
                else:
                    # 일정 조회 모드
                    query_date = entities.get("date", str(date.today()))
                    result = get_schedule.invoke({"date": query_date})
            else:
                # 예상치 못한 의도인 경우
                result = "요청을 처리할 수 없습니다."

            # 성공적인 도구 실행 결과 저장
            state["tool_results"] = [result]
            state["next_action"] = "response_generation"

        except Exception as e:
            # 도구 실행 중 오류 발생 시 오류 메시지 저장
            # 네트워크 오류, API 키 문제, 잘못된 매개변수 등
            state["tool_results"] = [f"도구 실행 중 오류 발생: {str(e)}"]
            state["next_action"] = "response_generation"

        return state

    def llm_response_node(self, state: AssistantState) -> AssistantState:
        """
        LLM을 사용해 직접 응답을 생성하는 노드

        일반적인 대화나 복잡한 추론이 필요한 경우에 사용됩니다.
        도구 없이도 처리 가능한 요청들(인사, 감사, 일반 질문 등)을 처리합니다.

        처리 과정:
        1. 시스템 메시지로 AI의 역할과 톤 설정
        2. 사용자 입력을 휴먼 메시지로 전달
        3. OpenAI GPT 모델을 통해 응답 생성
        4. 최종 응답으로 설정하고 그래프 종료

        Args:
            state (AssistantState): 사용자 입력이 포함된 상태

        Returns:
            AssistantState: 최종 응답이 설정된 상태
        """
        try:
            # LangChain 메시지 형식으로 대화 구성
            messages = [
                # 시스템 메시지: AI의 성격과 응답 방식 지정
                SystemMessage(
                    content="당신은 도움이 되는 AI 어시스턴트입니다. 친근하고 정중하게 응답하세요."
                ),
                # 휴먼 메시지: 사용자의 실제 입력
                HumanMessage(content=state["user_input"]),
            ]

            # OpenAI GPT 모델에 메시지 전달하여 응답 생성
            # temperature=0.1로 설정되어 있어 일관되고 예측 가능한 응답
            response = self.llm.invoke(messages)

            # 생성된 응답을 최종 응답으로 설정
            state["final_response"] = response.content
            # 그래프 실행 종료 지시
            state["next_action"] = "end"

        except Exception as e:
            # LLM 호출 실패 시 (API 키 문제, 네트워크 오류 등)
            state["final_response"] = f"응답 생성 중 오류 발생: {str(e)}"
            state["next_action"] = "end"

        return state

    def response_generation_node(self, state: AssistantState) -> AssistantState:
        """
        도구 실행 결과를 바탕으로 최종 응답을 생성하는 노드

        tool_execution_node에서 수집된 결과들을 사용자에게 전달할
        형태로 가공하여 최종 응답을 생성합니다.

        현재는 단순히 도구 결과를 연결하여 반환하지만,
        향후 LLM을 사용하여 더 자연스러운 응답으로 가공할 수 있습니다.

        Args:
            state (AssistantState): 도구 실행 결과가 포함된 상태

        Returns:
            AssistantState: 최종 응답이 설정된 상태
        """
        # 도구 실행 결과들 가져오기
        # get() 메서드 사용으로 키가 없을 때 안전하게 빈 리스트 반환
        tool_results = state.get("tool_results", [])

        if tool_results:
            # 도구 실행 결과들이 있는 경우
            # 여러 결과를 개행으로 연결하여 하나의 응답으로 구성
            # 예: 뉴스 결과 + 날씨 결과가 함께 요청된 경우
            state["final_response"] = "\n\n".join(tool_results)
        else:
            # 도구 실행 결과가 없는 경우 (오류 또는 예상치 못한 상황)
            state["final_response"] = "죄송합니다. 요청을 처리할 수 없습니다."

        # 그래프 실행 종료 지시
        state["next_action"] = "end"
        return state

    def _extract_entities(self, user_input: str, intents: List[str]) -> Dict:
        """
        사용자 입력에서 구조화된 엔티티 정보를 추출하는 헬퍼 메서드

        각 의도별로 필요한 정보들(도시명, 날짜, 종목코드, 통화 등)을
        사용자 입력에서 찾아서 구조화된 형태로 반환합니다.

        추출 가능한 엔티티들:
        - city: 날씨 조회에 필요한 도시명
        - date: 일정 관리에 필요한 날짜 정보
        - stock_symbol: 주식 조회에 필요한 종목코드
        - currencies: 환율 조회에 필요한 통화 코드들

        Args:
            user_input (str): 사용자의 원본 입력
            intents (List[str]): 감지된 의도들 (현재는 사용하지 않음)

        Returns:
            Dict: 추출된 엔티티들의 딕셔너리
        """
        entities = {}

        # 도시명 추출 - 날씨 조회에 사용
        # 지원되는 도시들의 리스트 (한글명과 영문명 모두 지원)
        cities = ["서울", "부산", "대구", "seoul", "busan", "tokyo", "new york"]
        for city in cities:
            if city in user_input.lower():
                # 첫 글자를 대문자로 변환하여 표준화
                entities["city"] = city.title()
                break  # 첫 번째로 발견된 도시만 사용

        # 날짜 추출 - 일정 관리에 사용
        # 지원하는 날짜 형식들의 정규식 패턴
        date_patterns = [
            r"오늘",
            r"내일",
            r"모레",
            r"\d{4}-\d{2}-\d{2}",
            r"\d{1,2}월 \d{1,2}일",
        ]
        for pattern in date_patterns:
            match = re.search(pattern, user_input)
            if match:
                date_text = match.group()
                # 상대적 날짜 표현을 절대 날짜로 변환
                if date_text == "오늘":
                    entities["date"] = str(date.today())
                elif date_text == "내일":
                    from datetime import timedelta

                    entities["date"] = str(date.today() + timedelta(days=1))
                else:
                    # 절대 날짜나 다른 형식은 그대로 사용
                    entities["date"] = date_text
                break  # 첫 번째로 발견된 날짜만 사용

        # 종목코드/회사명 추출 - 주식 조회에 사용
        # 회사명을 종목코드로 매핑하는 딕셔너리
        stock_mapping = {
            "삼성전자": "005930",  # 한국 코스피
            "SK하이닉스": "000660",  # 한국 코스피
            "애플": "AAPL",  # 미국 나스닥
            "마이크로소프트": "MSFT",  # 미국 나스닥
        }

        # 회사명이 입력에 포함되어 있는지 확인
        for company, symbol in stock_mapping.items():
            if company in user_input:
                entities["stock_symbol"] = symbol
                break

        # 직접 종목코드가 입력된 경우 처리
        stock_symbols = ["005930", "000660", "AAPL", "MSFT", "GOOGL"]
        for symbol in stock_symbols:
            if symbol in user_input:
                entities["stock_symbol"] = symbol
                break

        # 통화 추출 - 환율 조회에 사용
        # 한글 통화명을 ISO 코드로 매핑
        currency_mapping = {"달러": "USD", "원": "KRW", "엔": "JPY", "유로": "EUR"}
        # 검색 대상이 되는 모든 통화들 (ISO 코드 + 한글명)
        currencies = ["USD", "KRW", "EUR", "JPY"] + list(currency_mapping.keys())
        found_currencies = []

        # 입력에서 언급된 모든 통화 수집
        for currency in currencies:
            if currency in user_input.upper():
                if currency in currency_mapping:
                    # 한글명인 경우 ISO 코드로 변환
                    found_currencies.append(currency_mapping[currency])
                else:
                    # 이미 ISO 코드인 경우 그대로 사용
                    found_currencies.append(currency)

        # 발견된 통화가 있으면 엔티티에 추가
        if found_currencies:
            entities["currencies"] = found_currencies

        return entities

    def _extract_news_query(self, user_input: str) -> str:
        """
        뉴스 검색을 위한 키워드를 추출하는 헬퍼 메서드

        사용자 입력에서 불필요한 동사나 조사를 제거하고
        실제 검색에 사용될 키워드만 추출합니다.

        예시:
        - "삼성전자 뉴스 찾아줘" → "삼성전자"
        - "AI 관련 뉴스 알려줘" → "AI"
        - "뉴스 검색해줘" → "최신"

        Args:
            user_input (str): 사용자의 원본 입력

        Returns:
            str: 정제된 검색 키워드
        """
        search_words = []
        # 뉴스 검색에서 제외할 불필요한 단어들
        # 동사, 조사, 일반적인 요청 표현들
        exclude_words = ["뉴스", "찾아줘", "검색", "알려줘", "어때", "관련"]

        # 입력을 단어별로 분리하고 불필요한 단어 제거
        for word in user_input.split():
            if word not in exclude_words:
                search_words.append(word)

        # 추출된 단어들을 공백으로 연결
        # 키워드가 없으면 "최신" 뉴스로 검색
        return " ".join(search_words) if search_words else "최신"

    def _extract_task_content(self, user_input: str) -> str:
        """
        일정 추가를 위한 작업 내용을 추출하는 헬퍼 메서드

        사용자 입력에서 "일정", "추가" 등의 동작 관련 단어를 제거하고
        실제 일정 내용만 추출합니다.

        예시:
        - "내일 회의 일정 추가해줘" → "내일 회의"
        - "프로젝트 리뷰 등록해줘" → "프로젝트 리뷰"
        - "일정 추가해줘" → "새로운 일정"

        Args:
            user_input (str): 사용자의 원본 입력

        Returns:
            str: 정제된 일정 내용
        """
        # 일정 관리에서 제외할 동작 관련 단어들
        exclude_words = ["일정", "추가", "등록", "해줘", "알려줘"]
        words = user_input.split()

        # 불필요한 단어들을 제거하여 실제 일정 내용만 추출
        task_words = [word for word in words if word not in exclude_words]

        # 추출된 내용이 있으면 연결하고, 없으면 기본값 사용
        return " ".join(task_words) if task_words else "새로운 일정"
