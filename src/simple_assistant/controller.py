# flake8: noqa: E501
import re
from datetime import date

from src.simple_assistant.agents.finance_agent import FinanceAgent
from src.simple_assistant.agents.news_agent import NewsAgent
from src.simple_assistant.agents.schedule_agent import ScheduleAgent
from src.simple_assistant.agents.weather_agent import WeatherAgent


class SmartAssistantController:
    """모든 에이전트를 통합 관리하는 컨트롤러"""

    def __init__(self):
        self.news_agent = NewsAgent()
        self.weather_agent = WeatherAgent()
        self.finance_agent = FinanceAgent()
        self.schedule_agent = ScheduleAgent()

        # 의도 분석을 위한 키워드 패턴
        self.intent_patterns = {
            "news": [r"뉴스", r"news", r"기사", r"소식", r"헤드라인"],
            "weather": [
                r"날씨",
                r"weather",
                r"기온",
                r"온도",
                r"비",
                r"눈",
                r"바람",
                r"옷차림",
                r"입을거",
                r"입을옷",
            ],
            "exchange": [
                r"환율",
                r"exchange",
                r"달러",
                r"원",
                r"엔",
                r"유로",
                r"USD",
                r"KRW",
                r"JPY",
                r"EUR",
            ],
            "stock": [
                r"주식",
                r"주가",
                r"stock",
                r"종목",
                r"삼성전자",
                r"애플",
                r"AAPL",
                r"MSFT",
            ],
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

    def analyze_intent(self, user_input: str):
        """사용자 입력에서 의도를 분석"""
        user_input = user_input.lower()
        detected_intents = []

        for intent, patterns in self.intent_patterns.items():
            for pattern in patterns:
                if re.search(pattern, user_input, re.IGNORECASE):
                    detected_intents.append(intent)
                    break

        return list(set(detected_intents))  # 중복 제거

    def extract_entities(self, user_input: str, intents: list):
        """입력에서 엔티티 추출"""
        entities = {}

        # 도시명 추출
        cities = ["서울", "부산", "대구", "seoul", "busan", "tokyo", "new york"]
        for city in cities:
            if city in user_input.lower():
                entities["city"] = city.title()
                break

        # 날짜 추출
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
                if date_text == "오늘":
                    entities["date"] = str(date.today())
                elif date_text == "내일":
                    entities["date"] = str(
                        date.today().replace(day=date.today().day + 1)
                    )
                else:
                    entities["date"] = date_text
                break

        # 종목코드/회사명 추출
        stock_symbols = ["005930", "000660", "AAPL", "MSFT", "GOOGL"]
        companies = ["삼성전자", "애플", "마이크로소프트", "SK하이닉스"]

        for symbol in stock_symbols:
            if symbol in user_input:
                entities["stock_symbol"] = symbol
                break

        for company in companies:
            if company in user_input:
                if company == "삼성전자":
                    entities["stock_symbol"] = "005930"
                elif company == "SK하이닉스":
                    entities["stock_symbol"] = "000660"
                elif company == "애플":
                    entities["stock_symbol"] = "AAPL"
                elif company == "마이크로소프트":
                    entities["stock_symbol"] = "MSFT"
                break

        # 통화 추출
        currencies = ["USD", "KRW", "EUR", "JPY", "달러", "원", "엔", "유로"]
        found_currencies = []
        for currency in currencies:
            if currency in user_input.upper():
                if currency == "달러":
                    found_currencies.append("USD")
                elif currency == "원":
                    found_currencies.append("KRW")
                elif currency == "엔":
                    found_currencies.append("JPY")
                elif currency == "유로":
                    found_currencies.append("EUR")
                else:
                    found_currencies.append(currency)

        if found_currencies:
            entities["currencies"] = found_currencies

        return entities

    def process_single_intent(self, intent: str, entities: dict, user_input: str):
        """단일 의도 처리"""
        try:
            if intent == "news":
                # 뉴스 검색어 추출
                search_words = []
                for word in user_input.split():
                    if word not in ["뉴스", "찾아줘", "검색", "알려줘"]:
                        search_words.append(word)

                query = " ".join(search_words) if search_words else "최신"
                result = self.news_agent.search_news(query)

                if result["status"] == "success":
                    response = f"'{query}' 관련 뉴스 {result['total_results']}건을 찾았습니다:\n\n"
                    for i, article in enumerate(result["articles"], 1):
                        response += (
                            f"{i}. {article['title']}\n   {article['description']}\n\n"
                        )
                    return response
                else:
                    return f"뉴스 검색 중 오류가 발생했습니다: {result['message']}"

            elif intent == "weather":
                city = entities.get("city", "Seoul")
                result = self.weather_agent.get_current_weather(city)

                if result["status"] == "success":
                    weather = result["weather"]
                    clothing = result["clothing_recommendation"]

                    response = f"{result['city']} 현재 날씨:\n"
                    response += f"🌡️ 온도: {weather['temperature']}°C\n"
                    response += f"🌤️ 날씨: {weather['description']}\n"
                    response += f"💧 습도: {weather['humidity']}%\n\n"
                    response += f"👔 옷차림 추천 ({clothing['category']}):\n"
                    response += f"• {', '.join(clothing['items'])}\n"
                    response += f"💡 {clothing['advice']}"

                    return response
                else:
                    return f"날씨 조회 중 오류가 발생했습니다: {result['message']}"

            elif intent == "exchange":
                currencies = entities.get("currencies", ["USD", "KRW"])
                from_currency = currencies[0] if len(currencies) > 0 else "USD"
                to_currency = currencies[1] if len(currencies) > 1 else "KRW"

                result = self.finance_agent.get_exchange_rate(
                    from_currency, to_currency
                )

                if result["status"] == "success":
                    return f"💱 {from_currency} → {to_currency}: {result['rate']}"
                else:
                    return f"환율 조회 중 오류가 발생했습니다: {result['message']}"

            elif intent == "stock":
                symbol = entities.get("stock_symbol", "005930")  # 기본값: 삼성전자
                result = self.finance_agent.get_stock_price(symbol)

                if result["status"] == "success":
                    return f"📈 {result['name']} ({result['symbol']}): {result['price']} {result['currency']}"
                else:
                    return f"주식 조회 중 오류가 발생했습니다: {result['message']}"

            elif intent == "schedule":
                if "추가" in user_input or "등록" in user_input:
                    # 일정 추가 안내
                    return "일정을 추가하려면 다음 형식으로 입력해주세요:\n'[날짜] [제목] 일정 추가'\n예: '내일 회의 일정 추가'"
                else:
                    # 오늘 일정 조회
                    today = str(date.today())
                    result = self.schedule_agent.get_schedule(today)

                    if result["status"] == "success" and result["schedules"]:
                        response = f"📅 오늘({today}) 일정:\n"
                        for schedule in result["schedules"]:
                            priority_icon = {"high": "🔴", "medium": "🟡", "low": "🟢"}
                            time_str = (
                                f" ({schedule['time']})" if schedule["time"] else ""
                            )
                            response += f"{priority_icon[schedule['priority']]} {schedule['title']}{time_str}\n"
                        return response
                    else:
                        return "오늘 등록된 일정이 없습니다."

        except Exception as e:
            return f"처리 중 오류가 발생했습니다: {str(e)}"

        return "죄송합니다. 요청을 이해하지 못했습니다."

    def process_multiple_intents(self, intents: list, entities: dict, user_input: str):
        """복합 의도 처리"""
        responses = []

        for intent in intents:
            response = self.process_single_intent(intent, entities, user_input)
            responses.append(response)

        return "\n\n" + "=" * 50 + "\n\n".join(responses)

    def chat(self, user_input: str):
        """메인 채팅 인터페이스"""
        if not user_input.strip():
            return "무엇을 도와드릴까요?"

        # 의도 분석
        intents = self.analyze_intent(user_input)

        if not intents:
            return "죄송합니다. 요청을 이해하지 못했습니다. 뉴스, 날씨, 환율, 주식, 일정 관련 질문을 해주세요."

        # 엔티티 추출
        entities = self.extract_entities(user_input, intents)

        # 단일 또는 복합 의도 처리
        if len(intents) == 1:
            return self.process_single_intent(intents[0], entities, user_input)
        else:
            return self.process_multiple_intents(intents, entities, user_input)

    def get_dashboard_info(self):
        """대시보드용 정보 수집"""
        dashboard_data = {}

        try:
            # 오늘 일정
            today = str(date.today())
            schedule_result = self.schedule_agent.get_schedule(today)
            dashboard_data["today_schedules"] = schedule_result.get("schedules", [])

            # 현재 날씨 (서울)
            weather_result = self.weather_agent.get_current_weather("Seoul")
            dashboard_data["weather"] = (
                weather_result if weather_result["status"] == "success" else None
            )

            # 주요 환율
            exchange_result = self.finance_agent.get_multiple_exchange_rates("USD")
            dashboard_data["exchange_rates"] = (
                exchange_result.get("rates", {})
                if exchange_result["status"] == "success"
                else {}
            )

            # 관심 종목
            watchlist_result = self.finance_agent.get_watchlist()
            dashboard_data["watchlist"] = (
                watchlist_result.get("watchlist", {})
                if watchlist_result["status"] == "success"
                else {}
            )

            # 최신 헤드라인
            news_result = self.news_agent.get_top_headlines(limit=5)
            dashboard_data["headlines"] = (
                news_result.get("headlines", [])
                if news_result["status"] == "success"
                else []
            )

        except Exception as e:
            dashboard_data["error"] = str(e)

        return dashboard_data
