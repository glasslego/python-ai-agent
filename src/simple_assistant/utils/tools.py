# flake8: noqa: E501
import json

from langchain.tools import tool

from src.simple_assistant.utils.api_client import (
    ExchangeRateClient,
    NewsAPIClient,
    StockAPIClient,
    WeatherAPIClient,
)

news_client = NewsAPIClient()
weather_client = WeatherAPIClient()
exchange_client = ExchangeRateClient()
stock_client = StockAPIClient()


@tool
def search_news(query: str) -> str:
    """뉴스를 검색합니다. query는 검색어입니다."""
    try:
        result = news_client.search_news(query)
        articles = result.get("articles", [])[:3]  # 최대 3개

        news_summary = f"{query} 관련 뉴스:\n"
        for i, article in enumerate(articles, 1):
            news_summary += f"{i}. {article['title']}\n   {article['description']}\n\n"

        return news_summary
    except Exception as e:
        return f"뉴스 검색 중 오류 발생: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """현재 날씨 정보를 조회하고 옷차림을 추천합니다. city는 도시명입니다."""
    try:
        result = weather_client.get_current_weather(city)
        weather_desc = result["weather"][0]["description"]
        temp = result["main"]["temp"]
        humidity = result["main"]["humidity"]

        # 온도 기반 옷차림 추천
        clothing_advice = ""
        if temp >= 28:
            clothing_advice = "반팔, 반바지, 샌들 추천"
        elif temp >= 23:
            clothing_advice = "가벼운 긴팔, 면바지 추천"
        elif temp >= 17:
            clothing_advice = "가디건, 얇은 재킷 추천"
        elif temp >= 12:
            clothing_advice = "자켓, 니트 추천"
        elif temp >= 5:
            clothing_advice = "코트, 목도리 추천"
        else:
            clothing_advice = "패딩, 장갑, 모자 필수"

        return f"{city} 현재 날씨: {weather_desc}, 온도: {temp}°C, 습도: {humidity}%\n👔 옷차림 추천: {clothing_advice}"
    except Exception as e:
        return f"날씨 조회 중 오류 발생: {str(e)}"


@tool
def get_exchange_rate(from_currency: str, to_currency: str) -> str:
    """환율 정보를 조회합니다. from_currency는 기준통화, to_currency는 대상통화입니다."""
    try:
        result = exchange_client.get_exchange_rate(from_currency, to_currency)
        rate = result["rates"][to_currency]

        return f"{from_currency} → {to_currency}: {rate}"
    except Exception as e:
        return f"환율 조회 중 오류 발생: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """주식 가격을 조회합니다. symbol은 종목코드입니다."""
    try:
        result = stock_client.get_stock_price(symbol)
        name = result["name"]
        price = result["price"]

        return (
            f"{name}({symbol}): {price}원"
            if isinstance(price, int)
            else f"{name}({symbol}): ${price}"
        )
    except Exception as e:
        return f"주식 조회 중 오류 발생: {str(e)}"


@tool
def add_schedule(date: str, task: str) -> str:
    """일정을 추가합니다."""
    # 간단한 파일 저장
    schedule_file = "../tmp/schedules.json"

    try:
        try:
            with open(schedule_file, "r", encoding="utf-8") as f:
                schedules = json.load(f)
        except FileNotFoundError:
            schedules = {}

        if date not in schedules:
            schedules[date] = []
        schedules[date].append(task)

        with open(schedule_file, "w", encoding="utf-8") as f:
            json.dump(schedules, f, ensure_ascii=False, indent=2)

        return f"{date}에 '{task}' 일정이 추가되었습니다."
    except Exception as e:
        return f"일정 추가 중 오류 발생: {str(e)}"


@tool
def get_schedule(date: str) -> str:
    """특정 날짜의 일정을 조회합니다."""
    schedule_file = "../tmp/schedules.json"

    try:
        with open(schedule_file, "r", encoding="utf-8") as f:
            schedules = json.load(f)

        if date in schedules:
            tasks = schedules[date]
            return f"{date} 일정:\n" + "\n".join([f"- {task}" for task in tasks])
        else:
            return f"{date}에 등록된 일정이 없습니다."
    except FileNotFoundError:
        return f"{date}에 등록된 일정이 없습니다."
    except Exception as e:
        return f"일정 조회 중 오류 발생: {str(e)}"
