# flake8: noqa E501
"""
LangChain 도구(Tool) 정의 모듈

이 모듈은 LangGraph에서 사용할 수 있는 모든 도구들을 정의합니다.
각 도구는 @tool 데코레이터를 사용하여 LangChain과 호환되는 형태로 구현됩니다.

지원하는 도구들:
- search_news: 뉴스 검색 및 요약
- get_weather: 날씨 조회 및 옷차림 추천
- get_exchange_rate: 실시간 환율 정보 조회
- get_stock_price: 주식 가격 조회
- add_schedule: 일정 추가 (로컬 JSON 파일)
- get_schedule: 일정 조회 (로컬 JSON 파일)

각 도구는 외부 API나 로컬 데이터 소스에 접근하여 정보를 가져오고,
사용자 친화적인 형태로 가공하여 반환합니다.
"""

import json
import os
from datetime import datetime

from langchain_core.tools import tool

from src.simple_assistant.utils.api_client import (
    ExchangeRateClient,
    NewsAPIClient,
    StockAPIClient,
    WeatherAPIClient,
)

# 각종 외부 API 클라이언트 인스턴스 생성
# 이들은 실제 외부 서비스(NewsAPI, OpenWeatherMap 등)에 접근하는 클래스들
news_client = NewsAPIClient()  # 뉴스 검색을 위한 클라이언트
weather_client = WeatherAPIClient()  # 날씨 정보를 위한 클라이언트
exchange_client = ExchangeRateClient()  # 환율 정보를 위한 클라이언트
stock_client = StockAPIClient()  # 주식 정보를 위한 클라이언트


@tool
def search_news(query: str) -> str:
    """
    뉴스 검색 도구

    지정된 검색어로 최신 뉴스를 검색하고 요약하여 반환합니다.
    NewsAPI나 기타 뉴스 소스를 활용하여 관련 기사를 찾아
    제목과 요약을 포함한 읽기 쉬운 형태로 가공합니다.

    Args:
        query (str): 검색할 키워드 (예: "삼성전자", "AI", "경제")

    Returns:
        str: 검색된 뉴스 기사들의 요약 정보
            - 최대 3건의 기사 제공
            - 각 기사의 제목과 요약 포함
            - 검색 결과가 없을 경우 안내 메시지

    Example:
        >>> search_news.invoke({"query": "삼성전자"})
        "삼성전자 관련 뉴스 3건:
        1. 삼성전자 3분기 실적 발표
           반도체 부문 회복세 보여..."
    """
    try:
        # 뉴스 API 클라이언트를 통해 검색 실행
        result = news_client.search_news(query)

        # 검색 결과에서 기사 목록 추출, 최대 3개로 제한
        articles = result.get("articles", [])[:3]

        # 검색 결과가 없는 경우 안내 메시지 반환
        if not articles:
            return f"{query} 관련 뉴스를 찾을 수 없습니다."

        # 사용자 친화적인 형태로 뉴스 요약 생성
        news_summary = f"{query} 관련 뉴스 {len(articles)}건:\n\n"

        # 각 기사를 순번과 함께 정리
        for i, article in enumerate(articles, 1):
            title = article["title"]  # 기사 제목
            description = article["description"]  # 기사 요약
            news_summary += f"{i}. {title}\n   {description}\n\n"

        return news_summary

    except Exception as e:
        # API 오류, 네트워크 문제 등 모든 예외 상황 처리
        return f"뉴스 검색 중 오류 발생: {str(e)}"


@tool
def get_weather(city: str) -> str:
    """
    날씨 조회 및 옷차림 추천 도구

    지정된 도시의 현재 날씨 정보를 조회하고,
    온도를 기반으로 적절한 옷차림을 추천합니다.
    OpenWeatherMap과 같은 날씨 API를 통해 실시간 정보를 가져옵니다.

    Args:
        city (str): 날씨를 조회할 도시명 (예: "Seoul", "서울", "Tokyo")

    Returns:
        str: 날씨 정보와 옷차림 추천을 포함한 종합된 메시지
            - 현재 날씨 설명 (맑음, 흐림, 비 등)
            - 온도 (섭씨)
            - 습도 (%)
            - 온도 기반 옷차림 추천

    Example:
        >>> get_weather.invoke({"city": "Seoul"})
        "Seoul 현재 날씨: clear sky, 온도: 15°C, 습도: 60%
        👔 옷차림 추천: 가디건, 얇은 재킷 추천"
    """
    try:
        result = weather_client.get_current_weather(city)
        weather_desc = result["weather"][0]["description"]
        temp = result["main"]["temp"]
        humidity = result["main"]["humidity"]

        # 온도 기반 옷차림 추천
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

        return f"💱 {from_currency} → {to_currency}: {rate}"
    except Exception as e:
        return f"환율 조회 중 오류 발생: {str(e)}"


@tool
def get_stock_price(symbol: str) -> str:
    """주식 가격을 조회합니다. symbol은 종목코드입니다."""
    try:
        result = stock_client.get_stock_price(symbol)
        name = result["name"]
        price = result["price"]

        if isinstance(price, int):
            return f"📈 {name}({symbol}): {price:,}원"
        else:
            return f"📈 {name}({symbol}): ${price}"
    except Exception as e:
        return f"주식 조회 중 오류 발생: {str(e)}"


@tool
def add_schedule(date: str, task: str) -> str:
    """일정을 추가합니다. date는 YYYY-MM-DD 형식, task는 일정 내용입니다."""
    schedule_dir = "tmp"
    schedule_file = f"{schedule_dir}/schedules.json"

    try:
        # tmp 디렉토리가 없으면 생성
        os.makedirs(schedule_dir, exist_ok=True)

        # 기존 일정 파일 읽기
        try:
            with open(schedule_file, "r", encoding="utf-8") as f:
                schedules = json.load(f)
        except FileNotFoundError:
            schedules = {}

        # 새 일정 추가
        if date not in schedules:
            schedules[date] = []

        schedule_item = {
            "task": task,
            "created_at": datetime.now().isoformat(),
            "completed": False,
        }
        schedules[date].append(schedule_item)

        # 파일에 저장
        with open(schedule_file, "w", encoding="utf-8") as f:
            json.dump(schedules, f, ensure_ascii=False, indent=2)

        return f"📅 {date}에 '{task}' 일정이 추가되었습니다."
    except Exception as e:
        return f"일정 추가 중 오류 발생: {str(e)}"


@tool
def get_schedule(date: str) -> str:
    """특정 날짜의 일정을 조회합니다. date는 YYYY-MM-DD 형식입니다."""
    schedule_file = "tmp/schedules.json"

    try:
        with open(schedule_file, "r", encoding="utf-8") as f:
            schedules = json.load(f)

        if date in schedules:
            tasks = schedules[date]
            if not tasks:
                return f"📅 {date}에 등록된 일정이 없습니다."

            schedule_list = f"📅 {date} 일정 ({len(tasks)}개):\n\n"
            for i, task_item in enumerate(tasks, 1):
                task_text = (
                    task_item
                    if isinstance(task_item, str)
                    else task_item.get("task", "")
                )
                status = (
                    "✅"
                    if isinstance(task_item, dict) and task_item.get("completed", False)
                    else "⭕"
                )
                schedule_list += f"{status} {i}. {task_text}\n"

            return schedule_list
        else:
            return f"📅 {date}에 등록된 일정이 없습니다."
    except FileNotFoundError:
        return f"📅 {date}에 등록된 일정이 없습니다."
    except Exception as e:
        return f"일정 조회 중 오류 발생: {str(e)}"
