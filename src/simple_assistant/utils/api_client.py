import os

import requests
from dotenv import load_dotenv

load_dotenv()


class NewsAPIClient:
    def __init__(self):
        self.base_url = "https://newsapi.org/v2"
        # API 키가 없으면 더미 데이터 반환
        self.api_key = os.getenv("NEWS_API_KEY", "dummy")

    def search_news(self, query, language="ko"):
        if self.api_key == "dummy":
            return {
                "articles": [
                    {
                        "title": f"[더미] {query} 관련 뉴스 1",
                        "description": f"{query}에 대한 최신 소식입니다.",
                        "publishedAt": "2024-03-15T10:00:00Z",
                    }
                ]
            }

        url = f"{self.base_url}/everything"
        params = {
            "q": query,
            "language": language,
            "sortBy": "publishedAt",
            "apiKey": self.api_key,
        }
        response = requests.get(url, params=params)
        return response.json()


class WeatherAPIClient:
    def __init__(self):
        self.base_url = "https://api.openweathermap.org/data/2.5"
        self.api_key = os.getenv("WEATHER_API_KEY", "dummy")

    def get_current_weather(self, city="Seoul"):
        if self.api_key == "dummy":
            return {
                "weather": [{"description": "맑음"}],
                "main": {"temp": 22, "humidity": 60},
                "name": city,
            }

        url = f"{self.base_url}/weather"
        params = {"q": city, "appid": self.api_key, "units": "metric", "lang": "kr"}
        response = requests.get(url, params=params)
        return response.json()


class ExchangeRateClient:
    def __init__(self):
        self.api_key = os.getenv("EXCHANGE_API_KEY", "dummy")
        self.base_url = f"https://v6.exchangerate-api.com/v6/{self.api_key}/latest"

    def get_exchange_rate(self, from_currency="USD", to_currency="KRW"):
        # 더미 데이터 반환
        dummy_rates = {"USD": {"KRW": 1320}, "EUR": {"KRW": 1450}, "JPY": {"KRW": 9.2}}

        if from_currency in dummy_rates and to_currency in dummy_rates[from_currency]:
            return {
                "rates": {to_currency: dummy_rates[from_currency][to_currency]},
                "base": from_currency,
            }

        return {"rates": {to_currency: 1320}, "base": from_currency}


class StockAPIClient:
    def __init__(self):
        # 더미 주식 데이터
        self.dummy_stocks = {
            "005930": {"name": "삼성전자", "price": 75000},
            "000660": {"name": "SK하이닉스", "price": 120000},
            "AAPL": {"name": "Apple Inc.", "price": 180.50},
            "MSFT": {"name": "Microsoft", "price": 420.00},
        }

    def get_stock_price(self, symbol):
        if symbol in self.dummy_stocks:
            return self.dummy_stocks[symbol]
        return {"name": "알 수 없는 종목", "price": 0}


if __name__ == "__main__":
    news_client = NewsAPIClient()
    weather_client = WeatherAPIClient()
    exchange_client = ExchangeRateClient()
    stock_client = StockAPIClient()

    print(news_client.search_news("인공지능"))
    print(weather_client.get_current_weather("Seoul"))
    print(exchange_client.get_exchange_rate("USD", "KRW"))
    print(stock_client.get_stock_price("005930"))
