import json
from datetime import datetime

from src.simple_assistant.utils.api_client import ExchangeRateClient, StockAPIClient


class FinanceAgent:
    def __init__(self):
        self.exchange_client = ExchangeRateClient()
        self.stock_client = StockAPIClient()
        self.watchlist_file = "/tmp/stock_watchlist.json"

    def get_exchange_rate(self, from_currency: str = "USD", to_currency: str = "KRW"):
        """환율 정보 조회"""
        try:
            result = self.exchange_client.get_exchange_rate(from_currency, to_currency)
            rate = result["rates"][to_currency]

            return {
                "status": "success",
                "from_currency": from_currency,
                "to_currency": to_currency,
                "rate": rate,
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_multiple_exchange_rates(self, base_currency: str = "USD"):
        """여러 통화 환율 조회"""
        currencies = ["KRW", "EUR", "JPY", "GBP", "CNY"]
        rates = {}

        for currency in currencies:
            if currency != base_currency:
                result = self.get_exchange_rate(base_currency, currency)
                if result["status"] == "success":
                    rates[currency] = result["rate"]

        return {
            "status": "success",
            "base_currency": base_currency,
            "rates": rates,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    def get_stock_price(self, symbol: str):
        """주식 가격 조회"""
        try:
            result = self.stock_client.get_stock_price(symbol)

            return {
                "status": "success",
                "symbol": symbol,
                "name": result["name"],
                "price": result["price"],
                "currency": "KRW" if symbol.isdigit() else "USD",
                "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "symbol": symbol}

    def add_to_watchlist(self, symbol: str, name: str = ""):
        """관심 종목 추가"""
        try:
            try:
                with open(self.watchlist_file, "r", encoding="utf-8") as f:
                    watchlist = json.load(f)
            except FileNotFoundError:
                watchlist = {}

            watchlist[symbol] = {
                "name": name or f"종목-{symbol}",
                "added_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }

            with open(self.watchlist_file, "w", encoding="utf-8") as f:
                json.dump(watchlist, f, ensure_ascii=False, indent=2)

            return {
                "status": "success",
                "message": f"{symbol} 종목이 관심목록에 추가되었습니다",
                "symbol": symbol,
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def get_watchlist(self):
        """관심 종목 목록 조회"""
        try:
            with open(self.watchlist_file, "r", encoding="utf-8") as f:
                watchlist = json.load(f)

            # 각 종목의 현재 가격도 함께 조회
            watchlist_with_prices = {}
            for symbol, info in watchlist.items():
                stock_info = self.get_stock_price(symbol)
                watchlist_with_prices[symbol] = {
                    **info,
                    "current_price": (
                        stock_info.get("price", 0)
                        if stock_info["status"] == "success"
                        else 0
                    ),
                    "currency": (
                        stock_info.get("currency", "KRW")
                        if stock_info["status"] == "success"
                        else "KRW"
                    ),
                }

            return {"status": "success", "watchlist": watchlist_with_prices}
        except FileNotFoundError:
            return {
                "status": "success",
                "watchlist": {},
                "message": "관심 종목이 없습니다",
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def remove_from_watchlist(self, symbol: str):
        """관심 종목 제거"""
        try:
            with open(self.watchlist_file, "r", encoding="utf-8") as f:
                watchlist = json.load(f)

            if symbol in watchlist:
                del watchlist[symbol]

                with open(self.watchlist_file, "w", encoding="utf-8") as f:
                    json.dump(watchlist, f, ensure_ascii=False, indent=2)

                return {
                    "status": "success",
                    "message": f"{symbol} 종목이 관심목록에서 제거되었습니다",
                }
            else:
                return {
                    "status": "error",
                    "message": f"{symbol} 종목을 관심목록에서 찾을 수 없습니다",
                }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def calculate_currency_conversion(
        self, amount: float, from_currency: str, to_currency: str
    ):
        """통화 변환 계산"""
        rate_info = self.get_exchange_rate(from_currency, to_currency)

        if rate_info["status"] == "success":
            converted_amount = amount * rate_info["rate"]
            return {
                "status": "success",
                "original_amount": amount,
                "original_currency": from_currency,
                "converted_amount": round(converted_amount, 2),
                "converted_currency": to_currency,
                "exchange_rate": rate_info["rate"],
            }
        else:
            return rate_info


def main():
    print("🧪 FinanceAgent 테스트")

    agent = FinanceAgent()

    # 환율 조회 테스트
    print("\n💱 환율 조회 테스트:")
    usd_krw = agent.get_exchange_rate("USD", "KRW")
    print(f"USD → KRW: {usd_krw}")

    eur_krw = agent.get_exchange_rate("EUR", "KRW")
    print(f"EUR → KRW: {eur_krw}")

    # 다중 환율 조회 테스트
    print("\n💱 다중 환율 조회 테스트:")
    multiple_rates = agent.get_multiple_exchange_rates("USD")
    print(f"다중 환율: {multiple_rates}")

    # 주식 가격 조회 테스트
    print("\n📈 주식 가격 조회 테스트:")
    samsung = agent.get_stock_price("005930")
    print(f"삼성전자: {samsung}")

    apple = agent.get_stock_price("AAPL")
    print(f"애플: {apple}")

    unknown = agent.get_stock_price("UNKNOWN")
    print(f"알 수 없는 종목: {unknown}")

    # 관심 종목 관리 테스트
    print("\n⭐ 관심 종목 관리 테스트:")

    # 관심 종목 추가
    add_result = agent.add_to_watchlist("005930", "삼성전자")
    print(f"종목 추가: {add_result}")

    add_result2 = agent.add_to_watchlist("AAPL", "Apple Inc.")
    print(f"종목 추가2: {add_result2}")

    # 관심 종목 목록 조회
    watchlist = agent.get_watchlist()
    print(f"관심 종목 목록: {watchlist}")

    # 관심 종목 제거
    remove_result = agent.remove_from_watchlist("005930")
    print(f"종목 제거: {remove_result}")

    # 제거 후 목록 확인
    watchlist_after = agent.get_watchlist()
    print(f"제거 후 관심 종목 목록: {watchlist_after}")

    # 통화 변환 계산 테스트
    print("\n💰 통화 변환 계산 테스트:")
    conversion = agent.calculate_currency_conversion(100, "USD", "KRW")
    print(f"100 USD → KRW: {conversion}")

    conversion2 = agent.calculate_currency_conversion(1000000, "KRW", "USD")
    print(f"1,000,000 KRW → USD: {conversion2}")

    print("\n🎉 모든 테스트 완료!")


if __name__ == "__main__":
    main()
