from datetime import datetime, timedelta

from src.simple_assistant.utils.api_client import WeatherAPIClient


class WeatherAgent:
    def __init__(self):
        self.client = WeatherAPIClient()

    def get_current_weather(self, city: str = "Seoul"):
        """현재 날씨 조회"""
        try:
            result = self.client.get_current_weather(city)

            return {
                "status": "success",
                "city": result.get("name", city),
                "weather": {
                    "description": result["weather"][0]["description"],
                    "temperature": result["main"]["temp"],
                    "humidity": result["main"]["humidity"],
                    "feels_like": result["main"].get(
                        "feels_like", result["main"]["temp"]
                    ),
                },
                "clothing_recommendation": self._get_clothing_recommendation(
                    result["main"]["temp"]
                ),
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "city": city}

    def get_weather_forecast(self, city: str = "Seoul", days: int = 5):
        """날씨 예보 (더미 데이터)"""
        try:
            base_temp = 20
            forecasts = []

            for i in range(days):
                date = (datetime.now() + timedelta(days=i + 1)).strftime("%Y-%m-%d")
                temp = base_temp + (i * 2) - 1  # 간단한 온도 변화

                forecasts.append(
                    {
                        "date": date,
                        "temperature": {"max": temp + 5, "min": temp - 3},
                        "description": "맑음" if i % 2 == 0 else "흐림",
                        "humidity": 60 + (i * 5),
                    }
                )

            return {"status": "success", "city": city, "forecast": forecasts}
        except Exception as e:
            return {"status": "error", "message": str(e), "city": city}

    def _get_clothing_recommendation(self, temperature: float):
        """온도 기반 옷차림 추천"""
        if temperature >= 28:
            return {
                "category": "여름",
                "items": ["반팔", "반바지", "샌들", "선글라스"],
                "advice": "가볍고 통기성 좋은 옷을 입으세요",
            }
        elif temperature >= 23:
            return {
                "category": "초여름",
                "items": ["가벼운 긴팔", "면바지", "운동화"],
                "advice": "얇은 옷을 겹쳐 입으세요",
            }
        elif temperature >= 17:
            return {
                "category": "봄/가을",
                "items": ["가디건", "얇은 재킷", "청바지"],
                "advice": "가볍게 걸칠 수 있는 겉옷을 준비하세요",
            }
        elif temperature >= 12:
            return {
                "category": "쌀쌀함",
                "items": ["자켓", "니트", "긴바지", "운동화"],
                "advice": "따뜻한 옷을 입고 가벼운 겉옷을 준비하세요",
            }
        elif temperature >= 5:
            return {
                "category": "추움",
                "items": ["코트", "목도리", "긴바지", "부츠"],
                "advice": "따뜻한 겉옷과 목도리를 착용하세요",
            }
        else:
            return {
                "category": "매우 추움",
                "items": ["패딩", "장갑", "모자", "목도리", "부츠"],
                "advice": "방한용품을 모두 착용하고 보온에 신경쓰세요",
            }

    def get_weather_alerts(self, city: str = "Seoul"):
        """날씨 알림/경보 (더미 데이터)"""
        return {
            "status": "success",
            "city": city,
            "alerts": [
                {
                    "type": "미세먼지",
                    "level": "보통",
                    "message": "외출 시 마스크 착용 권장",
                }
            ],
        }
