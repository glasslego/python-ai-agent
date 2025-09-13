"""
간단한 뉴스 데이터 수집기
핵심 기능만 포함
"""

from typing import Any, Dict, List

import pandas as pd
import requests
from loguru import logger


class SimpleNewsCollector:
    """간소화된 뉴스 수집기"""

    def __init__(self, api_key: str = "9268c1b56a644f2fa03cc5979fbfcb75"):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def get_news(
        self, category: str = "technology", count: int = 20
    ) -> List[Dict[str, Any]]:
        """뉴스 기사 수집"""
        url = f"{self.base_url}/top-headlines"
        params = {
            "apiKey": self.api_key,
            "category": category,
            "country": "us",
            "pageSize": min(count, 100),
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            return data.get("articles", [])
        except Exception as e:
            logger.error(f"뉴스 수집 실패: {e}")
            return []

    def to_dataframe(self, articles: List[Dict[str, Any]]) -> pd.DataFrame:
        """기사 리스트를 DataFrame으로 변환"""
        if not articles:
            return pd.DataFrame()

        news_data = []
        for article in articles:
            news_data.append(
                {
                    "Title": article.get("title", ""),
                    "Description": article.get("description", ""),
                    "Content": article.get("content", ""),
                    "URL": article.get("url", ""),
                    "PublishedAt": article.get("publishedAt", ""),
                    "Source": article.get("source", {}).get("name", ""),
                }
            )

        return pd.DataFrame(news_data)

    def collect_and_save(
        self, category: str = "technology", count: int = 20, filename: str = "news.csv"
    ) -> pd.DataFrame:
        """뉴스 수집 후 CSV 저장"""
        articles = self.get_news(category, count)
        df = self.to_dataframe(articles)

        if not df.empty:
            df.to_csv(filename, index=False, encoding="utf-8-sig")
            logger.success(f"뉴스 {len(df)}개 수집 및 저장: {filename}")

        return df


if __name__ == "__main__":
    collector = SimpleNewsCollector()
    df = collector.collect_and_save("technology", 10, "tech_news.csv")
    print(f"수집된 기사 수: {len(df)}")
