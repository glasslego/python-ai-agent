"""
뉴스 데이터 수집 모듈
"""

import time
from typing import Any, Dict, List

import pandas as pd
import requests
from loguru import logger

from src.news_analysis.parse_url import parse_article_content

NEWS_API_KEY = "9268c1b56a644f2fa03cc5979fbfcb75"


class NewsDataCollector:
    """뉴스 API를 통한 데이터 수집 클래스"""

    def __init__(self, api_key: str = "9268c1b56a644f2fa03cc5979fbfcb75"):
        """
        Args:
            api_key: NewsAPI 키
        """
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

    def get_news_json(
        self, category: str = "technology", country: str = "us", num_articles: int = 10
    ) -> List[Dict[str, Any]]:
        """
        NewsAPI를 사용하여 뉴스 데이터를 JSON 형태로 수집

        Args:
            category: 뉴스 카테고리 (technology, business, health 등)
            country: 국가 코드 (us, kr 등)
            num_articles: 수집할 기사 수

        Returns:
            뉴스 기사 리스트
        """
        if not self.api_key:
            logger.error("NewsAPI 키가 설정되지 않았습니다.")
            return []

        url = f"{self.base_url}/top-headlines"
        params = {
            "apiKey": self.api_key,
            "category": category,
            "country": country,
            "pageSize": min(num_articles, 100),  # API 제한
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            if not articles:
                logger.info("새로운 뉴스를 찾을 수 없습니다.")
                return []

            return articles[:num_articles]

        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 중 에러가 발생했습니다: {e}")
            return []
        except Exception as e:
            logger.error(f"알 수 없는 에러가 발생했습니다: {e}")
            return []

    def get_news_by_query(
        self, query: str, num_articles: int = 10, sort_by: str = "publishedAt"
    ) -> List[Dict[str, Any]]:
        """
        검색 쿼리를 사용하여 뉴스 데이터 수집

        Args:
            query: 검색 쿼리
            num_articles: 수집할 기사 수
            sort_by: 정렬 기준 (publishedAt, relevancy, popularity)

        Returns:
            뉴스 기사 리스트
        """
        url = f"{self.base_url}/everything"
        params = {
            "apiKey": self.api_key,
            "q": query,
            "sortBy": sort_by,
            "pageSize": min(num_articles, 100),
            "language": "en",
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            return articles[:num_articles]

        except requests.exceptions.RequestException as e:
            logger.error(f"API 요청 중 에러가 발생했습니다: {e}")
            return []

    def articles_to_dataframe(self, articles: List[Dict[str, Any]]) -> pd.DataFrame:
        """
        뉴스 기사 리스트를 pandas DataFrame으로 변환

        Args:
            articles: 뉴스 기사 리스트

        Returns:
            pandas DataFrame
        """
        if not articles:
            return pd.DataFrame()

        data = {
            "Source": [
                article.get("source", {}).get("name", "") for article in articles
            ],
            "Author": [article.get("author", "") for article in articles],
            "Title": [article.get("title", "") for article in articles],
            "Description": [article.get("description", "") for article in articles],
            "URL": [article.get("url", "") for article in articles],
            "Published At": [article.get("publishedAt", "") for article in articles],
            "Content": [article.get("content", "") for article in articles],
        }

        df = pd.DataFrame(data)
        logger.info(f"DataFrame 생성 완료: {df.shape[0]}개 기사, {df.shape[1]}개 컬럼")

        return df

    def get_news_content(self, url: str) -> Dict[str, Any]:
        """
        URL에서 뉴스 기사의 상세 내용 추출

        Args:
            url: 뉴스 기사 URL

        Returns:
            파싱된 뉴스 내용 딕셔너리
        """
        try:
            parsed_data = parse_article_content(url)

            return {
                "Parsed Content": parsed_data.get("content", ""),
                "Parsed Author": parsed_data.get("author", ""),
                "Meta Description": parsed_data.get("meta_description", ""),
                "Parsed Publish Date": parsed_data.get("publish_date", ""),
                "Tags": ", ".join(parsed_data.get("tags", [])),
                "Word Count": parsed_data.get("word_count", 0),
                "Reading Time (min)": parsed_data.get("reading_time", 0),
                "Parse Status": parsed_data.get("status", "not_attempted"),
            }

        except Exception as e:
            logger.error(f"뉴스 기사 내용 파싱 중 에러 발생: {e}")
            return {
                "Parsed Content": "",
                "Parsed Author": "",
                "Meta Description": "",
                "Parsed Publish Date": "",
                "Tags": "",
                "Word Count": 0,
                "Reading Time (min)": 0,
                "Parse Status": "error",
            }

    def add_parsed_content(self, df: pd.DataFrame, delay: float = 1.0) -> pd.DataFrame:
        """
        DataFrame의 URL들을 파싱하여 상세 내용 추가

        Args:
            df: 뉴스 데이터 DataFrame
            delay: 요청 간 대기 시간 (초)

        Returns:
            파싱된 내용이 추가된 DataFrame
        """
        df_copy = df.copy()

        # 새 컬럼들 초기화
        new_columns = {
            "Parsed Content": "",
            "Parsed Author": "",
            "Meta Description": "",
            "Parsed Publish Date": "",
            "Tags": "",
            "Word Count": 0,
            "Reading Time (min)": 0,
            "Parse Status": "pending",
        }

        for col, default_val in new_columns.items():
            df_copy[col] = default_val

        # 각 URL 파싱
        for index, row in df_copy.iterrows():
            logger.info(f"파싱 중 [{index + 1}/{len(df_copy)}] {row['Title'][:50]}...")

            url = row["URL"]
            parsed_data = self.get_news_content(url)

            # 파싱 결과를 DataFrame에 할당
            for key, value in parsed_data.items():
                df_copy.at[index, key] = value

            # 서버 부하 방지를 위한 대기
            if delay > 0:
                time.sleep(delay)

        return df_copy

    def collect_and_process(
        self,
        category: str = "technology",
        num_articles: int = 10,
        include_content: bool = True,
    ) -> pd.DataFrame:
        """
        뉴스 수집부터 처리까지 통합 메서드

        Args:
            category: 뉴스 카테고리
            num_articles: 수집할 기사 수
            include_content: 상세 내용 파싱 여부

        Returns:
            처리된 뉴스 DataFrame
        """
        logger.info(f"{category} 카테고리의 최신 뉴스 {num_articles}개를 수집합니다...")

        # 뉴스 수집
        articles = self.get_news_json(category=category, num_articles=num_articles)
        if not articles:
            return pd.DataFrame()

        # DataFrame 변환
        df = self.articles_to_dataframe(articles)

        # 상세 내용 파싱 (선택적)
        if include_content and not df.empty:
            logger.info("상세 내용을 파싱합니다...")
            df = self.add_parsed_content(df)

        logger.success("뉴스 수집 및 처리 완료!")
        return df


if __name__ == "__main__":
    collector = NewsDataCollector()

    # IT 뉴스 수집 예제
    df = collector.collect_and_process(category="technology", num_articles=5)

    if not df.empty:
        logger.info("수집된 뉴스 데이터:")
        logger.info(f"데이터 샘플:\n{df[['Title', 'Source', 'Published At']].head()}")

        # CSV 저장
        df.to_csv("collected_news.csv", index=False)
        logger.success("collected_news.csv로 저장되었습니다.")
