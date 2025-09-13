"""
간단한 뉴스 분석 에이전트
핵심 기능만 포함한 리팩토링된 버전
"""

from datetime import datetime
from typing import Any, Dict, List

import pandas as pd
import requests
from loguru import logger
from text_util.pre_processor import NewsPreprocessor
from text_util.sentiment_analyzer import SentimentAnalyzer
from text_util.text_analyzer import TextAnalyzer


class SimpleNewsAgent:
    """간소화된 뉴스 분석 에이전트"""

    def __init__(self, api_key: str = "9268c1b56a644f2fa03cc5979fbfcb75"):
        self.api_key = api_key
        self.base_url = "https://newsapi.org/v2"

        # 핵심 분석 모듈만 초기화
        self.preprocessor = NewsPreprocessor()
        self.text_analyzer = TextAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()

        # 데이터
        self.current_data = None
        self.processed_data = None

    def collect_news(
        self, category: str = "technology", num_articles: int = 20
    ) -> pd.DataFrame:
        """뉴스 데이터 수집"""
        logger.info(f"뉴스 수집 시작: {category} 카테고리, {num_articles}개")

        url = f"{self.base_url}/top-headlines"
        params = {
            "apiKey": self.api_key,
            "category": category,
            "country": "us",
            "pageSize": min(num_articles, 100),
        }

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()

            articles = data.get("articles", [])
            if not articles:
                logger.warning("수집된 뉴스가 없습니다")
                return pd.DataFrame()

            # DataFrame 생성 (preprocessor가 기대하는 컬럼명 사용)
            news_data = []
            for article in articles[:num_articles]:
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

            self.current_data = pd.DataFrame(news_data)
            logger.success(f"뉴스 수집 완료: {len(self.current_data)}개")
            return self.current_data

        except Exception as e:
            logger.error(f"뉴스 수집 실패: {e}")
            return pd.DataFrame()

    def preprocess_data(self) -> pd.DataFrame:
        """데이터 전처리"""
        if self.current_data is None or self.current_data.empty:
            logger.error("수집된 데이터가 없습니다")
            return pd.DataFrame()

        logger.info("데이터 전처리 시작")
        self.processed_data = self.preprocessor.preprocess_dataframe(self.current_data)
        logger.success(f"전처리 완료: {len(self.processed_data)}개")
        return self.processed_data

    def analyze_keywords(self, n_keywords: int = 20) -> List[tuple]:
        """TF-IDF 키워드 분석"""
        if self.processed_data is None or self.processed_data.empty:
            logger.error("전처리된 데이터가 없습니다")
            return []

        logger.info("키워드 분석 시작")
        texts = self.processed_data["processed_text"].tolist()
        self.text_analyzer.fit_transform(texts)

        keywords = self.text_analyzer.get_top_keywords_global(n_keywords)
        logger.success(f"키워드 분석 완료: {len(keywords)}개")
        return keywords

    def analyze_sentiment(self) -> Dict[str, Any]:
        """감정 분석"""
        if self.processed_data is None or self.processed_data.empty:
            logger.error("전처리된 데이터가 없습니다")
            return {}

        logger.info("감정 분석 시작")
        sentiment_df = self.sentiment_analyzer.analyze_dataframe(self.processed_data)
        stats = self.sentiment_analyzer.get_sentiment_statistics(sentiment_df)
        logger.success("감정 분석 완료")
        return stats

    def run_full_analysis(
        self, category: str = "technology", num_articles: int = 20
    ) -> Dict[str, Any]:
        """전체 분석 실행"""
        logger.info("전체 분석 시작")
        results = {}

        # 1. 데이터 수집
        df = self.collect_news(category, num_articles)
        if df.empty:
            return {"error": "데이터 수집 실패"}

        # 2. 전처리
        processed_df = self.preprocess_data()
        if processed_df.empty:
            return {"error": "데이터 전처리 실패"}

        # 3. 키워드 분석
        keywords = self.analyze_keywords(20)
        results["keywords"] = keywords[:10]  # 상위 10개만

        # 4. 감정 분석
        sentiment_stats = self.analyze_sentiment()
        results["sentiment"] = sentiment_stats

        # 5. 기본 통계
        results["total_articles"] = len(self.current_data)
        results["processed_articles"] = len(self.processed_data)
        results["timestamp"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        logger.success("전체 분석 완료")
        return results

    def save_data(self, filename: str = None) -> str:
        """데이터 저장"""
        if self.current_data is None or self.current_data.empty:
            logger.error("저장할 데이터가 없습니다")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"news_data/news_data_{timestamp}.csv"

        self.current_data.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.success(f"데이터 저장 완료: {filename}")
        return filename

    def get_summary(self) -> Dict[str, Any]:
        """현재 상태 요약"""
        return {
            "has_raw_data": self.current_data is not None
            and not self.current_data.empty,
            "has_processed_data": self.processed_data is not None
            and not self.processed_data.empty,
            "raw_count": len(self.current_data) if self.current_data is not None else 0,
            "processed_count": (
                len(self.processed_data) if self.processed_data is not None else 0
            ),
        }


if __name__ == "__main__":
    # 사용 예제
    agent = SimpleNewsAgent()

    # 전체 분석 실행
    results = agent.run_full_analysis(category="technology", num_articles=10)

    if "error" not in results:
        print("\n=== 뉴스 분석 결과 ===")
        print(f"총 기사 수: {results['total_articles']}")
        print(f"처리된 기사 수: {results['processed_articles']}")

        print("\n상위 키워드:")
        for i, (keyword, score) in enumerate(results["keywords"], 1):
            print(f"{i}. {keyword} ({score:.3f})")

        print("\n감정 분석:")
        sentiment = results["sentiment"]
        print(f"긍정: {sentiment.get('positive_ratio', 0):.1%}")
        print(f"부정: {sentiment.get('negative_ratio', 0):.1%}")
        print(f"중립: {sentiment.get('neutral_ratio', 0):.1%}")

        # 데이터 저장
        filename = agent.save_data()
        print(f"\n데이터 저장됨: {filename}")
    else:
        print(f"분석 실패: {results['error']}")
