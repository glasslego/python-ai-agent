from src.simple_assistant.utils.api_client import NewsAPIClient


class NewsAgent:
    def __init__(self):
        self.client = NewsAPIClient()
        self.categories = [
            "general",
            "business",
            "technology",
            "sports",
            "entertainment",
        ]

    def search_news(self, query: str, category: str = "general", limit: int = 5):
        """뉴스 검색"""
        try:
            result = self.client.search_news(query)
            articles = result.get("articles", [])[:limit]

            processed_articles = []
            for article in articles:
                processed_articles.append(
                    {
                        "title": article.get("title", ""),
                        "description": article.get("description", ""),
                        "published_at": article.get("publishedAt", ""),
                        "url": article.get("url", ""),
                    }
                )

            return {
                "status": "success",
                "query": query,
                "total_results": len(processed_articles),
                "articles": processed_articles,
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "articles": []}

    def get_top_headlines(self, country: str = "kr", limit: int = 10):
        """주요 헤드라인 뉴스"""
        try:
            # 더미 헤드라인 데이터
            headlines = [
                {
                    "title": "경제 성장률 3.2% 기록",
                    "description": "올해 경제 성장률이 예상을 뛰어넘었습니다.",
                },
                {
                    "title": "AI 기술 발전 가속화",
                    "description": "인공지능 기술이 빠르게 발전하고 있습니다.",
                },
                {
                    "title": "친환경 에너지 투자 확대",
                    "description": "재생 가능 에너지 투자가 늘어나고 있습니다.",
                },
            ]

            return {
                "status": "success",
                "country": country,
                "headlines": headlines[:limit],
            }
        except Exception as e:
            return {"status": "error", "message": str(e), "headlines": []}

    def analyze_sentiment(self, articles):
        """뉴스 감성 분석 (간단한 키워드 기반)"""
        positive_words = ["성장", "상승", "증가", "개선", "성공", "발전"]
        negative_words = ["하락", "감소", "악화", "실패", "위기", "문제"]

        results = []
        for article in articles:
            text = article.get("title", "") + " " + article.get("description", "")

            positive_count = sum(1 for word in positive_words if word in text)
            negative_count = sum(1 for word in negative_words if word in text)

            if positive_count > negative_count:
                sentiment = "긍정적"
            elif negative_count > positive_count:
                sentiment = "부정적"
            else:
                sentiment = "중립적"

            results.append(
                {
                    "title": article.get("title", ""),
                    "sentiment": sentiment,
                    "positive_score": positive_count,
                    "negative_score": negative_count,
                }
            )

        return results
