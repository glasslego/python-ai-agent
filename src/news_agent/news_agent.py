import os
from datetime import datetime
from typing import Any, Dict

from data_collector import NewsDataCollector
from loguru import logger
from text_util import NewsPreprocessor, SentimentAnalyzer, TextAnalyzer

from src.news_agent.text_util.similarity_analyzer import SimilarityAnalyzer
from src.news_agent.text_util.topic_modeler import TopicModeler


class RefactoredNewsAgent:
    """간소화된 뉴스 분석 에이전트"""

    def __init__(
        self,
        api_key: str = "9268c1b56a644f2fa03cc5979fbfcb75",
        data_dir: str = "news_data",
    ):
        self.api_key = api_key
        self.data_dir = data_dir

        # 데이터 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)

        # 핵심 모듈만 초기화
        self.data_collector = NewsDataCollector(api_key)
        self.preprocessor = NewsPreprocessor()
        self.text_analyzer = TextAnalyzer()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.topic_modeler = TopicModeler()
        self.similarity_analyzer = SimilarityAnalyzer()

        # 데이터
        self.current_data = None
        self.processed_data = None
        self.tfidf_matrix = None

    def collect_news(
        self, n_articles: int = 20, category: str = "technology"
    ) -> Dict[str, Any]:
        """뉴스 데이터 수집"""
        logger.info(f"뉴스 수집 시작: {category} 카테고리, {n_articles}개")

        try:
            df = self.data_collector.collect_and_process(
                category=category, num_articles=n_articles, include_content=True
            )

            if df.empty:
                return {"success": False, "message": "수집된 뉴스가 없습니다"}

            # 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/news_{category}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding="utf-8-sig")

            self.current_data = df
            logger.success(f"뉴스 수집 완료: {len(df)}개")

            return {
                "success": True,
                "count": len(df),
                "filename": filename,
                "message": f"{len(df)}개 뉴스 수집 완료",
            }

        except Exception as e:
            logger.error(f"뉴스 수집 실패: {e}")
            return {"success": False, "error": str(e)}

    def preprocess_data(self) -> Dict[str, Any]:
        """데이터 전처리"""
        if self.current_data is None:
            return {"success": False, "message": "먼저 뉴스를 수집해주세요"}

        logger.info("데이터 전처리 시작")

        try:
            self.processed_data = self.preprocessor.preprocess_dataframe(
                self.current_data
            )

            if self.processed_data.empty:
                return {
                    "success": False,
                    "message": "전처리 후 유효한 데이터가 없습니다",
                }

            logger.success(f"전처리 완료: {len(self.processed_data)}개")
            return {
                "success": True,
                "count": len(self.processed_data),
                "message": "데이터 전처리 완료",
            }

        except Exception as e:
            logger.error(f"전처리 실패: {e}")
            return {"success": False, "error": str(e)}

    def analyze_keywords(self, n_keywords: int = 20) -> Dict[str, Any]:
        """TF-IDF 키워드 분석"""
        if not self._ensure_processed_data():
            return {"success": False, "message": "전처리된 데이터가 필요합니다"}

        logger.info("키워드 분석 시작")

        try:
            texts = self.processed_data["processed_text"].tolist()
            self.tfidf_matrix = self.text_analyzer.fit_transform(texts)
            keywords = self.text_analyzer.get_top_keywords_global(n_keywords)

            logger.success(f"키워드 분석 완료: {len(keywords)}개")
            return {
                "success": True,
                "keywords": keywords[:10],  # 상위 10개만 반환
                "total_keywords": len(keywords),
                "message": "키워드 분석 완료",
            }

        except Exception as e:
            logger.error(f"키워드 분석 실패: {e}")
            return {"success": False, "error": str(e)}

    def analyze_sentiment(self) -> Dict[str, Any]:
        """감정 분석"""
        if not self._ensure_processed_data():
            return {"success": False, "message": "전처리된 데이터가 필요합니다"}

        logger.info("감정 분석 시작")

        try:
            sentiment_df = self.sentiment_analyzer.analyze_dataframe(
                self.processed_data
            )
            stats = self.sentiment_analyzer.get_sentiment_statistics(sentiment_df)

            logger.success("감정 분석 완료")
            return {"success": True, "statistics": stats, "message": "감정 분석 완료"}

        except Exception as e:
            logger.error(f"감정 분석 실패: {e}")
            return {"success": False, "error": str(e)}

    def analyze_topics(self, n_topics: int = 5) -> Dict[str, Any]:
        """토픽 모델링 분석"""
        if not self._ensure_processed_data():
            return {"success": False, "message": "전처리된 데이터가 필요합니다"}

        logger.info("토픽 모델링 시작")

        try:
            texts = self.processed_data["processed_text"].tolist()
            self.topic_modeler = TopicModeler(n_topics=n_topics)
            doc_topics = self.topic_modeler.fit_transform(texts)
            print(doc_topics)

            topic_df = self.topic_modeler.analyze(self.processed_data)
            print(topic_df.head())
            topics = self.topic_modeler.get_top_words(5)

            logger.success(f"토픽 모델링 완료: {n_topics}개 토픽")
            return {
                "success": True,
                "topics": topics,
                "n_topics": n_topics,
                "message": f"{n_topics}개 토픽 추출 완료",
            }

        except Exception as e:
            logger.error(f"토픽 모델링 실패: {e}")
            return {"success": False, "error": str(e)}

    def analyze_similarity(self) -> Dict[str, Any]:
        """유사도 분석"""
        if self.tfidf_matrix is None:
            return {"success": False, "message": "먼저 키워드 분석을 실행해주세요"}

        logger.info("유사도 분석 시작")

        try:
            self.similarity_analyzer.set_tfidf_matrix(self.tfidf_matrix)
            stats = self.similarity_analyzer.get_statistics()
            similar_pairs = self.similarity_analyzer.find_similar_pairs(
                self.processed_data, 5
            )

            logger.success("유사도 분석 완료")
            return {
                "success": True,
                "statistics": stats,
                "similar_pairs": similar_pairs[:3],  # 상위 3개만
                "message": "유사도 분석 완료",
            }

        except Exception as e:
            logger.error(f"유사도 분석 실패: {e}")
            return {"success": False, "error": str(e)}

    def run_full_analysis(
        self, category: str = "technology", n_articles: int = 20
    ) -> Dict[str, Any]:
        """전체 분석 실행"""
        logger.info("전체 뉴스 분석 시작")

        results = {}

        # 1. 데이터 수집
        logger.info("1/5: 데이터 수집")
        collect_result = self.collect_news(n_articles, category)
        results["collection"] = collect_result
        if not collect_result.get("success"):
            return results

        # 2. 전처리
        logger.info("2/5: 데이터 전처리")
        preprocess_result = self.preprocess_data()
        results["preprocessing"] = preprocess_result
        if not preprocess_result.get("success"):
            return results

        # 3. 키워드 분석
        logger.info("3/5: 키워드 분석")
        keyword_result = self.analyze_keywords(20)
        results["keywords"] = keyword_result

        # 4. 감정 분석
        logger.info("4/5: 감정 분석")
        sentiment_result = self.analyze_sentiment()
        results["sentiment"] = sentiment_result

        # 5. 토픽 분석
        logger.info("5/5: 토픽 분석")
        topic_result = self.analyze_topics(3)
        results["topics"] = topic_result

        # 성공 개수 계산
        success_count = sum(1 for r in results.values() if r.get("success", False))

        logger.success(f"전체 분석 완료: {success_count}/5 성공")

        results["summary"] = {
            "total_analyses": 5,
            "successful_analyses": success_count,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        return results

    def get_status(self) -> Dict[str, Any]:
        """현재 상태 반환"""
        return {
            "has_raw_data": self.current_data is not None,
            "has_processed_data": self.processed_data is not None,
            "has_tfidf_matrix": self.tfidf_matrix is not None,
            "raw_count": len(self.current_data) if self.current_data is not None else 0,
            "processed_count": (
                len(self.processed_data) if self.processed_data is not None else 0
            ),
        }

    def save_results(self, filename: str = None) -> str:
        """결과 저장"""
        if self.current_data is None:
            logger.error("저장할 데이터가 없습니다")
            return ""

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/analysis_results_{timestamp}.csv"

        self.current_data.to_csv(filename, index=False, encoding="utf-8-sig")
        logger.success(f"결과 저장 완료: {filename}")
        return filename

    def _ensure_processed_data(self) -> bool:
        """전처리된 데이터 확인"""
        if self.processed_data is None:
            if self.current_data is None:
                return False
            self.preprocess_data()
        return self.processed_data is not None


if __name__ == "__main__":
    # 사용 예제
    logger.info("리팩토링된 뉴스 에이전트 테스트")

    agent = RefactoredNewsAgent()

    # 전체 분석 실행
    results = agent.run_full_analysis("technology", 5)

    if results.get("summary", {}).get("successful_analyses", 0) > 0:
        logger.success("분석 성공!")

        # 키워드 결과 출력
        if "keywords" in results and results["keywords"].get("success"):
            keywords = results["keywords"]["keywords"]
            logger.info("상위 키워드:")
            for i, (keyword, score) in enumerate(keywords[:5], 1):
                logger.info(f"  {i}. {keyword}: {score:.3f}")

        # 감정 분석 결과 출력
        if "sentiment" in results and results["sentiment"].get("success"):
            stats = results["sentiment"]["statistics"]
            logger.info(f"감정 분석: 긍정 {stats.get('positive_ratio', 0):.1%}")

        # 상태 확인
        status = agent.get_status()
        logger.info(f"상태: {status['processed_count']}개 기사 처리됨")

    else:
        logger.error("분석 실패")
