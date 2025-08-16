"""
뉴스 에이전트 메인 오케스트레이터
모든 분석 모듈을 통합하여 자연어 쿼리 기반 뉴스 분석 시스템 제공
"""

import os
from datetime import datetime
from typing import Any, Dict

import pandas as pd
from loguru import logger

from src.news_agent.clustering import NewsClustering
from src.news_agent.data_collector import NewsDataCollector
from src.news_agent.llm_interface import LLMInterface
from src.news_agent.pre_processor import NewsPreprocessor
from src.news_agent.sentiment_analyzer import SentimentAnalyzer
from src.news_agent.similarity_analyzer import SimilarityAnalyzer
from src.news_agent.text_analyzer import TextAnalyzer
from src.news_agent.topic_modeler import TopicModeler


class NewsAgent:
    """통합 뉴스 분석 에이전트"""

    def __init__(
        self,
        api_key: str = "9268c1b56a644f2fa03cc5979fbfcb75",
        data_dir: str = "news_data",
    ):
        """
        Args:
            api_key: NewsAPI 키
            data_dir: 데이터 저장 디렉토리
        """
        self.api_key = api_key
        self.data_dir = data_dir

        # 데이터 디렉토리 생성
        os.makedirs(data_dir, exist_ok=True)

        # 분석 모듈들 초기화
        self.data_collector = NewsDataCollector(api_key)
        self.preprocessor = NewsPreprocessor()
        self.text_analyzer = TextAnalyzer()
        self.clustering = NewsClustering()
        self.topic_modeler = TopicModeler()
        self.sentiment_analyzer = SentimentAnalyzer()
        self.similarity_analyzer = SimilarityAnalyzer()
        self.llm_interface = LLMInterface()

        # 현재 데이터
        self.current_data = None
        self.processed_data = None
        self.tfidf_matrix = None

        # LLM 인터페이스에 함수들 등록
        self._register_functions()

    def _register_functions(self):
        """LLM 인터페이스에 분석 함수들 등록"""
        self.llm_interface.register_function("collect_news", self._collect_news_wrapper)
        self.llm_interface.register_function(
            "analyze_keywords", self._analyze_keywords_wrapper
        )
        self.llm_interface.register_function(
            "analyze_sentiment", self._analyze_sentiment_wrapper
        )
        self.llm_interface.register_function(
            "perform_clustering", self._perform_clustering_wrapper
        )
        self.llm_interface.register_function("model_topics", self._model_topics_wrapper)
        self.llm_interface.register_function(
            "analyze_similarity", self._analyze_similarity_wrapper
        )
        self.llm_interface.register_function(
            "run_comprehensive_analysis", self._run_comprehensive_analysis_wrapper
        )

    def _collect_news_wrapper(
        self, n_articles: int = 20, category: str = "technology", **kwargs
    ):
        """뉴스 수집 래퍼 함수"""
        return self.collect_news(n_articles=n_articles, category=category)

    def _analyze_keywords_wrapper(self, n_keywords: int = 20, **kwargs):
        """키워드 분석 래퍼 함수"""
        return self.analyze_keywords(n_keywords=n_keywords)

    def _analyze_sentiment_wrapper(self, **kwargs):
        """감정 분석 래퍼 함수"""
        return self.analyze_sentiment()

    def _perform_clustering_wrapper(self, n_clusters: int = 5, **kwargs):
        """클러스터링 래퍼 함수"""
        return self.perform_clustering(n_clusters=n_clusters)

    def _model_topics_wrapper(self, n_topics: int = 5, **kwargs):
        """토픽 모델링 래퍼 함수"""
        return self.model_topics(n_topics=n_topics)

    def _analyze_similarity_wrapper(self, **kwargs):
        """유사도 분석 래퍼 함수"""
        return self.analyze_similarity()

    def _run_comprehensive_analysis_wrapper(self, **kwargs):
        """종합 분석 래퍼 함수"""
        return self.run_comprehensive_analysis()

    def collect_news(
        self,
        n_articles: int = 20,
        category: str = "technology",
        include_content: bool = True,
    ) -> Dict[str, Any]:
        """
        뉴스 데이터 수집

        Args:
            n_articles: 수집할 기사 수
            category: 뉴스 카테고리
            include_content: 상세 내용 포함 여부

        Returns:
            수집 결과
        """
        logger.info(f"\n📰 뉴스 수집 시작 ({category} 카테고리, {n_articles}개 기사)")

        try:
            # 뉴스 수집
            df = self.data_collector.collect_and_process(
                category=category,
                num_articles=n_articles,
                include_content=include_content,
            )

            if df.empty:
                return {"success": False, "message": "수집된 뉴스가 없습니다."}

            # 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/news_{category}_{timestamp}.csv"
            df.to_csv(filename, index=False, encoding="utf-8-sig")

            self.current_data = df
            logger.success(f" 뉴스 수집 완료: {len(df)}개 기사")
            logger.info(f" 저장됨: {filename}")

            return {
                "success": True,
                "count": len(df),
                "filename": filename,
                "message": f"{len(df)}개 뉴스 기사가 수집되었습니다.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"뉴스 수집 중 오류: {e}",
            }

    def preprocess_data(self) -> Dict[str, Any]:
        """
        데이터 전처리

        Returns:
            전처리 결과
        """
        if self.current_data is None:
            return {"success": False, "message": "먼저 뉴스 데이터를 수집해주세요."}

        logger.info("\n🔄 데이터 전처리 시작")

        try:
            # 전처리 수행
            processed_df = self.preprocessor.preprocess_dataframe(self.current_data)

            if processed_df.empty:
                return {
                    "success": False,
                    "message": "전처리 후 유효한 데이터가 없습니다.",
                }

            self.processed_data = processed_df

            # 전처리된 데이터 저장
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{self.data_dir}/news_processed_{timestamp}.csv"
            processed_df.to_csv(filename, index=False, encoding="utf-8-sig")

            logger.success(f" 전처리 완료: {len(processed_df)}개 기사")

            return {
                "success": True,
                "count": len(processed_df),
                "filename": filename,
                "message": "데이터 전처리가 완료되었습니다.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"전처리 중 오류: {e}",
            }

    def analyze_keywords(self, n_keywords: int = 20) -> Dict[str, Any]:
        """
        TF-IDF 키워드 분석

        Args:
            n_keywords: 추출할 키워드 수

        Returns:
            키워드 분석 결과
        """
        if not self._ensure_processed_data():
            return {"success": False, "message": "전처리된 데이터가 필요합니다."}

        logger.info(f"\n🔍 TF-IDF 키워드 분석 시작 (상위 {n_keywords}개)")

        try:
            # TF-IDF 분석
            texts = self.processed_data["processed_text"].tolist()
            self.tfidf_matrix = self.text_analyzer.fit_transform(texts)

            # 키워드 추출
            keywords = self.text_analyzer.get_top_keywords_global(n_keywords)

            # 결과 저장
            results = self.text_analyzer.export_analysis_results(
                self.processed_data, output_prefix=f"{self.data_dir}/keywords_analysis"
            )

            logger.success(" 키워드 분석 완료")

            return {
                "success": True,
                "keywords": keywords[:10],  # 상위 10개만 반환
                "total_keywords": len(keywords),
                "files": results,
                "message": f"상위 {n_keywords}개 키워드가 분석되었습니다.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"키워드 분석 중 오류: {e}",
            }

    def analyze_sentiment(self) -> Dict[str, Any]:
        """
        감정 분석

        Returns:
            감정 분석 결과
        """
        if not self._ensure_processed_data():
            return {"success": False, "message": "전처리된 데이터가 필요합니다."}

        logger.info("\n😊 감정 분석 시작")

        try:
            # 감정 분석 수행
            sentiment_df = self.sentiment_analyzer.analyze_dataframe(
                self.processed_data
            )

            # 통계 계산
            stats = self.sentiment_analyzer.get_sentiment_statistics(sentiment_df)

            # 결과 저장
            results = self.sentiment_analyzer.export_results(
                sentiment_df, output_prefix=f"{self.data_dir}/sentiment_analysis"
            )

            logger.success(" 감정 분석 완료")

            return {
                "success": True,
                "statistics": stats,
                "files": results,
                "message": "감정 분석이 완료되었습니다.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"감정 분석 중 오류: {e}",
            }

    def perform_clustering(self, n_clusters: int = 5) -> Dict[str, Any]:
        """
        K-Means 클러스터링

        Args:
            n_clusters: 클러스터 수

        Returns:
            클러스터링 결과
        """
        if not self._ensure_tfidf_matrix():
            return {
                "success": False,
                "message": "TF-IDF 행렬이 필요합니다. 먼저 키워드 분석을 실행해주세요.",
            }

        logger.info(f"\n🎯 K-Means 클러스터링 시작 ({n_clusters}개 클러스터)")

        try:
            # 클러스터링 객체 새로 생성 (수정된 클래스 사용)
            feature_names = self.text_analyzer.feature_names
            self.clustering = NewsClustering(
                n_clusters=n_clusters,
                tfidf_matrix=self.tfidf_matrix,
                feature_names=feature_names,
            )

            # 클러스터링 수행
            labels = self.clustering.fit_predict()
            logger.info(f"클러스터 라벨 샘플: {labels[:10]}")

            # 클러스터 분석
            clustered_df = self.clustering.analyze_clusters(self.processed_data)

            # 결과 저장
            results = self.clustering.export_results(
                clustered_df, output_prefix=f"{self.data_dir}/clustering"
            )

            # 요약 정보
            summary = self.clustering.get_cluster_summary()

            logger.success(" 클러스터링 완료")

            return {
                "success": True,
                "summary": summary,
                "files": results,
                "message": f"{n_clusters}개 클러스터로 분류가 완료되었습니다.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"클러스터링 중 오류: {e}",
            }

    def model_topics(self, n_topics: int = 5) -> Dict[str, Any]:
        """
        LDA 토픽 모델링

        Args:
            n_topics: 토픽 수

        Returns:
            토픽 모델링 결과
        """
        if not self._ensure_processed_data():
            return {"success": False, "message": "전처리된 데이터가 필요합니다."}

        logger.info(f"\n🎭 LDA 토픽 모델링 시작 ({n_topics}개 토픽)")

        try:
            # 토픽 모델링 수행
            self.topic_modeler.n_topics = n_topics
            texts = self.processed_data["processed_text"].tolist()
            doc_topic_probs = self.topic_modeler.fit_transform(texts)
            logger.info(doc_topic_probs[:10])

            # 토픽 분석
            topic_df = self.topic_modeler.analyze_topics(self.processed_data)

            # 결과 저장
            results = self.topic_modeler.export_results(
                topic_df, output_prefix=f"{self.data_dir}/topic_modeling"
            )

            # 모델 정보
            model_info = self.topic_modeler.get_model_info()

            logger.success(" 토픽 모델링 완료")

            return {
                "success": True,
                "model_info": model_info,
                "files": results,
                "message": f"{n_topics}개 토픽이 추출되었습니다.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"토픽 모델링 중 오류: {e}",
            }

    def analyze_similarity(self) -> Dict[str, Any]:
        """
        문서 유사도 분석

        Returns:
            유사도 분석 결과
        """
        if not self._ensure_tfidf_matrix():
            return {
                "success": False,
                "message": "TF-IDF 행렬이 필요합니다. 먼저 키워드 분석을 실행해주세요.",
            }

        logger.info("\n🔗 문서 유사도 분석 시작")

        try:
            # 유사도 분석 수행
            self.similarity_analyzer.set_tfidf_matrix(self.tfidf_matrix)
            similarity_matrix = self.similarity_analyzer.compute_similarity_matrix()
            logger.info(similarity_matrix[:10])

            # 통계 계산
            stats = self.similarity_analyzer.get_similarity_statistics()

            # 유사한 문서 쌍 찾기
            similar_pairs = self.similarity_analyzer.find_most_similar_pairs(
                self.processed_data, 5
            )

            # 결과 저장
            results = self.similarity_analyzer.export_results(
                self.processed_data,
                output_prefix=f"{self.data_dir}/similarity_analysis",
            )

            logger.success(" 유사도 분석 완료")

            return {
                "success": True,
                "statistics": stats,
                "similar_pairs": similar_pairs[:3],  # 상위 3개만 반환
                "files": results,
                "message": "문서 유사도 분석이 완료되었습니다.",
            }

        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"유사도 분석 중 오류: {e}",
            }

    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """
        종합적인 뉴스 분석 실행

        Returns:
            전체 분석 결과
        """
        logger.info("\n🚀 종합 뉴스 분석 시작")

        results = {}

        # 1. 데이터 전처리 (필요한 경우)
        if self.processed_data is None:
            logger.info("📋 1/6: 데이터 전처리")
            results["preprocessing"] = self.preprocess_data()

        # 2. 키워드 분석
        logger.info("📋 2/6: 키워드 분석")
        results["keywords"] = self.analyze_keywords(20)

        # 3. 감정 분석
        logger.info("📋 3/6: 감정 분석")
        results["sentiment"] = self.analyze_sentiment()

        # 4. 클러스터링
        logger.info("📋 4/6: 클러스터링")
        results["clustering"] = self.perform_clustering(5)

        # 5. 토픽 모델링
        logger.info("📋 5/6: 토픽 모델링")
        results["topics"] = self.model_topics(5)

        # 6. 유사도 분석
        logger.info("📋 6/6: 유사도 분석")
        results["similarity"] = self.analyze_similarity()

        # 종합 리포트 생성
        success_count = sum(1 for r in results.values() if r.get("success", False))
        total_count = len(results)

        logger.info(f"\n✅ 종합 분석 완료: {success_count}/{total_count} 성공")

        return {
            "success": success_count > 0,
            "completed_analyses": success_count,
            "total_analyses": total_count,
            "results": results,
            "message": f"종합 분석이 완료되었습니다. ({success_count}/{total_count} 성공)",
        }

    def process_natural_query(self, query: str) -> Dict[str, Any]:
        """
        자연어 쿼리 처리

        Args:
            query: 사용자의 자연어 쿼리

        Returns:
            쿼리 처리 결과
        """
        logger.info(f"\n🤖 자연어 쿼리 처리: '{query}'")

        try:
            result = self.llm_interface.process_query(query)
            return result
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"쿼리 처리 중 오류가 발생했습니다: {e}",
            }

    def get_status(self) -> Dict[str, Any]:
        """
        현재 상태 정보 반환

        Returns:
            시스템 상태 정보
        """
        return {
            "has_raw_data": self.current_data is not None,
            "has_processed_data": self.processed_data is not None,
            "has_tfidf_matrix": self.tfidf_matrix is not None,
            "raw_data_count": (
                len(self.current_data) if self.current_data is not None else 0
            ),
            "processed_data_count": (
                len(self.processed_data) if self.processed_data is not None else 0
            ),
            "data_directory": self.data_dir,
        }

    def _ensure_processed_data(self) -> bool:
        """전처리된 데이터 확인 및 자동 처리"""
        if self.processed_data is None:
            if self.current_data is None:
                return False
            self.preprocess_data()
        return self.processed_data is not None

    def _ensure_tfidf_matrix(self) -> bool:
        """TF-IDF 행렬 확인"""
        if self.tfidf_matrix is None:
            if not self._ensure_processed_data():
                return False
            # 자동으로 키워드 분석 실행
            result = self.analyze_keywords()
            return result.get("success", False)
        return True

    def get_help(self) -> str:
        """도움말 반환"""
        return self.llm_interface.get_help_message()

    def load_data_from_file(self, filename: str) -> Dict[str, Any]:
        """
        파일에서 데이터 로드

        Args:
            filename: CSV 파일 경로

        Returns:
            로드 결과
        """
        try:
            df = pd.read_csv(filename)
            self.current_data = df

            # 전처리된 데이터인지 확인
            if "processed_text" in df.columns:
                self.processed_data = df

            return {
                "success": True,
                "count": len(df),
                "message": f"{len(df)}개 기사를 파일에서 로드했습니다.",
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "message": f"파일 로드 중 오류: {e}",
            }

    def run_full_system_test(
        self,
        test_articles: int = 5,
        test_category: str = "technology",
        save_results: bool = True,
    ) -> Dict[str, Any]:
        """
        모든 분석 모듈이 정상 동작하는지 확인하는 종합 테스트 함수

        Args:
            test_articles: 테스트용 기사 수 (기본값: 5개)
            test_category: 테스트 카테고리 (기본값: technology)
            save_results: 결과 파일 저장 여부

        Returns:
            테스트 결과 상세 보고서
        """
        logger.info("\n🧪 뉴스 에이전트 시스템 종합 테스트 시작")
        logger.info(
            f"   테스트 설정: {test_articles}개 기사, 카테고리: {test_category}"
        )
        logger.info("=" * 60)

        test_results = {
            "overall_success": False,
            "test_timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "test_config": {
                "articles": test_articles,
                "category": test_category,
                "save_results": save_results,
            },
            "module_tests": {},
            "performance_metrics": {},
            "error_log": [],
        }

        start_time = datetime.now()

        try:
            # 1. 데이터 수집 테스트
            logger.info("🔍 1/7: 데이터 수집 모듈 테스트")
            collect_start = datetime.now()
            collect_result = self.collect_news(
                n_articles=test_articles,
                category=test_category,
                include_content=False,  # 빠른 테스트를 위해 상세 내용 제외
            )
            collect_time = (datetime.now() - collect_start).total_seconds()

            test_results["module_tests"]["data_collection"] = {
                "success": collect_result.get("success", False),
                "execution_time_seconds": collect_time,
                "articles_collected": collect_result.get("count", 0),
                "result": collect_result,
            }

            if not collect_result.get("success"):
                test_results["error_log"].append("데이터 수집 실패")
                logger.error(f"   ❌ 데이터 수집 실패: {collect_result.get('message')}")
                return test_results

            logger.success(
                f"   ✅ 데이터 수집 성공 ({collect_result.get('count')}개 기사, {collect_time:.1f}초)"  # noqa: E501
            )

            # 2. 데이터 전처리 테스트
            logger.info("🔄 2/7: 데이터 전처리 모듈 테스트")
            preprocess_start = datetime.now()
            preprocess_result = self.preprocess_data()
            preprocess_time = (datetime.now() - preprocess_start).total_seconds()

            test_results["module_tests"]["preprocessing"] = {
                "success": preprocess_result.get("success", False),
                "execution_time_seconds": preprocess_time,
                "processed_articles": preprocess_result.get("count", 0),
                "result": preprocess_result,
            }

            if not preprocess_result.get("success"):
                test_results["error_log"].append("데이터 전처리 실패")
                logger.error(f"   ❌ 전처리 실패: {preprocess_result.get('message')}")
                return test_results

            logger.success(
                f"   ✅ 데이터 전처리 성공 ({preprocess_result.get('count')}개 기사, {preprocess_time:.1f}초)"  # noqa: E501
            )

            # 3. TF-IDF 키워드 분석 테스트
            logger.info("🔍 3/7: TF-IDF 키워드 분석 테스트")
            keywords_start = datetime.now()
            keywords_result = self.analyze_keywords(n_keywords=10)
            keywords_time = (datetime.now() - keywords_start).total_seconds()

            test_results["module_tests"]["keywords_analysis"] = {
                "success": keywords_result.get("success", False),
                "execution_time_seconds": keywords_time,
                "keywords_found": keywords_result.get("total_keywords", 0),
                "top_keywords": keywords_result.get("keywords", [])[:3],  # 상위 3개만
                "result": keywords_result,
            }

            if not keywords_result.get("success"):
                test_results["error_log"].append("키워드 분석 실패")
                logger.error(
                    f"   ❌ 키워드 분석 실패: {keywords_result.get('message')}"
                )
            else:
                logger.success(
                    f"   ✅ 키워드 분석 성공 ({keywords_result.get('total_keywords')}개 키워드, {keywords_time:.1f}초)"  # noqa: E501
                )

            # 4. 감정 분석 테스트
            logger.info("😊 4/7: 감정 분석 테스트")
            sentiment_start = datetime.now()
            sentiment_result = self.analyze_sentiment()
            sentiment_time = (datetime.now() - sentiment_start).total_seconds()

            test_results["module_tests"]["sentiment_analysis"] = {
                "success": sentiment_result.get("success", False),
                "execution_time_seconds": sentiment_time,
                "sentiment_stats": sentiment_result.get("statistics", {}),
                "result": sentiment_result,
            }

            if not sentiment_result.get("success"):
                test_results["error_log"].append("감정 분석 실패")
                logger.error(f"   ❌ 감정 분석 실패: {sentiment_result.get('message')}")
            else:
                stats = sentiment_result.get("statistics", {})
                logger.success(
                    f"   ✅ 감정 분석 성공 (긍정: {stats.get('positive_ratio', 0):.1%}, {sentiment_time:.1f}초)"  # noqa: E501
                )

            # 5. 클러스터링 테스트
            logger.info("🎯 5/7: K-Means 클러스터링 테스트")
            clustering_start = datetime.now()

            # 클러스터링 객체를 새로 초기화 (수정된 클래스 사용)
            if self.tfidf_matrix is not None:
                feature_names = self.text_analyzer.feature_names
                self.clustering.set_features(self.tfidf_matrix, feature_names)
                self.clustering.n_clusters = min(
                    3, len(self.processed_data) // 2
                )  # 데이터 수에 맞게 조정

                clustering_result = {}
                try:
                    labels = self.clustering.fit_predict()
                    logger.info(f"클러스터 라벨 샘플: {labels[:10]}")
                    clustered_df = self.clustering.analyze_clusters(
                        self.processed_data, top_keywords=5
                    )
                    summary = self.clustering.get_cluster_summary()

                    clustering_result = {
                        "success": True,
                        "summary": summary,
                        "message": f"{self.clustering.n_clusters}개 클러스터로 분류 완료",
                    }

                    if save_results:
                        export_results = self.clustering.export_results(
                            clustered_df,
                            output_prefix=f"{self.data_dir}/test_clustering",
                        )
                        clustering_result["files"] = export_results

                except Exception as e:
                    clustering_result = {
                        "success": False,
                        "error": str(e),
                        "message": f"클러스터링 중 오류: {e}",
                    }
            else:
                clustering_result = {
                    "success": False,
                    "message": "TF-IDF 행렬이 없어 클러스터링을 건너뜁니다.",
                }

            clustering_time = (datetime.now() - clustering_start).total_seconds()

            test_results["module_tests"]["clustering"] = {
                "success": clustering_result.get("success", False),
                "execution_time_seconds": clustering_time,
                "clusters_created": clustering_result.get("summary", {}).get(
                    "n_clusters", 0
                ),
                "result": clustering_result,
            }

            if not clustering_result.get("success"):
                test_results["error_log"].append("클러스터링 실패")
                logger.error(
                    f"   ❌ 클러스터링 실패: {clustering_result.get('message')}"
                )
            else:
                n_clusters = clustering_result.get("summary", {}).get("n_clusters", 0)
                logger.success(
                    f"   ✅ 클러스터링 성공 ({n_clusters}개 클러스터, {clustering_time:.1f}초)"
                )

            # 6. 토픽 모델링 테스트
            logger.info("🎭 6/7: LDA 토픽 모델링 테스트")
            topics_start = datetime.now()
            topics_result = self.model_topics(
                n_topics=min(3, len(self.processed_data) // 3)
            )  # 데이터 수에 맞게 조정
            topics_time = (datetime.now() - topics_start).total_seconds()

            test_results["module_tests"]["topic_modeling"] = {
                "success": topics_result.get("success", False),
                "execution_time_seconds": topics_time,
                "topics_found": topics_result.get("model_info", {}).get("n_topics", 0),
                "perplexity": topics_result.get("model_info", {}).get("perplexity", 0),
                "result": topics_result,
            }

            if not topics_result.get("success"):
                test_results["error_log"].append("토픽 모델링 실패")
                logger.error(f"   ❌ 토픽 모델링 실패: {topics_result.get('message')}")
            else:
                n_topics = topics_result.get("model_info", {}).get("n_topics", 0)
                perplexity = topics_result.get("model_info", {}).get("perplexity", 0)
                logger.success(
                    f"   ✅ 토픽 모델링 성공 ({n_topics}개 토픽, Perplexity: {perplexity:.1f}, {topics_time:.1f}초)"  # noqa: E501
                )

            # 7. 유사도 분석 테스트
            logger.info("🔗 7/7: 문서 유사도 분석 테스트")
            similarity_start = datetime.now()
            similarity_result = self.analyze_similarity()
            similarity_time = (datetime.now() - similarity_start).total_seconds()

            test_results["module_tests"]["similarity_analysis"] = {
                "success": similarity_result.get("success", False),
                "execution_time_seconds": similarity_time,
                "similarity_stats": similarity_result.get("statistics", {}),
                "similar_pairs_found": len(similarity_result.get("similar_pairs", [])),
                "result": similarity_result,
            }

            if not similarity_result.get("success"):
                test_results["error_log"].append("유사도 분석 실패")
                logger.error(
                    f"   ❌ 유사도 분석 실패: {similarity_result.get('message')}"
                )
            else:
                avg_similarity = similarity_result.get("statistics", {}).get(
                    "mean_similarity", 0
                )
                logger.success(
                    f"   ✅ 유사도 분석 성공 (평균 유사도: {avg_similarity:.3f}, {similarity_time:.1f}초)"  # noqa: E501
                )

            # 성과 측정 및 요약
            total_time = (datetime.now() - start_time).total_seconds()
            successful_tests = sum(
                1 for test in test_results["module_tests"].values() if test["success"]
            )
            total_tests = len(test_results["module_tests"])

            test_results["performance_metrics"] = {
                "total_execution_time_seconds": total_time,
                "successful_tests": successful_tests,
                "total_tests": total_tests,
                "success_rate": (
                    successful_tests / total_tests if total_tests > 0 else 0
                ),
                "articles_processed": test_articles,
                "throughput_articles_per_second": (
                    test_articles / total_time if total_time > 0 else 0
                ),
            }

            test_results["overall_success"] = (
                successful_tests >= 6
            )  # 7개 중 6개 이상 성공

            # 결과 요약 출력
            logger.info("\n" + "=" * 60)
            logger.info("🏁 뉴스 에이전트 시스템 종합 테스트 완료")
            logger.info("=" * 60)

            if test_results["overall_success"]:
                logger.success(
                    f"✅ 전체 테스트 성공! ({successful_tests}/{total_tests} 모듈 통과)"
                )
            else:
                logger.warning(
                    f"⚠️ 일부 테스트 실패 ({successful_tests}/{total_tests} 모듈 통과)"
                )

            logger.info("📊 성능 지표:")
            logger.info(f"   • 총 실행 시간: {total_time:.1f}초")
            logger.info(f"   • 처리 속도: {test_articles / total_time:.2f} 기사/초")
            logger.info(f"   • 성공률: {successful_tests / total_tests:.1%}")

            if test_results["error_log"]:
                logger.info(f"❌ 실패한 모듈: {', '.join(test_results['error_log'])}")

            # 테스트 결과 파일 저장
            if save_results:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_filename = f"{self.data_dir}/system_test_result_{timestamp}.json"

                import json

                with open(result_filename, "w", encoding="utf-8") as f:
                    json.dump(
                        test_results, f, ensure_ascii=False, indent=2, default=str
                    )

                logger.info(f"📁 테스트 결과가 저장됨: {result_filename}")

            return test_results

        except Exception as e:
            logger.error(f"❌ 시스템 테스트 중 예외 발생: {e}")
            test_results["error_log"].append(f"시스템 테스트 예외: {str(e)}")
            test_results["overall_success"] = False
            return test_results

    def quick_health_check(self) -> Dict[str, Any]:
        """
        간단한 시스템 상태 확인 (API 연결, 모듈 로드 등)

        Returns:
            상태 확인 결과
        """
        logger.info("\n🏥 시스템 헬스 체크 시작")

        health_status = {
            "overall_healthy": True,
            "checks": {},
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

        # 1. 모듈 로드 확인
        try:
            modules_check = {
                "data_collector": self.data_collector is not None,
                "preprocessor": self.preprocessor is not None,
                "text_analyzer": self.text_analyzer is not None,
                "clustering": self.clustering is not None,
                "topic_modeler": self.topic_modeler is not None,
                "sentiment_analyzer": self.sentiment_analyzer is not None,
                "similarity_analyzer": self.similarity_analyzer is not None,
                "llm_interface": self.llm_interface is not None,
            }

            health_status["checks"]["modules_loaded"] = {
                "status": all(modules_check.values()),
                "details": modules_check,
            }

            if not all(modules_check.values()):
                health_status["overall_healthy"] = False
                logger.warning("   ⚠️ 일부 모듈이 로드되지 않음")
            else:
                logger.success("   ✅ 모든 모듈 정상 로드")

        except Exception as e:
            health_status["checks"]["modules_loaded"] = {
                "status": False,
                "error": str(e),
            }
            health_status["overall_healthy"] = False
            logger.error(f"   ❌ 모듈 확인 실패: {e}")

        # 2. API 키 확인
        try:
            api_check = self.api_key is not None and len(self.api_key) > 0
            health_status["checks"]["api_key"] = {
                "status": api_check,
                "has_key": api_check,
            }

            if api_check:
                logger.success("   ✅ API 키 설정됨")
            else:
                logger.warning("   ⚠️ API 키가 설정되지 않음")

        except Exception as e:
            health_status["checks"]["api_key"] = {"status": False, "error": str(e)}
            logger.error(f"   ❌ API 키 확인 실패: {e}")

        # 3. 데이터 디렉토리 확인
        try:
            import os

            dir_exists = os.path.exists(self.data_dir)
            dir_writable = os.access(self.data_dir, os.W_OK) if dir_exists else False

            health_status["checks"]["data_directory"] = {
                "status": dir_exists and dir_writable,
                "exists": dir_exists,
                "writable": dir_writable,
                "path": self.data_dir,
            }

            if dir_exists and dir_writable:
                logger.success(f"   ✅ 데이터 디렉토리 정상 ({self.data_dir})")
            else:
                logger.warning(
                    f"   ⚠️ 데이터 디렉토리 문제 (존재: {dir_exists}, 쓰기가능: {dir_writable})"
                )

        except Exception as e:
            health_status["checks"]["data_directory"] = {
                "status": False,
                "error": str(e),
            }
            health_status["overall_healthy"] = False
            logger.error(f"   ❌ 디렉토리 확인 실패: {e}")

        # 4. 데이터 상태 확인
        status = self.get_status()
        health_status["checks"]["data_status"] = {"status": True, "details": status}

        logger.info(
            f"   📊 데이터 상태: 원본 {status['raw_data_count']}개, 전처리 {status['processed_data_count']}개"  # noqa: E501
        )

        # 전체 결과
        if health_status["overall_healthy"]:
            logger.success("🎉 시스템 헬스 체크 완료 - 모든 상태 정상")
        else:
            logger.warning("⚠️ 시스템 헬스 체크 완료 - 일부 문제 발견")

        return health_status


if __name__ == "__main__":
    logger.info("🤖 뉴스 에이전트 시스템 테스트")

    # 에이전트 초기화
    agent = NewsAgent()

    # 1. 헬스 체크 먼저 실행
    logger.info("\n" + "=" * 50)
    logger.info("🏥 시스템 헬스 체크 실행")
    logger.info("=" * 50)

    health_result = agent.quick_health_check()

    if health_result["overall_healthy"]:
        logger.success("✅ 시스템 상태 정상 - 종합 테스트 진행")

        # 2. 종합 테스트 실행
        logger.info("\n" + "=" * 50)
        logger.info("🧪 종합 시스템 테스트 실행")
        logger.info("=" * 50)

        test_result = agent.run_full_system_test(
            test_articles=3,  # 빠른 테스트를 위해 3개 기사만
            test_category="technology",
            save_results=True,
        )

        # 3. 테스트 결과 요약
        if test_result["overall_success"]:
            logger.success("\n🎉 뉴스 에이전트 시스템이 정상적으로 작동합니다!")
            logger.info("📊 모든 핵심 기능이 테스트를 통과했습니다.")
        else:
            logger.warning("\n⚠️ 일부 기능에서 문제가 발견되었습니다.")
            logger.info("🔧 로그를 확인하고 문제를 해결해주세요.")

        # 4. 사용법 안내
        logger.info("\n" + "=" * 50)
        logger.info("📖 뉴스 에이전트 사용법")
        logger.info("=" * 50)

        sample_queries = [
            "최신 IT 뉴스 10개 수집해줘",
            "중요한 키워드 15개 분석해줘",
            "감정 분석 실행해줘",
            "비슷한 뉴스끼리 5개 그룹으로 분류해줘",
            "모든 분석을 실행해줘",
        ]

        logger.info("💬 지원하는 자연어 쿼리 예시:")
        for i, query in enumerate(sample_queries, 1):
            logger.info(f"   {i}. '{query}'")

        logger.info("\n🚀 실제 사용 예시:")
        logger.info("```python")
        logger.info("from src.news_agent.news_agent import NewsAgent")
        logger.info("")
        logger.info("# 에이전트 초기화")
        logger.info("agent = NewsAgent()")
        logger.info("")
        logger.info("# 자연어 쿼리로 분석 실행")
        logger.info(
            "result = agent.process_natural_query('최신 IT 뉴스 20개 수집해줘')"
        )
        logger.info("result = agent.process_natural_query('모든 분석 실행해줘')")
        logger.info("")
        logger.info("# 개별 함수 직접 호출")
        logger.info("agent.collect_news(n_articles=20, category='technology')")
        logger.info("agent.run_comprehensive_analysis()")
        logger.info("```")

    else:
        logger.error("❌ 시스템 헬스 체크 실패 - 설정을 확인해주세요")

        failed_checks = [
            name
            for name, check in health_result.get("checks", {}).items()
            if not check.get("status", False)
        ]

        if failed_checks:
            logger.error(f"실패한 체크 항목: {', '.join(failed_checks)}")

        logger.info("\n🔧 문제 해결 방법:")
        logger.info("1. API 키가 올바르게 설정되었는지 확인")
        logger.info("2. 필요한 Python 패키지가 모두 설치되었는지 확인")
        logger.info("3. 데이터 디렉토리 권한 확인")
        logger.info("4. 인터넷 연결 상태 확인")
