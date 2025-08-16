"""
LLM 인터페이스 모듈 - 사용자 쿼리를 분석하고 적절한 분석 함수를 호출
"""

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from loguru import logger


@dataclass
class QueryIntent:
    """쿼리 의도 분석 결과"""

    intent_type: str
    confidence: float
    parameters: Dict[str, Any]
    suggested_function: str
    description: str


class LLMInterface:
    """LLM 스타일의 자연어 쿼리 인터페이스"""

    def __init__(self):
        """인터페이스 초기화"""
        self.intent_patterns = self._initialize_intent_patterns()
        self.function_registry = {}

    def _initialize_intent_patterns(self) -> Dict[str, Dict]:
        """의도 분석을 위한 패턴 정의"""
        return {
            "keyword_analysis": {
                "patterns": [
                    r"키워드.*(\d+)개?",
                    r"인기.*단어",
                    r"tf.?idf",
                    r"중요.*단어",
                    r"주요.*키워드",
                    r"단어.*분석",
                    r"키워드.*추출",
                ],
                "function": "analyze_keywords",
                "description": "TF-IDF 기반 키워드 분석 수행",
            },
            "sentiment_analysis": {
                "patterns": [
                    r"감정.*분석",
                    r"긍정.*부정",
                    r"감성.*분석",
                    r"sentiment",
                    r"기분.*분석",
                    r"감정.*점수",
                ],
                "function": "analyze_sentiment",
                "description": "뉴스 기사의 감정 분석 수행",
            },
            "clustering": {
                "patterns": [
                    r"클러스터.*(\d+)개?",
                    r"그룹.*(\d+)개?",
                    r"분류.*(\d+)개?",
                    r"clustering",
                    r"묶기",
                    r"k.?means",
                ],
                "function": "perform_clustering",
                "description": "K-Means 클러스터링을 통한 뉴스 분류",
            },
            "topic_modeling": {
                "patterns": [
                    r"토픽.*모델링",
                    r"주제.*분석",
                    r"topic.*model",
                    r"lda",
                    r"주제.*추출",
                    r"토픽.*(\d+)개?",
                ],
                "function": "model_topics",
                "description": "LDA 토픽 모델링을 통한 주제 분석",
            },
            "similarity_analysis": {
                "patterns": [
                    r"유사.*문서",
                    r"비슷한.*기사",
                    r"유사도.*분석",
                    r"similarity",
                    r"similar.*documents",
                    r"문서.*유사성",
                ],
                "function": "analyze_similarity",
                "description": "문서 간 유사도 분석",
            },
            "data_collection": {
                "patterns": [
                    r"뉴스.*수집",
                    r"데이터.*가져오",
                    r"기사.*(\d+)개?",
                    r"최신.*뉴스",
                    r"collect.*news",
                    r"fetch.*articles",
                ],
                "function": "collect_news",
                "description": "뉴스 API를 통한 데이터 수집",
            },
            "comprehensive_analysis": {
                "patterns": [
                    r"전체.*분석",
                    r"모든.*분석",
                    r"종합.*분석",
                    r"complete.*analysis",
                    r"전반적.*분석",
                ],
                "function": "run_comprehensive_analysis",
                "description": "모든 분석 기능을 순차적으로 실행",
            },
        }

    def register_function(self, name: str, function: Callable) -> None:
        """
        분석 함수 등록

        Args:
            name: 함수 이름
            function: 실행할 함수
        """
        self.function_registry[name] = function

    def analyze_query_intent(self, query: str) -> QueryIntent:
        """
        자연어 쿼리의 의도 분석

        Args:
            query: 사용자 입력 쿼리

        Returns:
            분석된 의도 정보
        """
        query_lower = query.lower()
        best_match = None
        best_confidence = 0.0
        best_parameters = {}

        for intent_type, intent_config in self.intent_patterns.items():
            confidence = 0.0
            parameters = {}

            # 패턴 매칭
            for pattern in intent_config["patterns"]:
                match = re.search(pattern, query_lower)
                if match:
                    confidence += 0.3

                    # 숫자 파라미터 추출
                    if match.groups():
                        try:
                            number = int(match.group(1))
                            if "키워드" in pattern or "단어" in pattern:
                                parameters["n_keywords"] = number
                            elif (
                                "클러스터" in pattern
                                or "그룹" in pattern
                                or "분류" in pattern
                            ):
                                parameters["n_clusters"] = number
                            elif "토픽" in pattern:
                                parameters["n_topics"] = number
                            elif "기사" in pattern:
                                parameters["n_articles"] = number
                        except (ValueError, IndexError):
                            pass

            # 키워드 기반 신뢰도 증가
            keywords = {
                "keyword_analysis": ["키워드", "단어", "중요", "tfidf"],
                "sentiment_analysis": ["감정", "긍정", "부정", "감성"],
                "clustering": ["클러스터", "그룹", "분류", "묶"],
                "topic_modeling": ["토픽", "주제", "lda"],
                "similarity_analysis": ["유사", "비슷", "similarity"],
                "data_collection": ["수집", "가져오", "최신", "뉴스"],
                "comprehensive_analysis": ["전체", "모든", "종합", "전반적"],
            }

            for keyword in keywords.get(intent_type, []):
                if keyword in query_lower:
                    confidence += 0.2

            # 최고 점수 업데이트
            if confidence > best_confidence:
                best_confidence = confidence
                best_match = intent_type
                best_parameters = parameters

        # 기본값 설정
        if best_match is None:
            best_match = "keyword_analysis"
            best_confidence = 0.1

        return QueryIntent(
            intent_type=best_match,
            confidence=best_confidence,
            parameters=best_parameters,
            suggested_function=self.intent_patterns[best_match]["function"],
            description=self.intent_patterns[best_match]["description"],
        )

    def process_query(self, query: str) -> Dict[str, Any]:
        """
        쿼리 처리 및 분석 실행

        Args:
            query: 사용자 쿼리

        Returns:
            분석 결과
        """
        logger.info(f"\n🤖 사용자 쿼리: '{query}'")

        # 의도 분석
        intent = self.analyze_query_intent(query)

        logger.info(f" 분석된 의도: {intent.intent_type}")
        logger.info(f"   신뢰도: {intent.confidence:.2f}")
        logger.info(f"   설명: {intent.description}")
        if intent.parameters:
            logger.info(f"   파라미터: {intent.parameters}")

        # 함수 실행
        if intent.suggested_function in self.function_registry:
            try:
                function = self.function_registry[intent.suggested_function]

                # 파라미터 전달하여 함수 실행
                if intent.parameters:
                    result = function(**intent.parameters)
                else:
                    result = function()

                return {
                    "success": True,
                    "intent": intent,
                    "result": result,
                    "message": f"{intent.description}이 완료되었습니다.",
                }

            except Exception as e:
                return {
                    "success": False,
                    "intent": intent,
                    "error": str(e),
                    "message": f"분석 중 오류가 발생했습니다: {e}",
                }
        else:
            return {
                "success": False,
                "intent": intent,
                "error": "Function not found",
                "message": f"'{intent.suggested_function}' 함수가 등록되지 않았습니다.",
            }

    def get_help_message(self) -> str:
        """도움말 메시지 반환"""
        help_text = """
🤖 뉴스 분석 AI 도우미 사용법

📝 지원하는 쿼리 예시:

1. 키워드 분석:
   • "최근 IT 뉴스 인기 키워드 10개"
   • "중요한 단어들을 분석해줘"
   • "TF-IDF 키워드 추출"

2. 감정 분석:
   • "삼성전자 관련 뉴스 감정 분석"
   • "긍정적인 뉴스와 부정적인 뉴스 비율"
   • "감성 분석 결과 보여줘"

3. 클러스터링:
   • "AI 뉴스 5개 그룹으로 클러스터링"
   • "비슷한 뉴스끼리 3개 그룹으로 분류"
   • "K-means로 뉴스 묶어줘"

4. 토픽 모델링:
   • "뉴스의 주요 토픽 3개 분석"
   • "LDA 토픽 모델링 실행"
   • "주제별로 뉴스 분석"

5. 유사도 분석:
   • "비슷한 문서들 찾아줘"
   • "문서 유사도 분석"
   • "유사한 뉴스 기사 검색"

6. 데이터 수집:
   • "최신 IT 뉴스 50개 수집"
   • "뉴스 데이터 가져와줘"
   • "기사 100개 수집해줘"

7. 종합 분석:
   • "모든 분석 실행해줘"
   • "전체적인 뉴스 분석"
   • "종합적인 데이터 분석"

💡 팁: 자연스러운 한국어로 요청하시면 됩니다!
        """
        return help_text

    def suggest_queries(self, category: Optional[str] = None) -> List[str]:
        """
        쿼리 제안

        Args:
            category: 특정 카테고리 (선택적)

        Returns:
            제안 쿼리 리스트
        """
        suggestions = {
            "keyword_analysis": [
                "최근 1주일 IT 뉴스 인기 키워드 10개",
                "중요한 단어 20개 추출해줘",
                "TF-IDF로 핵심 키워드 분석",
            ],
            "sentiment_analysis": [
                "삼성전자 관련 뉴스 감정 분석",
                "전체 뉴스의 긍정/부정 비율 분석",
                "가장 긍정적인 뉴스와 부정적인 뉴스",
            ],
            "clustering": [
                "AI 뉴스 5개 그룹으로 클러스터링",
                "비슷한 뉴스끼리 3개 그룹으로 분류",
                "경제 뉴스를 4개 클러스터로 나눠줘",
            ],
            "topic_modeling": [
                "뉴스의 주요 토픽 5개 분석",
                "LDA로 주제 모델링 실행",
                "IT 뉴스의 주요 주제들 찾아줘",
            ],
            "similarity_analysis": [
                "비슷한 뉴스 기사들 찾아줘",
                "문서 유사도 분석 실행",
                "중복되는 뉴스들 검색",
            ],
        }

        if category and category in suggestions:
            return suggestions[category]

        # 모든 제안 반환
        all_suggestions = []
        for category_suggestions in suggestions.values():
            all_suggestions.extend(category_suggestions)
        return all_suggestions

    def explain_intent(self, query: str) -> str:
        """
        쿼리 의도 설명

        Args:
            query: 분석할 쿼리

        Returns:
            의도 설명 텍스트
        """
        intent = self.analyze_query_intent(query)

        explanation = f"""
🔍 쿼리 분석 결과

📝 입력 쿼리: "{query}"

🎯 분석된 의도:
   • 유형: {intent.intent_type}
   • 신뢰도: {intent.confidence:.2f}
   • 실행 함수: {intent.suggested_function}
   • 설명: {intent.description}

⚙️ 추출된 파라미터:
"""

        if intent.parameters:
            for key, value in intent.parameters.items():
                explanation += f"   • {key}: {value}\n"
        else:
            explanation += "   • 파라미터 없음\n"

        explanation += f"""
💡 제안 사항:
   이 쿼리는 '{intent.description}' 기능을 실행합니다.
   신뢰도가 낮다면 더 구체적인 표현을 사용해보세요.
        """

        return explanation


if __name__ == "__main__":
    logger.info("🤖 LLM 인터페이스 모듈 테스트")

    # 인터페이스 초기화
    llm_interface = LLMInterface()

    # 더미 함수들 등록
    def dummy_analyze_keywords(**kwargs):
        return f"키워드 분석 실행됨 - 파라미터: {kwargs}"

    def dummy_analyze_sentiment(**kwargs):
        return f"감정 분석 실행됨 - 파라미터: {kwargs}"

    def dummy_perform_clustering(**kwargs):
        return f"클러스터링 실행됨 - 파라미터: {kwargs}"

    llm_interface.register_function("analyze_keywords", dummy_analyze_keywords)
    llm_interface.register_function("analyze_sentiment", dummy_analyze_sentiment)
    llm_interface.register_function("perform_clustering", dummy_perform_clustering)

    # 테스트 쿼리들
    test_queries = [
        "최근 IT 뉴스 인기 키워드 10개",
        "삼성전자 관련 뉴스 감정 분석",
        "AI 뉴스 5개 그룹으로 클러스터링",
        "전체적인 뉴스 분석해줘",
    ]

    for query in test_queries:
        logger.info(f"\n{'=' * 50}")
        result = llm_interface.process_query(query)
        logger.info(f"결과: {result}")

    # 도움말 출력
    logger.info(f"\n도움말:\n{llm_interface.get_help_message()}")

    logger.info("\n📁 실제 분석 함수들과 연결하면 완전한 뉴스 분석 AI가 됩니다.")
