"""
Simple OpenAI Agent - 기본 학습용
2주차 LLM 연동 및 프롬프트 엔지니어링 실습

목표:
1. OpenAI API 기본 연동
2. 간단한 프롬프트 설계
3. 사용자 데이터를 LLM에 전달하여 이해하기 쉬운 답변 생성
"""

import os

from openai import OpenAI


class SimpleAgent:
    def __init__(self, api_key: str = None):
        """
        Simple Agent 초기화

        Args:
            api_key: OpenAI API 키
        """
        # API 키 설정 (환경변수 또는 직접 입력)
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            print("⚠️  API 키를 설정해주세요:")
            print("   방법1: 환경변수 설정 - export OPENAI_API_KEY='your-key'")
            print("   방법2: 코드에서 직접 - SimpleAgent(api_key='your-key')")
            return

        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=self.api_key)

        print("🤖 Simple Agent가 준비되었습니다!")
        print("💬 질문하거나 텍스트 분석을 요청해보세요.")

    def chat(self, user_input: str) -> str:
        """
        기본 채팅 기능

        Args:
            user_input: 사용자 입력

        Returns:
            AI 응답
        """
        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 도움이 되는 AI 어시스턴트입니다. 친근하고 이해하기 쉽게 답변하세요.",
                    },
                    {"role": "user", "content": user_input},
                ],
                temperature=0.7,
                max_tokens=300,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"❌ 오류가 발생했습니다: {str(e)}"

    def analyze_text(self, text: str, analysis_type: str = "summary") -> str:
        """
        텍스트 분석 기능 (1주차 데이터 활용 예시)

        Args:
            text: 분석할 텍스트
            analysis_type: 분석 유형 (summary, sentiment, keywords)

        Returns:
            분석 결과
        """
        # 분석 유형별 시스템 프롬프트
        prompts = {
            "summary": "다음 텍스트를 3줄로 요약해주세요. 핵심 내용만 간단하고 이해하기 쉽게 설명하세요.",
            "sentiment": "다음 텍스트의 감정을 분석해주세요. 긍정적/부정적/중립적 중 하나로 판단하고 그 이유를 설명하세요.",
            "keywords": "다음 텍스트에서 가장 중요한 키워드 5개를 추출하고, 각각이 왜 중요한지 설명해주세요.",
        }

        system_prompt = prompts.get(analysis_type, prompts["summary"])

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": text},
                ],
                temperature=0.3,  # 분석은 일관성을 위해 낮게 설정
                max_tokens=400,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"❌ 분석 중 오류가 발생했습니다: {str(e)}"

    def process_data(self, data_list: list, task: str = "explain") -> str:
        """
        1주차에서 수집한 데이터를 처리하는 함수

        Args:
            data_list: 처리할 데이터 리스트
            task: 수행할 작업 (explain, compare, insights)

        Returns:
            처리 결과
        """
        # 데이터를 문자열로 변환
        data_str = "\n".join([f"- {item}" for item in data_list])

        # 작업별 프롬프트
        task_prompts = {
            "explain": f"""
            다음 데이터를 분석하여 일반인이 이해하기 쉽게 설명해주세요:

            {data_str}

            데이터의 의미와 특징을 친근한 언어로 설명하고,
            중요한 패턴이나 인사이트가 있다면 알려주세요.
            """,
            "compare": f"""
            다음 데이터들을 비교 분석해주세요:

            {data_str}

            각 항목의 차이점과 공통점을 찾아서
            쉽게 이해할 수 있도록 설명해주세요.
            """,
            "insights": f"""
            다음 데이터에서 흥미로운 인사이트를 찾아주세요:

            {data_str}

            숨겨진 패턴, 의외의 발견, 또는 주목할 만한
            특징들을 발견하여 설명해주세요.
            """,
        }

        prompt = task_prompts.get(task, task_prompts["explain"])

        try:
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 데이터 분석 전문가입니다. 복잡한 데이터를 누구나 이해할 수 있게 설명하는 것이 특기입니다.",  # noqa E501
                    },
                    {"role": "user", "content": prompt},
                ],
                temperature=0.5,
                max_tokens=500,
            )

            return response.choices[0].message.content

        except Exception as e:
            return f"❌ 데이터 처리 중 오류가 발생했습니다: {str(e)}"

    def get_sample_data(self):
        """
        1주차에서 수집했다고 가정하는 샘플 데이터
        """
        return {
            "웹사이트_방문자": [1200, 1350, 1100, 1500, 1800, 1650, 1900],
            "판매_데이터": ["노트북 15대", "마우스 45개", "키보드 32개", "모니터 8대"],
            "고객_리뷰": [
                "제품이 정말 좋아요! 추천합니다.",
                "배송이 좀 늦었지만 품질은 만족해요.",
                "가격 대비 성능이 뛰어나네요.",
                "다음에도 이용하고 싶습니다.",
            ],
            "서버_로그": [
                "INFO: 서버 시작",
                "WARNING: 메모리 사용량 85%",
                "ERROR: 데이터베이스 연결 실패",
                "INFO: 백업 완료",
            ],
        }


# 사용 예시
def main():
    """
    Simple Agent 사용 예시
    """
    print("=" * 50)
    print("🚀 Simple OpenAI Agent 실습")
    print("=" * 50)

    # API 키 설정 방법 안내
    print("\n📋 시작하기 전에:")
    print("1. OpenAI API 키를 발급받으세요 (https://platform.openai.com)")
    print("2. 환경변수로 설정: export OPENAI_API_KEY='your-key'")
    print("3. 또는 코드에서 직접: agent = SimpleAgent(api_key='your-key')")

    # Agent 초기화 (실제 사용시에는 API 키 필요)
    print("\n🤖 Agent 초기화...")
    # agent = SimpleAgent()  # 실제 사용

    # 샘플 데이터 생성
    sample_data = {
        "웹사이트_방문자": [1200, 1350, 1100, 1500, 1800, 1650, 1900],
        "고객_리뷰": [
            "제품이 정말 좋아요! 추천합니다.",
            "배송이 좀 늦었지만 품질은 만족해요.",
            "가격 대비 성능이 뛰어나네요.",
        ],
    }

    print("\n📊 1주차에서 수집한 샘플 데이터:")
    for key, value in sample_data.items():
        print(f"   {key}: {value}")

    print("\n💡 사용 가능한 기능들:")
    print("   1. agent.chat('안녕하세요!')  # 기본 채팅")
    print("   2. agent.analyze_text(text, 'summary')  # 텍스트 분석")
    print("   3. agent.process_data(data_list, 'explain')  # 데이터 처리")

    print("\n🔧 실습 과제:")
    print("   1. Git 브랜치 생성: git checkout -b feature/llm-integration")
    print("   2. OpenAI API 키 발급 및 설정")
    print("   3. openai 라이브러리 설치: pip install openai")
    print("   4. 이 코드를 실행하여 LLM 연동 테스트")

    # 테스트 실행 예시 (API 키가 있을 때)
    print("\n🧪 테스트 예시:")
    print("# agent = SimpleAgent(api_key='your-key')")
    print("# result = agent.process_data(['데이터1', '데이터2'], 'explain')")
    print("# print(result)")


if __name__ == "__main__":
    main()
