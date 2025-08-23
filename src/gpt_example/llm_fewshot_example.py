import os
from typing import Dict, List

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# API 키 설정 (환경변수 사용 권장)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


class FewShotLearning:
    """퓨샷 러닝 시스템 구현"""

    def __init__(self):
        self.examples = []

    def add_examples(self, examples: List[Dict[str, str]]):
        """학습 예시 추가"""
        self.examples = examples
        print(f"📝 {len(examples)}개의 예시가 추가되었습니다.")

    def generate_few_shot_prompt(self, query: str) -> str:
        """퓨샷 프롬프트 생성"""
        prompt = "다음은 질문과 답변의 예시들입니다:\n\n"

        # 예시들을 프롬프트에 추가
        for i, example in enumerate(self.examples, 1):
            prompt += f"예시 {i}:\n"
            prompt += f"질문: {example['question']}\n"
            prompt += f"답변: {example['answer']}\n\n"

        # 실제 질문 추가
        prompt += "이제 다음 질문에 위의 예시와 같은 방식으로 답변해주세요:\n"
        prompt += f"질문: {query}\n답변:"

        return prompt

    def generate_answer(self, query: str) -> str:
        """퓨샷 러닝으로 답변 생성"""
        if not self.examples:
            return "❌ 학습 예시가 없습니다. 먼저 예시를 추가해주세요."

        prompt = self.generate_few_shot_prompt(query)

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {
                        "role": "system",
                        "content": "당신은 주어진 예시들의 패턴을 학습하여 일관된 방식으로 답변하는 AI입니다.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=300,
                temperature=0.2,  # 일관성을 위해 낮은 temperature 사용
            )

            return response.choices[0].message.content.strip()

        except Exception as e:
            return f"❌ 오류가 발생했습니다: {str(e)}"


def func1():
    fewshot_system = FewShotLearning()

    # 감정 분석 예시들 추가
    sentiment_examples = [
        {
            "question": "이 영화 정말 재미있었어요! 강력 추천합니다.",
            "answer": "긍정 (95% 확신)",
        },
        {
            "question": "서비스가 별로였고 음식도 맛없었습니다.",
            "answer": "부정 (90% 확신)",
        },
        {
            "question": "그냥 보통이었어요. 나쁘지도 좋지도 않고.",
            "answer": "중립 (85% 확신)",
        },
        {
            "question": "최고의 경험이었습니다! 다시 가고 싶어요!",
            "answer": "긍정 (98% 확신)",
        },
    ]

    fewshot_system.add_examples(sentiment_examples)

    # 퓨샷 러닝 테스트 질문들
    fewshot_questions = [
        "이 제품은 정말 훌륭합니다. 모든 기능이 완벽해요!",
        "배송이 너무 느리고 포장도 엉망이었습니다.",
        "가격 대비 괜찮은 것 같아요. 특별하지는 않지만.",
        "최악의 구매였습니다. 돈 낭비했네요.",
    ]

    for question in fewshot_questions:
        print(f"\n🎯 분석할 텍스트: {question}")
        answer = fewshot_system.generate_answer(question)
        print(f"💭 퓨샷 러닝 결과: {answer}")


def func2():
    email_fewshot = FewShotLearning()

    email_examples = [
        {
            "question": "안녕하세요, 제품 교환 요청드립니다. 주문번호는 12345입니다.",
            "answer": "카테고리: 고객서비스 | 우선순위: 높음 | 담당부서: CS팀",
        },
        {
            "question": "회사 소개서와 제품 카탈로그를 보내주실 수 있나요?",
            "answer": "카테고리: 영업문의 | 우선순위: 중간 | 담당부서: 영업팀",
        },
        {
            "question": "시스템 로그인이 안 되고 있습니다. 빠른 지원 부탁드립니다.",
            "answer": "카테고리: 기술지원 | 우선순위: 높음 | 담당부서: IT팀",
        },
    ]

    email_fewshot.add_examples(email_examples)

    test_emails = [
        "결제가 제대로 처리되지 않았습니다. 확인 부탁드립니다.",
        "파트너십 제안서를 검토해보시고 연락 주시기 바랍니다.",
        "앱이 계속 크래시됩니다. 업데이트 예정이 있나요?",
    ]

    for email in test_emails:
        print(f"\n📧 이메일: {email}")
        classification = email_fewshot.generate_answer(email)
        print(f"📋 분류 결과: {classification}")


if __name__ == "__main__":
    func1()
    func2()
