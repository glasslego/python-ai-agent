import json
import os
from datetime import datetime

import openai
from dotenv import load_dotenv

# 환경 변수 로드
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")


# =============================================================================
# 1. OpenAI Function Call 방식 (순수 OpenAI API)
# =============================================================================


class OpenAIFunctionCallExample:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(api_key=api_key)

    # 실제 함수들 정의
    def get_current_time(self) -> str:
        """현재 시각 반환"""
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def get_weather(self, city: str) -> str:
        """간단한 날씨 정보 (Mock)"""
        weather_data = {"서울": "맑음, 22°C", "부산": "흐림, 18°C", "대구": "비, 15°C"}
        return weather_data.get(city, f"{city}의 날씨 정보를 찾을 수 없습니다")

    def calculate(self, expression: str) -> str:
        """간단한 계산"""
        try:
            result = eval(expression)  # 실제로는 더 안전한 방법 사용 권장
            return f"{expression} = {result}"
        except Exception:
            return "계산할 수 없는 식입니다"

    # OpenAI가 이해할 수 있는 함수 스키마 정의
    @property
    def function_schemas(self):
        return [
            {
                "type": "function",
                "function": {
                    "name": "get_current_time",
                    "description": "현재 시각을 조회합니다",
                    "parameters": {"type": "object", "properties": {}, "required": []},
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "get_weather",
                    "description": "특정 도시의 날씨를 조회합니다",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "city": {
                                "type": "string",
                                "description": "날씨를 조회할 도시명",
                            }
                        },
                        "required": ["city"],
                    },
                },
            },
            {
                "type": "function",
                "function": {
                    "name": "calculate",
                    "description": "수학 계산을 수행합니다",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "expression": {
                                "type": "string",
                                "description": "계산할 수식 (예: 2+3, 10*5)",
                            }
                        },
                        "required": ["expression"],
                    },
                },
            },
        ]

    # 함수 이름과 실제 함수 매핑
    @property
    def available_functions(self):
        return {
            "get_current_time": self.get_current_time,
            "get_weather": self.get_weather,
            "calculate": self.calculate,
        }

    def chat_with_functions(self, user_message: str) -> str:
        """OpenAI Function Call을 사용한 채팅"""
        messages = [
            {
                "role": "system",
                "content": "당신은 도움이 되는 AI 어시스턴트입니다. 필요시 제공된 함수들을 사용하세요.",
            },
            {"role": "user", "content": user_message},
        ]

        print(f"👤 사용자: {user_message}")

        # 1단계: LLM에게 메시지와 사용 가능한 함수들 전달
        response = self.client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=messages,
            tools=self.function_schemas,
            tool_choice="auto",
        )

        response_message = response.choices[0].message

        # 2단계: 함수 호출이 필요한지 확인
        if response_message.tool_calls:
            print("🔧 OpenAI Function Call 실행:")

            # 함수 호출 결과를 저장할 메시지 추가
            messages.append(response_message)

            # 3단계: 각 함수 호출 실행
            for tool_call in response_message.tool_calls:
                function_name = tool_call.function.name
                function_args = json.loads(tool_call.function.arguments)

                print(f"  📞 함수 호출: {function_name}({function_args})")

                # 실제 함수 실행
                if function_name in self.available_functions:
                    function_to_call = self.available_functions[function_name]
                    function_result = function_to_call(**function_args)

                    print(f"  ✅ 결과: {function_result}")

                    # 함수 실행 결과를 메시지에 추가
                    messages.append(
                        {
                            "tool_call_id": tool_call.id,
                            "role": "tool",
                            "name": function_name,
                            "content": str(function_result),
                        }
                    )

            # 4단계: 함수 결과를 바탕으로 최종 응답 생성
            final_response = self.client.chat.completions.create(
                model="gpt-3.5-turbo", messages=messages
            )

            final_answer = final_response.choices[0].message.content
            print(f"🤖 최종 답변: {final_answer}")
            return final_answer

        else:
            # 함수 호출이 필요없는 경우
            answer = response_message.content
            print(f"🤖 답변: {answer}")
            return answer


if __name__ == "__main__":
    test_questions = [
        "지금 몇 시야?",
        "서울 날씨 어때?",
        "10 * 25는 얼마야?",
        "안녕하세요!",  # 함수/도구 호출 불필요
        "현재 시간이랑 서울 날씨 둘 다 알려줘",  # 멀티 호출
    ]

    openai_client = OpenAIFunctionCallExample(api_key)
    for question in test_questions:
        print(f"\n📝 테스트: {question}")
        openai_client.chat_with_functions(question)
