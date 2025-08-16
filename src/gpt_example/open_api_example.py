import os

from openai import OpenAI

# https://platform.openai.com/docs/quickstart?lang=python

"""
# OpenAI API Python 사용 예제

# 설치:
pip install openai
"""


# API 키 설정 (환경변수 사용 권장)
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ===== 기본 채팅 완성 예제 =====
def basic_chat_example():
    """기본적인 채팅 완성 예제"""
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",  # 또는 "gpt-4"
            messages=[
                {
                    "role": "system",
                    "content": "당신은 도움이 되는 AI 어시스턴트입니다.",
                },
                {
                    "role": "user",
                    "content": "Python에서 리스트를 정렬하는 방법을 알려주세요.",
                },
            ],
            max_tokens=150,
            temperature=0.7,
        )

        print("AI 응답:", response.choices[0].message.content)

    except Exception as e:
        print(f"오류 발생: {e}")


# ===== 대화 이력을 유지하는 채팅봇 =====
def chatbot_with_history():
    """대화 이력을 유지하는 채팅봇"""
    messages = [{"role": "system", "content": "당신은 친근한 AI 어시스턴트입니다."}]

    print("채팅을 시작합니다. 'quit'를 입력하면 종료됩니다.")

    while True:
        user_input = input("\n사용자: ")
        if user_input.lower() == "quit":
            break

        # 사용자 메시지 추가
        messages.append({"role": "user", "content": user_input})

        try:
            response = client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=messages,
                max_tokens=200,
                temperature=0.8,
            )

            ai_response = response.choices[0].message.content
            print(f"AI: {ai_response}")

            # AI 응답을 대화 이력에 추가
            messages.append({"role": "assistant", "content": ai_response})

        except Exception as e:
            print(f"오류: {e}")


# ===== 스트리밍 응답 예제 =====
def streaming_example():
    """실시간으로 응답을 받는 스트리밍 예제"""
    print("스트리밍 응답:")

    try:
        stream = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "AI의 미래에 대해 설명해주세요."}],
            stream=True,
        )

        for chunk in stream:
            if chunk.choices[0].delta.content is not None:
                print(chunk.choices[0].delta.content, end="")
        print()  # 줄바꿈

    except Exception as e:
        print(f"오류: {e}")


# ===== 함수 호출 예제 =====
def function_calling_example():
    """함수 호출 기능을 사용하는 예제"""

    def get_weather(location):
        """날씨 정보를 가져오는 함수 (예시)"""
        return f"{location}의 날씨는 맑고 기온은 22도입니다."

    # 함수 정의
    functions = [
        {
            "name": "get_weather",
            "description": "특정 지역의 날씨 정보를 가져옵니다",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {"type": "string", "description": "도시 이름"}
                },
                "required": ["location"],
            },
        }
    ]

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": "서울 날씨 알려줘"}],
            functions=functions,
            function_call="auto",
        )

        message = response.choices[0].message

        if message.function_call:
            function_name = message.function_call.name
            function_args = eval(message.function_call.arguments)

            if function_name == "get_weather":
                result = get_weather(function_args["location"])
                print(f"날씨 정보: {result}")
        else:
            print(f"AI 응답: {message.content}")

    except Exception as e:
        print(f"오류: {e}")


# ===== 텍스트 요약 예제 =====
def text_summarization():
    """긴 텍스트를 요약하는 예제"""
    long_text = """
    인공지능(AI)은 현재 우리 사회의 많은 분야에서 혁신을 이끌고 있습니다.
    의료 분야에서는 질병 진단과 신약 개발에 활용되고 있으며,
    교육 분야에서는 개인화된 학습 경험을 제공하고 있습니다.
    또한 자율주행차, 스마트홈, 금융 서비스 등 다양한 영역에서
    AI 기술이 적용되어 우리의 일상생활을 더욱 편리하게 만들고 있습니다.
    하지만 동시에 일자리 변화, 개인정보 보호, 윤리적 문제 등
    해결해야 할 과제들도 많이 남아있는 상황입니다.
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "당신은 텍스트를 간결하게 요약하는 전문가입니다.",
                },
                {
                    "role": "user",
                    "content": f"다음 텍스트를 3줄로 요약해주세요:\n\n{long_text}",
                },
            ],
            max_tokens=100,
        )

        print("요약 결과:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"오류: {e}")


# ===== 이미지 분석 예제 (GPT-4 Vision) =====
def image_analysis_example():
    """이미지를 분석하는 예제 (GPT-4V 필요)"""
    try:
        response = client.chat.completions.create(
            model="gpt-4-vision-preview",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": "이 이미지에 무엇이 있는지 설명해주세요.",
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": "https://example.com/image.jpg"  # 실제 이미지 URL
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
        )

        print("이미지 분석 결과:")
        print(response.choices[0].message.content)

    except Exception as e:
        print(f"오류: {e}")


# ===== 사용량 및 비용 확인 =====
def check_usage():
    """API 사용량 확인 (별도 라이브러리 필요할 수 있음)"""
    try:
        # 실제로는 OpenAI 대시보드에서 확인하는 것이 일반적
        print("사용량은 OpenAI 대시보드(platform.openai.com)에서 확인하세요.")
    except Exception as e:
        print(f"오류: {e}")


if __name__ == "__main__":
    print("OpenAI API Python 예제들")
    print("=" * 50)

    # 각 예제 실행 (주석을 해제하여 사용)
    basic_chat_example()
    # chatbot_with_history()
    # streaming_example()
    # function_calling_example()
    # text_summarization()
    # image_analysis_example()
