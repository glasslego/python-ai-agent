"""
# claude API Python 사용 예제

# 설치:
pip install anthropic
"""

import os

import anthropic
from dotenv import load_dotenv

load_dotenv()

# API 키 설정
client = anthropic.Anthropic(api_key=os.getenv("CLAUDE_API_KEY"))


def basic_chat_example():
    """기본적인 채팅 완성 예제"""
    try:
        # 메시지 전송
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            messages=[{"role": "user", "content": "파이썬으로 간단한 계산기 만들어줘"}],
        )

        print(message.content)

    except Exception as e:
        print(f"오류 발생: {e}")


if __name__ == "__main__":
    basic_chat_example()
