"""
스마트 개인 비서 챗봇
사용자의 자연어 질문을 분석해서 적절한 API를 호출하고 답변하는 멀티 기능 챗봇
"""

import os

from workflows.main_workflow import SmartAssistant


def print_welcome():
    print("=" * 50)
    print("🤖 스마트 개인 비서 챗봇")
    print("=" * 50)
    print("사용 가능한 기능:")
    print("📰 뉴스 검색: '삼성전자 뉴스 찾아줘'")
    print("🌤️  날씨 조회: '서울 날씨 어때?'")
    print("💱 환율 조회: '달러 환율 알려줘'")
    print("📈 주식 조회: '삼성전자 주가 확인해줘'")
    print("📅 일정 관리: '내일 회의 일정 추가해줘'")
    print("💡 복합 질문: '삼성전자 주가, 뉴스, 달러환율 모두 알려줘'")
    print("-" * 50)
    print("종료: 'quit' 또는 'exit'")
    print("=" * 50)


def main():
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        print("export OPENAI_API_KEY='your-api-key'")
        return

    print_welcome()

    try:
        assistant = SmartAssistant()
        print("✅ 챗봇 초기화 완료!")

        while True:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ["quit", "exit", "종료"]:
                print("👋 안녕히 가세요!")
                break

            if not user_input:
                continue

            print("\n🤖 Assistant: ", end="")
            response = assistant.chat(user_input)
            print(response)

    except KeyboardInterrupt:
        print("\n\n👋 안녕히 가세요!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


if __name__ == "__main__":
    main()
