"""
LangGraph 기반 스마트 개인 비서 챗봇

기존 simple_assistant의 기능을 LangGraph를 사용하여 상태 그래프로 구현
"""

import os

from dotenv import load_dotenv
from graph.graph import SmartAssistantGraph

# 환경 변수 로드
load_dotenv()


def print_welcome():
    """환영 메시지 출력"""
    print("=" * 60)
    print("🤖 스마트 개인 비서 챗봇 v2.0 (LangGraph)")
    print("=" * 60)
    print("사용 가능한 기능:")
    print("📰 뉴스 검색: '삼성전자 뉴스 찾아줘'")
    print("🌤️  날씨 조회: '서울 날씨 어때?'")
    print("💱 환율 조회: '달러 환율 알려줘'")
    print("📈 주식 조회: '삼성전자 주가 확인해줘'")
    print("📅 일정 관리: '내일 회의 일정 추가해줘'")
    print("💡 일반 대화: '안녕하세요'")
    print("-" * 60)
    print("🔄 LangGraph 상태 기반 처리:")
    print("   1. 의도 분석 → 2. 도구 선택 → 3. 도구 실행 → 4. 응답 생성")
    print("-" * 60)
    print("종료: 'quit' 또는 'exit'")
    print("=" * 60)


def run_individual_tests():
    """각 기능별 개별 테스트 실행"""
    print("\n🧪 개별 기능 테스트 시작")
    print("=" * 60)

    try:
        assistant = SmartAssistantGraph()
        print("✅ 어시스턴트 초기화 완료\n")

        # 테스트 케이스 정의
        test_cases = [
            {
                "category": "일반 대화",
                "icon": "💬",
                "tests": ["안녕하세요", "오늘 날씨가 좋네요", "감사합니다"],
            },
            {
                "category": "뉴스 검색",
                "icon": "📰",
                "tests": [
                    "삼성전자 뉴스 찾아줘",
                    "AI 뉴스 검색해줘",
                    "최신 뉴스 알려줘",
                ],
            },
            {
                "category": "날씨 조회",
                "icon": "🌤️",
                "tests": ["서울 날씨 어때?", "부산 날씨 알려줘", "오늘 날씨는?"],
            },
            {
                "category": "환율 조회",
                "icon": "💱",
                "tests": ["달러 환율 알려줘", "USD KRW 환율 확인해줘", "엔화 환율은?"],
            },
            {
                "category": "주식 조회",
                "icon": "📈",
                "tests": [
                    "삼성전자 주가 확인해줘",
                    "애플 주식 알려줘",
                    "005930 주가는?",
                ],
            },
            {
                "category": "일정 관리",
                "icon": "📅",
                "tests": [
                    "오늘 일정 알려줘",
                    "내일 회의 일정 추가해줘",
                    "2024-03-25 일정 확인해줘",
                ],
            },
        ]

        # 각 카테고리별 테스트 실행
        for test_group in test_cases:
            print(f"\n{test_group['icon']} {test_group['category']} 테스트:")
            print("-" * 50)

            for i, test_input in enumerate(test_group["tests"], 1):
                print(f"\n[테스트 {i}/{len(test_group['tests'])}]")
                print(f"👤 입력: {test_input}")
                print("🤖 응답: ", end="")

                try:
                    response = assistant.chat(test_input)
                    print(response)
                except Exception as e:
                    print(f"❌ 오류: {e}")

                # 테스트 간 잠시 대기
                input("\n⏸️  다음 테스트로 계속하려면 Enter를 누르세요...")

            print(f"\n✅ {test_group['category']} 테스트 완료")

        print("\n🎉 모든 개별 테스트 완료!")

    except Exception as e:
        print(f"❌ 테스트 초기화 오류: {e}")


def run_interactive_mode():
    """대화형 모드 실행"""
    print_welcome()

    try:
        # LangGraph 기반 어시스턴트 초기화
        print("\n🔄 LangGraph 기반 어시스턴트 초기화 중...")
        assistant = SmartAssistantGraph()
        print("✅ 챗봇 초기화 완료!")
        print("💡 상태 그래프가 준비되었습니다.")

        while True:
            user_input = input("\n👤 You: ").strip()

            if user_input.lower() in ["quit", "exit", "종료"]:
                print("👋 안녕히 가세요!")
                break

            if not user_input:
                continue

            print("\n🤖 Assistant: ", end="")
            try:
                # LangGraph를 통한 응답 생성
                response = assistant.chat(user_input)
                print(response)
            except Exception as e:
                print(f"오류 발생: {e}")

    except KeyboardInterrupt:
        print("\n\n👋 안녕히 가세요!")
    except Exception as e:
        print(f"❌ 오류 발생: {e}")


def main():
    """메인 실행 함수"""
    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        print("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        return

    # 실행 모드 선택
    print("🤖 Simple Assistant v2.0 - LangGraph")
    print("=" * 50)
    print("실행 모드를 선택하세요:")
    print("1. 대화형 모드 (기본)")
    print("2. 개별 기능 테스트")
    print("3. 종료")
    print("-" * 50)

    while True:
        try:
            choice = input("선택 (1-3): ").strip()

            if choice == "1" or choice == "":
                run_interactive_mode()
                break
            elif choice == "2":
                run_individual_tests()
                break
            elif choice == "3":
                print("👋 프로그램을 종료합니다.")
                break
            else:
                print("❌ 잘못된 선택입니다. 1, 2, 또는 3을 입력하세요.")
        except KeyboardInterrupt:
            print("\n👋 프로그램을 종료합니다.")
            break


if __name__ == "__main__":
    main()
