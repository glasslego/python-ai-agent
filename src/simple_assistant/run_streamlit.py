import subprocess
import sys


def run_demo_data_creation():
    """데모 데이터 생성 실행"""
    print("🔧 데모 데이터 생성 중...")
    try:
        result = subprocess.run(
            [sys.executable, "demo_data.py"], capture_output=True, text=True
        )
        if result.returncode == 0:
            print("✅ 데모 데이터 생성 완료")
            print(result.stdout)
        else:
            print("❌ 데모 데이터 생성 실패")
            print(result.stderr)
    except Exception as e:
        print(f"❌ 데모 데이터 생성 중 오류: {e}")


def run_streamlit():
    """Streamlit 앱 실행"""
    print("🚀 Streamlit 앱 실행 중...")
    try:
        subprocess.run(["streamlit", "run", "app.py", "--server.port", "8501"])
    except KeyboardInterrupt:
        print("\n👋 Streamlit 앱이 종료되었습니다.")
    except Exception as e:
        print(f"❌ Streamlit 실행 중 오류: {e}")


def main():
    print("=" * 50)
    print("🤖 스마트 개인 비서 Streamlit 앱")
    print("=" * 50)

    # 데모 데이터 생성
    run_demo_data_creation()

    print("\n" + "=" * 50)
    print("🌐 Streamlit 웹 인터페이스 실행")
    print("브라우저에서 http://localhost:8501 로 접속하세요")
    print("종료하려면 Ctrl+C 를 누르세요")
    print("=" * 50)

    # Streamlit 앱 실행
    run_streamlit()


if __name__ == "__main__":
    main()
