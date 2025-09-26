"""
Streamlit 기반 웹 인터페이스

LangGraph를 사용한 스마트 어시스턴트의 웹 UI
"""

import os

from dotenv import load_dotenv

import streamlit as st
from src.simple_assistant2.graph.graph import SmartAssistantGraph

# 환경 변수 로드
load_dotenv()


def init_session_state():
    """세션 상태 초기화"""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "assistant" not in st.session_state:
        try:
            st.session_state.assistant = SmartAssistantGraph()
        except Exception as e:
            st.error(f"어시스턴트 초기화 실패: {e}")
            st.session_state.assistant = None


def main():
    """Streamlit 앱 메인 함수"""
    st.set_page_config(
        page_title="스마트 개인 비서 v2.0", page_icon="🤖", layout="wide"
    )

    # 헤더
    st.title("🤖 스마트 개인 비서 v2.0")
    st.caption("LangGraph 기반 상태 그래프 처리")

    # API 키 확인
    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ OPENAI_API_KEY 환경변수를 설정해주세요.")
        st.code("export OPENAI_API_KEY2'your-api-key'")
        return

    # 세션 상태 초기화
    init_session_state()

    if st.session_state.assistant is None:
        st.error("어시스턴트를 초기화할 수 없습니다.")
        return

    # 사이드바에 기능 설명
    with st.sidebar:
        st.header("🔧 주요 기능")
        st.markdown(
            """
        **📰 뉴스 검색**
        - "삼성전자 뉴스 찾아줘"

        **🌤️ 날씨 조회**
        - "서울 날씨 어때?"

        **💱 환율 조회**
        - "달러 환율 알려줘"

        **📈 주식 조회**
        - "삼성전자 주가 확인해줘"

        **📅 일정 관리**
        - "내일 회의 일정 추가해줘"
        - "오늘 일정 알려줘"

        **💡 일반 대화**
        - "안녕하세요"
        """
        )

        st.divider()

        st.header("🔄 LangGraph 처리 과정")
        st.markdown(
            """
        1. **의도 분석**: 사용자 입력 분석
        2. **도구 선택**: 적절한 도구 결정
        3. **도구 실행**: API 호출 및 데이터 처리
        4. **응답 생성**: 최종 답변 생성
        """
        )

        if st.button("대화 기록 지우기"):
            st.session_state.messages = []
            st.rerun()

    # 메인 채팅 영역
    st.header("💬 채팅")

    # 이전 대화 표시
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # 사용자 입력
    if prompt := st.chat_input("메시지를 입력하세요..."):
        # 사용자 메시지 추가
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # 어시스턴트 응답
        with st.chat_message("assistant"):
            with st.spinner("답변 생성 중..."):
                try:
                    response = st.session_state.assistant.chat(prompt)
                    st.markdown(response)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": response}
                    )
                except Exception as e:
                    error_msg = f"오류가 발생했습니다: {e}"
                    st.error(error_msg)
                    st.session_state.messages.append(
                        {"role": "assistant", "content": error_msg}
                    )


if __name__ == "__main__":
    main()
