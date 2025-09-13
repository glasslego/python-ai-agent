# flake8: noqa E501

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

import streamlit as st

load_dotenv()

"""
실행 하기

streamlit run src/llm/langchain/simple_agent/simple_chatbot.py
"""


# LLM이 답변을 생성하는 함수
def generate_response(input_text: str) -> str:
    """
    OpenAI LLM을 사용하여 주어진 텍스트에 대한 응답을 생성합니다.
    """
    # LLM 모델을 초기화합니다.
    llm = ChatOpenAI(
        model="gpt-3.5-turbo",  # 최신 모델 사용 권장 (또는 gpt-3.5-turbo)
        temperature=0,  # 답변의 창의성을 0으로 설정하여 일관된 답변 유도
    )

    response = llm.invoke(input_text)
    return response.content


# Streamlit 앱의 메인 함수
def main():
    """
    Streamlit을 사용하여 간단한 챗봇 UI를 실행합니다.
    """
    # 웹 페이지의 제목 설정
    st.title("🤖 간단한 AI 챗봇")

    # 'st.form'을 사용하면 사용자가 입력을 마치고 '보내기' 버튼을 누를 때까지
    # 페이지가 다시 로드되는 것을 방지하여 효율적입니다.
    with st.form("my_form"):
        # 여러 줄의 텍스트를 입력받을 수 있는 입력창을 생성합니다.
        user_input = st.text_area(
            "질문을 입력하세요:", "OpenAI가 제공하는 텍스트 모델의 종류는 무엇인가요?"
        )

        # 폼 제출(전송) 버튼
        submitted = st.form_submit_button("질문하기")

        # '질문하기' 버튼이 클릭되었을 때만 아래 로직을 실행합니다.
        if submitted:
            response = generate_response(user_input)
            # 생성된 답변을 정보 상자(st.info) 형태로 화면에 표시합니다.
            st.info(response)


if __name__ == "__main__":
    main()
