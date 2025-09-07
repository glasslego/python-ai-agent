import streamlit as st

st.title("간단한 계산기")

# 사이드바에 계산 옵션
st.sidebar.header("계산 설정")

# 숫자 입력
col1, col2 = st.columns(2)
with col1:
    num1 = st.number_input("첫 번째 숫자:", value=0.0)
with col2:
    num2 = st.number_input("두 번째 숫자:", value=0.0)

# 연산 선택
operation = st.radio("연산을 선택하세요:", ["덧셈", "뺄셈", "곱셈", "나눗셈"])

# 계산 버튼
if st.button("계산하기"):
    if operation == "덧셈":
        result = num1 + num2
    elif operation == "뺄셈":
        result = num1 - num2
    elif operation == "곱셈":
        result = num1 * num2
    elif operation == "나눗셈":
        if num2 != 0:
            result = num1 / num2
        else:
            st.error("0으로 나눌 수 없습니다!")
            result = None

    if result is not None:
        st.success(f"결과: {result}")
