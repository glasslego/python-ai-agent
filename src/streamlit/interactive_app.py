import streamlit as st

st.title("상호작용 위젯 예시")

# 사이드바
st.sidebar.header("설정")

# 텍스트 입력
name = st.text_input("이름을 입력하세요:")
if name:
    st.write(f"안녕하세요, {name}님!")

# 숫자 입력
age = st.number_input("나이를 입력하세요:", min_value=0, max_value=150, value=25)
st.write(f"당신의 나이는 {age}세입니다.")

# 슬라이더
temperature = st.slider("온도를 선택하세요 (°C):", -50, 50, 20)
st.write(f"현재 온도: {temperature}°C")

# 선택 박스
city = st.selectbox("도시를 선택하세요:", ["서울", "부산", "대구", "인천"])
st.write(f"선택된 도시: {city}")

# 체크박스
show_data = st.checkbox("데이터 표시")
if show_data:
    st.write("체크박스가 선택되었습니다!")
