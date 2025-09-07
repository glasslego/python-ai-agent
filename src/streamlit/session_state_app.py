import streamlit as st

st.title("세션 상태 예시 - 카운터")

# 세션 상태 초기화
if "count" not in st.session_state:
    st.session_state.count = 0

# 현재 카운트 표시
st.write(f"현재 카운트: {st.session_state.count}")

# 버튼들
col1, col2, col3 = st.columns(3)

with col1:
    if st.button("증가"):
        st.session_state.count += 1

with col2:
    if st.button("감소"):
        st.session_state.count -= 1

with col3:
    if st.button("리셋"):
        st.session_state.count = 0

# 진행률 바
if st.session_state.count > 0:
    progress = min(st.session_state.count / 10, 1.0)
    st.progress(progress)
