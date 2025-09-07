import numpy as np
import pandas as pd

import streamlit as st

st.title("파일 업로드 및 처리")

# 파일 업로드
uploaded_file = st.file_uploader("CSV 파일을 선택하세요", type=["csv"])

if uploaded_file is not None:
    # 파일 읽기
    df = pd.read_csv(uploaded_file)

    st.success("파일이 성공적으로 업로드되었습니다!")

    # 기본 정보 표시
    st.subheader("데이터 기본 정보")
    st.write(f"행 개수: {len(df)}")
    st.write(f"열 개수: {len(df.columns)}")

    # 데이터 미리보기
    st.subheader("데이터 미리보기")
    st.dataframe(df.head())

    # 통계 요약
    st.subheader("통계 요약")
    st.dataframe(df.describe())

    # 컬럼 선택해서 차트 그리기
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    if numeric_columns:
        selected_column = st.selectbox("차트로 표시할 컬럼 선택:", numeric_columns)
        st.line_chart(df[selected_column])
