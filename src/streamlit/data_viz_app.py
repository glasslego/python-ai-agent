import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

import streamlit as st

st.title("데이터 시각화 예시")


# 샘플 데이터 생성
@st.cache_data  # 데이터 캐싱으로 성능 향상
def load_data():
    dates = pd.date_range("2024-01-01", periods=100)
    data = pd.DataFrame(
        {
            "date": dates,
            "value": np.random.randn(100).cumsum(),
            "category": np.random.choice(["A", "B", "C"], 100),
        }
    )
    return data


data = load_data()

# 데이터프레임 표시
st.subheader("데이터 테이블")
st.dataframe(data)

# 라인 차트
st.subheader("라인 차트")
st.line_chart(data.set_index("date")["value"])

# 바 차트
st.subheader("카테고리별 개수")
category_counts = data["category"].value_counts()
st.bar_chart(category_counts)

# Matplotlib 차트
st.subheader("Matplotlib 차트")
fig, ax = plt.subplots()
ax.hist(data["value"], bins=20)
ax.set_xlabel("Value")
ax.set_ylabel("Frequency")
st.pyplot(fig)
