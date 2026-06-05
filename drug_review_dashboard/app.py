from __future__ import annotations

import streamlit as st


st.set_page_config(
    page_title="약물 리뷰 부작용 위험 탐지",
    page_icon="ADE",
    layout="wide",
    initial_sidebar_state="expanded",
)

eda = st.Page("pages/1_EDA.py", title="EDA 대시보드", default=True)
visual = st.Page("pages/2_Visualization.py", title="인사이트 시각화")
service = st.Page("pages/3_Model_Service.py", title="모델 서비스")
data = st.Page("pages/4_Data.py", title="데이터 조회")
chatbot = st.Page("pages/5_Chatbot.py", title="AI 상담 챗봇")

pg = st.navigation(
    {
        "UCI Drug Review ADE Monitor": [eda, visual, service, data, chatbot],
    }
)

st.sidebar.markdown("### 약물 리뷰 부작용 위험 탐지")
st.sidebar.caption("UCI ML Drug Review dataset 기반 Streamlit 프로젝트")
st.sidebar.markdown("---")
st.sidebar.caption("데이터 로딩 순서: data/ CSV -> KaggleHub -> 데모 샘플")

pg.run()
