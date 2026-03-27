# eda_dashboard.py
import streamlit as st
import pandas as pd
import numpy as np

st.set_page_config(page_title='기초 EDA 대시보드', layout='wide')
st.title('기초 EDA 대시보드')

# ---- 더미 데이터 생성 ----
np.random.seed(42)

df = pd.DataFrame({
    '날짜': pd.date_range('2026-01-01', periods=100),
    '카테고리': np.random.choice(['전자제품', '의류', '식품'], 100),
    '매출': np.random.randint(100, 1000, 100),
    '고객수': np.random.randint(10, 200, 100),
    '전환율': np.random.uniform(0.01, 0.20, 100).round(4)
})

# ---- 사이드바: 필터 ----
st.sidebar.header('필터')

# 카테고리 선택
selected_category = st.sidebar.selectbox(
    '카테고리 선택',
    ['전체'] + list(df['카테고리'].unique())
)

# 데이터 범위 (행 수) 슬라이더
data_range = st.sidebar.slider(
    '데이터 범위 (일수)',
    min_value=10,
    max_value=100,
    value=50,
    step=10
)

# ---- 데이터 필터링 ----
filtered_df = df.head(data_range)

if selected_category != '전체':
    filtered_df = filtered_df[filtered_df['카테고리'] == selected_category]

# 사이드바에 필터링 결과 요약
st.sidebar.write('---')
st.sidebar.write(f'필터링된 데이터: **{len(filtered_df)}행**')

# ---- 메인 영역: 탭 구성 ----
tab1, tab2 = st.tabs(['요약 대시보드', '원본 데이터'])

with tab1:
    # -- KPI 지표 --
    st.subheader('핵심 지표')
    kpi1, kpi2, kpi3 = st.columns(3)

    kpi1.metric(
        label='총 매출',
        value=f"₩{filtered_df['매출'].sum():,}만",
        delta=f"{filtered_df['매출'].mean():.0f} (평균)"
    )
    kpi2.metric(
        label='총 고객수',
        value=f"{filtered_df['고객수'].sum():,}명",
        delta=f"{filtered_df['고객수'].mean():.0f} (평균)"
    )
    kpi3.metric(
        label='평균 전환율',
        value=f"{filtered_df['전환율'].mean():.2%}",
        delta=f"{filtered_df['전환율'].std():.2%} (표준편차)"
    )

    st.write('---')

     # -- 차트 영역 --
    st.subheader('매출 추이')

    # 날짜별 매출 집계
    chart_data = filtered_df.groupby('날짜')['매출'].sum().reset_index()
    chart_data = chart_data.set_index('날짜')

    st.line_chart(chart_data)

with tab2:
    st.subheader('필터링된 원본 데이터')

    # 데이터 건수 표시
    st.write(f'총 **{len(filtered_df)}건**의 데이터')

    # 인터랙티브 데이터프레임
    st.dataframe(
        filtered_df,
        use_container_width=True, # 전체 너비 사용
        height=400 # 높이 고정
    )

    # Expander로 기술통계 숨기기
    with st.expander('기술통계 보기'):
        st.dataframe(filtered_df.describe())

