# pages/1_EDA.py — 1차 작업: 데이터 들여다보기
# 7주차 data_load_explore.ipynb 의 Streamlit 버전이라고 생각하면 됩니다.
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))  # src/ import 가능하게

import streamlit as st
from src.data_loader import load_data

st.title("📊 EDA — 데이터 살펴보기")

df = load_data()

# --- 1. 기본 정보 ---
st.header("1. 데이터 개요")
c1, c2, c3 = st.columns(3)
c1.metric("행 수", f"{len(df):,}")
c2.metric("열 수", f"{df.shape[1]}")
c3.metric("결측 있는 열", f"{int(df.isna().any().sum())}")

st.subheader("미리보기")
st.dataframe(df.head(20), use_container_width=True)

# --- 2. 요약 통계 ---
st.header("2. 요약 통계")
st.dataframe(df.describe(include="all").T, use_container_width=True)

# --- 3. 결측치 ---
st.header("3. 결측치")
na = df.isna().sum()
na = na[na > 0].sort_values(ascending=False)
if len(na) == 0:
    st.success("결측치 없음 ✅")
else:
    st.bar_chart(na)

# --- 4. 발견 사실 메모 ---
# TODO: EDA로 알게 된 것을 여기에 글로 적으세요 (보고서에 그대로 쓰입니다)
st.header("4. 내가 발견한 것")
st.info("""
- (예) target 클래스가 8:2로 불균형하다
- (예) 'X' 컬럼에 결측이 30% 있어 제거/대체가 필요하다
""")
