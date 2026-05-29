from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loader import load_drug_reviews
from src.features import add_features


@st.cache_data(show_spinner="약물 리뷰 데이터 로딩 중...")
def get_data(max_rows: int, prefer_kaggle: bool):
    df, source = load_drug_reviews(max_rows=max_rows, prefer_kaggle=prefer_kaggle)
    return add_features(df), source, df.attrs.get("load_warning", "")


st.title("약물 리뷰 EDA 대시보드")
st.caption("리뷰, 평점, 약물명을 이용해 심각한 부작용 위험군 후보를 탐색합니다.")

with st.sidebar:
    st.subheader("데이터 설정")
    max_rows = st.number_input("최대 로딩 행 수", min_value=500, max_value=200_000, value=50_000, step=5_000)
    prefer_kaggle = st.toggle("KaggleHub 우선 시도", value=False)

df, source, warning = get_data(int(max_rows), prefer_kaggle)

if warning:
    st.info(f"실제 데이터 로딩에 실패해 데모 샘플을 사용 중입니다. 원인: {warning}")

with st.sidebar:
    st.markdown("---")
    st.subheader("필터")
    risk_filter = st.multiselect("위험 라벨", ["안전군", "위험군"], default=["안전군", "위험군"])
    rating_range = st.slider("평점 범위", 1.0, 10.0, (1.0, 10.0), step=0.5)
    min_reviews = st.slider("약물별 최소 리뷰 수", 1, 100, 1)

risk_map = {"안전군": 0, "위험군": 1}
filtered = df[
    df["risk_label"].isin([risk_map[name] for name in risk_filter])
    & df["rating"].between(rating_range[0], rating_range[1])
].copy()
drug_counts = filtered["drug_name"].value_counts()
filtered = filtered[filtered["drug_name"].isin(drug_counts[drug_counts >= min_reviews].index)]

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("데이터 출처", source)
k2.metric("리뷰 수", f"{len(filtered):,}건")
k3.metric("약물 수", f"{filtered['drug_name'].nunique():,}개")
k4.metric("평균 평점", f"{filtered['rating'].mean():.2f}")
k5.metric("위험군 비율", f"{filtered['risk_label'].mean() * 100:.1f}%")

st.divider()

tab_summary, tab_missing, tab_dist = st.tabs(["요약", "결측/스키마", "분포 그래프"])

with tab_summary:
    col1, col2 = st.columns([1.2, 1])
    with col1:
        st.subheader("상위 약물")
        top_drugs = filtered["drug_name"].value_counts().head(20).reset_index()
        top_drugs.columns = ["drug_name", "count"]
        fig = px.bar(
            top_drugs,
            x="count",
            y="drug_name",
            orientation="h",
            color="count",
            color_continuous_scale="Tealrose",
            labels={"count": "리뷰 수", "drug_name": "약물명"},
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"}, height=520)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("약물별 요약 통계")
        summary = (
            filtered.groupby("drug_name")
            .agg(
                reviews=("review", "count"),
                avg_rating=("rating", "mean"),
                risk_ratio=("risk_label", "mean"),
                avg_useful=("useful_count", "mean"),
            )
            .sort_values("reviews", ascending=False)
            .head(20)
        )
        summary["risk_ratio"] = summary["risk_ratio"] * 100
        st.dataframe(
            summary.round(2),
            use_container_width=True,
            column_config={
                "risk_ratio": st.column_config.ProgressColumn("위험군 비율(%)", min_value=0, max_value=100),
                "avg_rating": st.column_config.NumberColumn("평균 평점", format="%.2f"),
            },
        )

with tab_missing:
    st.subheader("컬럼별 결측치와 자료형")
    missing = pd.DataFrame(
        {
            "column": df.columns,
            "dtype": [str(df[col].dtype) for col in df.columns],
            "missing_count": [int(df[col].isna().sum()) for col in df.columns],
            "missing_ratio": [df[col].isna().mean() * 100 for col in df.columns],
            "unique_count": [df[col].nunique(dropna=True) for col in df.columns],
        }
    )
    st.dataframe(
        missing.sort_values("missing_ratio", ascending=False),
        use_container_width=True,
        hide_index=True,
        column_config={
            "missing_ratio": st.column_config.ProgressColumn("결측률(%)", min_value=0, max_value=100, format="%.2f"),
        },
    )

    st.info(
        "원본 데이터에는 공식적인 '심각한 ADE' 라벨이 없어, 이 프로젝트는 심각 증상 키워드와 낮은 평점을 결합한 약한 라벨을 target으로 사용합니다."
    )

with tab_dist:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("평점 분포")
        fig = px.histogram(
            filtered,
            x="rating",
            color="risk_label",
            nbins=10,
            barmode="overlay",
            color_discrete_map={0: "#14b8a6", 1: "#ef4444"},
            labels={"rating": "평점", "risk_label": "위험 라벨"},
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("리뷰 길이 분포")
        fig = px.histogram(
            filtered,
            x="review_length",
            color="risk_label",
            nbins=60,
            marginal="box",
            color_discrete_map={0: "#14b8a6", 1: "#ef4444"},
            labels={"review_length": "리뷰 길이", "risk_label": "위험 라벨"},
        )
        fig.update_layout(height=430)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("위험군/안전군 비율")
    risk_counts = filtered["risk_label"].map({0: "안전군", 1: "위험군"}).value_counts().reset_index()
    risk_counts.columns = ["label", "count"]
    fig = px.pie(
        risk_counts,
        names="label",
        values="count",
        hole=0.45,
        color="label",
        color_discrete_map={"안전군": "#14b8a6", "위험군": "#ef4444"},
    )
    fig.update_layout(height=420)
    st.plotly_chart(fig, use_container_width=True)
