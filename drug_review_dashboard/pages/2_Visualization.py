from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_cache import get_prepared_data, get_summary
from src.features import extract_keywords


st.title("인사이트 시각화")
st.caption("평점 이상치, 약물별 위험군 비율, 부작용 키워드의 차이를 확인합니다.")

with st.sidebar:
    max_rows = st.number_input("최대 로딩 행 수", min_value=500, max_value=200_000, value=50_000, step=5_000)
    min_reviews = st.slider("약물별 최소 리뷰 수", 1, 100, 5)

df, source, _ = get_prepared_data(int(max_rows), False)
summary = get_summary(int(max_rows), False, min_reviews=min_reviews)

if summary.empty:
    st.warning("필터 조건에 맞는 약물이 없습니다. 최소 리뷰 수를 낮춰보세요.")
    st.stop()

left, right = st.columns([1.1, 1])

with left:
    st.subheader("약물별 위험군 비율 Top 20")
    top_risk = summary.head(20).sort_values("risk_ratio")
    fig = px.bar(
        top_risk,
        x="risk_ratio",
        y="drug_name",
        orientation="h",
        color="avg_rating",
        color_continuous_scale="RdYlGn",
        labels={"risk_ratio": "위험군 비율(%)", "drug_name": "약물명", "avg_rating": "평균 평점"},
        hover_data=["reviews", "condition"],
    )
    fig.update_layout(height=560)
    st.plotly_chart(fig, use_container_width=True)

with right:
    st.subheader("평점 대비 위험군 산점도")
    fig = px.scatter(
        summary,
        x="avg_rating",
        y="risk_ratio",
        size="reviews",
        color="condition",
        hover_name="drug_name",
        labels={"avg_rating": "평균 평점", "risk_ratio": "위험군 비율(%)", "reviews": "리뷰 수"},
    )
    fig.update_layout(height=560, legend_title_text="주요 질환")
    st.plotly_chart(fig, use_container_width=True)

st.divider()

drug_options = summary["drug_name"].tolist()
selected_drug = st.selectbox("약물 상세 보기", drug_options, index=0)
drug_df = df[df["drug_name"] == selected_drug].copy()

c1, c2, c3, c4 = st.columns(4)
c1.metric("선택 약물", selected_drug)
c2.metric("리뷰 수", f"{len(drug_df):,}건")
c3.metric("평균 평점", f"{drug_df['rating'].mean():.2f}")
c4.metric("위험군 비율", f"{drug_df['risk_label'].mean() * 100:.1f}%")

tab_rating, tab_keywords, tab_trend = st.tabs(["평점/IQR", "키워드 비교", "시간 추이"])

with tab_rating:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("선택 약물 평점 분포")
        fig = px.histogram(
            drug_df,
            x="rating",
            color="risk_label",
            nbins=10,
            color_discrete_map={0: "#0f9f8f", 1: "#dc2626"},
            labels={"rating": "평점", "risk_label": "위험 라벨"},
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("상위 약물 평점 Box Plot")
        top_drug_names = summary.head(10)["drug_name"]
        box_df = df[df["drug_name"].isin(top_drug_names)]
        fig = px.box(
            box_df,
            x="drug_name",
            y="rating",
            color="drug_name",
            labels={"drug_name": "약물명", "rating": "평점"},
        )
        fig.update_layout(showlegend=False, xaxis_tickangle=-35)
        st.plotly_chart(fig, use_container_width=True)

    cutoff = drug_df["drug_low_outlier_cutoff"].iloc[0] if len(drug_df) else 0
    outliers = drug_df[drug_df["rating_iqr_low_outlier"] == 1]
    st.info(
        f"{selected_drug}의 낮은 평점 IQR 기준값은 {cutoff:.2f}점입니다. "
        f"이 기준보다 낮은 리뷰는 {len(outliers):,}건입니다."
    )

with tab_keywords:
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("위험군 키워드")
        risky_keywords = extract_keywords(drug_df[drug_df["risk_label"] == 1]["review"], top_n=15)
        if risky_keywords.empty:
            st.info("위험군 키워드가 충분하지 않습니다.")
        else:
            fig = px.bar(
                risky_keywords.sort_values("count"),
                x="count",
                y="keyword",
                orientation="h",
                color_discrete_sequence=["#dc2626"],
                labels={"count": "빈도", "keyword": "키워드"},
            )
            st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("안전군/일반 리뷰 키워드")
        safe_keywords = extract_keywords(drug_df[drug_df["risk_label"] == 0]["review"], top_n=15)
        if safe_keywords.empty:
            st.info("안전군 키워드가 충분하지 않습니다.")
        else:
            fig = px.bar(
                safe_keywords.sort_values("count"),
                x="count",
                y="keyword",
                orientation="h",
                color_discrete_sequence=["#0f9f8f"],
                labels={"count": "빈도", "keyword": "키워드"},
            )
            st.plotly_chart(fig, use_container_width=True)

with tab_trend:
    dated = drug_df.dropna(subset=["date"]).copy()
    if dated.empty:
        st.info("날짜가 없어 시간 추이를 표시할 수 없습니다.")
    else:
        dated["month"] = dated["date"].dt.to_period("M").dt.to_timestamp()
        monthly = (
            dated.groupby("month")
            .agg(avg_rating=("rating", "mean"), risk_ratio=("risk_label", "mean"), reviews=("review", "count"))
            .reset_index()
        )
        monthly["risk_ratio"] = monthly["risk_ratio"] * 100
        fig = px.line(
            monthly,
            x="month",
            y=["avg_rating", "risk_ratio"],
            markers=True,
            labels={"month": "월", "value": "값", "variable": "지표"},
            hover_data=["reviews"],
        )
        st.plotly_chart(fig, use_container_width=True)
        st.caption("평점 평균과 위험군 비율이 같은 방향으로 움직이는지 확인해 약물별 모니터링 포인트를 잡습니다.")
