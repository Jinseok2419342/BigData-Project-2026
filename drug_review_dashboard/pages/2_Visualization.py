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
    st.caption("→ 위험군 비율 상위 약물은 평균 평점(막대 색)도 낮은 경향이다. 텍스트 기반 위험 신호와 평점 신호가 일관되게 움직여, 두 신호를 함께 쓰는 근거가 된다.")

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
    st.caption("→ 평균 평점이 낮을수록 위험군 비율이 높아지는 음(−)의 관계가 보인다. 점이 작은(리뷰가 적은) 약물은 비율이 과장될 수 있어 최소 리뷰 수 필터를 적용했다.")

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
        st.caption("→ 선택 약물에서 위험군(빨강)이 어느 평점대에 몰리는지 보여준다. 1~3점에 집중될수록 평점·텍스트 신호가 일치하는 약물이다.")

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
        st.caption("→ 약물마다 평점의 중앙값과 IQR(상자 폭)이 크게 다르다. 전체 기준이 아닌 **약물별 IQR 기준**으로 낮은 평점 이상치를 잡는 근거다.")

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
            st.caption("→ 위험군 리뷰에는 심각 증상 계열 단어가 상위에 온다. 약한 라벨이 실제 텍스트 신호와 정합함을 보여준다.")

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
            st.caption("→ 안전군 리뷰는 효과·개선 계열 단어가 상위다. 두 군의 어휘가 뚜렷이 갈리는 것이 리뷰 원문 TF-IDF만으로도 분류가 가능한 이유다.")

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
