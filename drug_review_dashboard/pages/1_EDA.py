from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_cache import get_prepared_data


st.title("약물 리뷰 EDA 대시보드")
st.caption("리뷰, 평점, 약물명을 이용해 심각한 부작용 위험군 후보를 탐색합니다.")

with st.sidebar:
    st.subheader("데이터 설정")
    max_rows = st.number_input("최대 로딩 행 수", min_value=500, max_value=200_000, value=50_000, step=5_000)
    prefer_kaggle = st.toggle("KaggleHub 우선 시도", value=False)

df, source, warning = get_prepared_data(int(max_rows), prefer_kaggle)

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

tab_summary, tab_missing, tab_dist, tab_findings = st.tabs(
    ["요약", "결측/스키마", "분포 그래프", "내가 발견한 것"]
)

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
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "해석: 리뷰가 피임 관련 약물(Levonorgestrel, Etonogestrel 등)에 크게 편중되어 있습니다. "
            "약물별 리뷰 수 편차가 커서 비율 비교 시 사이드바의 '약물별 최소 리뷰 수' 필터가 필요합니다."
        )

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
            width="stretch",
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
        width="stretch",
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
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "해석: 평점이 10점과 1점 양극단에 몰린 J자형 분포입니다. 위험군(빨강)은 1~3점, "
            "안전군(청록)은 8~10점에 집중되어 평점이 위험 신호와 강하게 연결됨을 보여줍니다."
        )

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
        st.plotly_chart(fig, width="stretch")
        st.caption(
            "해석: 위험군 리뷰가 안전군보다 길이가 긴 쪽으로 치우쳐 있습니다. 불편을 겪은 사용자가 "
            "증상을 더 길게 서술하는 경향으로, 리뷰 길이를 파생 특성으로 만든 근거입니다."
        )

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
    st.plotly_chart(fig, width="stretch")
    st.caption(
        "해석: 약한 라벨 기준 위험군은 전체의 약 18.6%(전체 데이터 기준)로 불균형 데이터입니다. "
        "그래서 모델 학습 시 class_weight 보정과 F1/Recall 중심 평가를 사용합니다."
    )

with tab_findings:
    st.subheader("내가 발견한 것 (전체 215,063행 EDA 기준)")
    st.markdown(
        """
1. **평점은 J자형 양극화** — 10점(68,005건)과 1점(28,918건)이 양극단. 평균 6.99, 중앙값 8.
   만족/불만이 또렷이 갈려 평점이 강한 위험 신호가 된다.
2. **usefulCount는 극단적 우편향(이상치)** — 평균 28, 최대 1,291, 왜도 4.5. 소수의 '공감 많은' 리뷰가
   존재하므로 평균 대신 분포·중앙값 기준으로 해석해야 한다.
3. **질환·약물 편중** — 최다 질환은 Birth Control(38,436건), 약물은 3,671종으로 약물별 리뷰 수 편차가 크다.
   → 약물별 비교에는 '최소 리뷰 수' 필터가 필요(사이드바에 구현).
4. **데이터 품질 이슈** — `condition` 1,171행에 `</span> users found this comment helpful` 같은
   **HTML 잔여물**, `review`에 `&quot;` 등 HTML 엔티티 발견 → 전처리에서 제거/복원했다.
5. **평점 이상치(IQR)** — 약물별 평점의 Q1−1.5·IQR 미만 리뷰를 '낮은 평점 이상치'로 플래그화했다
   (표본 5건 미만 약물은 전체 분포로 보정). 약물별 상세는 *인사이트 시각화 → 평점/IQR* 탭에서 확인.
6. **위험군(약한 라벨) 18.6%** — 위험 40,042 / 안전 175,021. 위험군 평점 중앙값 2 vs 안전군 9로 뚜렷이 분리.
7. **상관관계에서 누수(leakage) 발견** — `risk_label`과 severe_keyword_count 상관 0.64 등 라벨 정의 변수의
   상관이 비정상적으로 높았다. 이 변수들을 학습 특성에서 **제외**해 정직한 성능(F1≈0.90)을 보고한다(보고서 4·6장).
"""
    )
    st.caption("위 수치는 전체 데이터(215,063행) 기준 — 현재 화면 지표는 사이드바의 로딩 행 수/필터에 따라 달라질 수 있습니다.")
