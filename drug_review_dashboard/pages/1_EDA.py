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
from src.features import LABEL_DEFINING_FEATURES, MODEL_FEATURES


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

# st.metric은 긴 문자열을 잘라 보여주므로 출처는 짧은 라벨로 표시하고,
# 어떤 파일을 읽었는지 전체 내용은 툴팁(help)에 담는다.
SOURCE_SHORT = {"local": "로컬 CSV", "kaggle cache": "Kaggle 캐시", "kagglehub": "KaggleHub", "demo sample": "데모 샘플"}
source_label = next((label for prefix, label in SOURCE_SHORT.items() if source.startswith(prefix)), source)

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("데이터 출처", source_label, help=source)
k2.metric("리뷰 수", f"{len(filtered):,}건")
k3.metric("약물 수", f"{filtered['drug_name'].nunique():,}개")
k4.metric("평균 평점", f"{filtered['rating'].mean():.2f}")
k5.metric("위험군 비율", f"{filtered['risk_label'].mean() * 100:.1f}%")

st.divider()

# 루브릭 "EDA & 데이터 이해": 발견 사실을 글로 정리해 앱 안에서도 바로 보이게 한다.
with st.expander("📌 EDA 발견 요약 — 글로 정리한 5가지 사실 (전체 215,063행 기준)", expanded=True):
    st.markdown(
        """
1. **평점은 J자형 분포** — 10점(68,005건)과 1점(28,918건) 양극단에 몰려 만족/불만이 또렷이 갈린다 (평균 6.99, 중앙값 8).
2. **위험군(약한 라벨) 비율 약 18.6%** — 위험군 평점 중앙값은 2점, 안전군은 9점으로 라벨과 평점이 뚜렷이 분리된다.
3. **질환·약물 편중** — 최다 질환은 Birth Control(38,436건), 약물은 3,671종으로 리뷰 수 편차가 커 **약물별 최소 리뷰 수 필터**(사이드바)가 필요하다.
4. **데이터 품질 이슈** — `condition` 1,171행에 HTML 잔여물, `review`에 `&quot;` 등 HTML 엔티티가 섞여 있어 로딩 단계에서 복원·정제했다.
5. **상관 분석에서 누수 단서 발견** — `severe_keyword_count`와 라벨의 상관 0.64. 라벨을 정의한 컬럼이므로 **학습 특성에서 제외**했다 (보고서 3.3절·4장, 모델 서비스 페이지 참고).
        """
    )

tab_summary, tab_missing, tab_dist, tab_features = st.tabs(["요약", "결측/스키마", "분포 그래프", "특성 엔지니어링"])

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
        st.caption("→ 리뷰가 피임·정신과 계열 약물에 집중되어 있다. 약물별 리뷰 수 편차가 크므로, 약물 간 비교에는 최소 리뷰 수 필터가 필요하다.")

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
        st.caption("→ 리뷰가 많은 인기 약물이라도 위험군 비율은 제각각이다. 리뷰 수(인기)와 안전 신호는 별개임을 확인할 수 있다.")

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
    st.caption("→ 핵심 컬럼(review·rating)에는 결측이 거의 없다. 로딩 단계에서 condition은 'Unknown', useful_count는 0으로 보정하고, date 결측 행은 시간 추이 분석에서만 제외한다.")

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
        st.caption("→ 평점은 10점·1점 양극단의 J자형 분포이고, 위험군(빨강)은 1~3점 구간에 몰린다. 낮은 평점과 텍스트 위험 신호가 같은 방향으로 움직임을 보여준다.")

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
        st.caption("→ 위험군 리뷰가 더 긴 구간까지 분포한다. 부작용 경험은 길게 서술되는 경향이 있어, 리뷰 길이(review_length)를 모델의 비누수 특성으로 사용한다.")

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
    st.caption("→ 위험군은 대략 5건 중 1건(전체 기준 18.6%)인 불균형 데이터다. 모델 평가에서 Accuracy만이 아니라 F1·Recall을 함께 보고하는 이유다.")

with tab_features:
    # 원본 7컬럼에서 어떤 특성이 새로 만들어졌는지 한눈에 보여주는 탭.
    # (특성별 EDA 근거·계산식 상세는 docs/특성_엔지니어링.md)
    st.subheader("원본 컬럼 → 새로 만든 특성")
    st.markdown(
        "원본 데이터는 **7개 컬럼**(`drug_name`, `condition`, `review`, `rating`, `date`, "
        "`useful_count`, ID)뿐이며, 아래 특성들은 전부 `src/features.py`의 `add_features()`가 "
        "리뷰 텍스트·평점·약물 통계에서 **새로 만들어낸(가공·결합·도출한)** 값입니다."
    )

    FEATURE_GUIDE = [
        ("review_length", "리뷰 글자 수", "위험군 리뷰가 더 길다는 EDA 발견", "✅ 사용"),
        ("word_count", "리뷰 단어 수", "위와 동일 (문장부호에 강건한 길이 신호)", "✅ 사용"),
        ("exclamation_count", "느낌표(!) 개수", "위험 리뷰의 격한 감정 표현", "✅ 사용"),
        ("uppercase_ratio", "대문자 비율", "'NEVER take this' 같은 강조 표현", "✅ 사용"),
        ("positive_keyword_count", "긍정 키워드 종류 수", "안전군 상위 어휘(worked/helped...) 발견", "✅ 사용"),
        ("rating_iqr_low_outlier", "약물별 IQR 기준 저평점 이상치", "약물마다 평점 분포가 제각각(Box Plot)", "✅ 사용"),
        ("drug_review_count", "약물별 리뷰 수", "약물 3,671종 — 표본 신뢰도 프록시", "✅ 사용"),
        ("drug_avg_rating", "약물별 평균 평점", "약물 평판 기준선", "✅ 사용"),
        ("severe_keyword_count", "심각 증상 키워드 종류 수", "약한 라벨 재료 (clause A)", "❌ 라벨 정의용 — 학습 제외"),
        ("symptom_keyword_count", "일반 증상 키워드 종류 수", "약한 라벨 재료 (clause B)", "❌ 라벨 정의용 — 학습 제외"),
        ("low_rating_flag", "평점 ≤ 3 플래그", "평점 J자형 — 불만이 1~3점 집중", "❌ 라벨 정의용 — 학습 제외"),
        ("risk_label", "약한 라벨 (target)", "심각 키워드 OR (평점≤3 AND 증상 키워드)", "🎯 예측 대상"),
    ]
    guide_df = pd.DataFrame(FEATURE_GUIDE, columns=["특성명", "의미", "만든 이유 (EDA 근거)", "모델 사용"])
    st.dataframe(guide_df, use_container_width=True, hide_index=True)
    st.caption(
        f"→ 학습에는 비누수 특성 {len(MODEL_FEATURES)}개 + 리뷰 원문 TF-IDF만 들어가고, "
        f"라벨을 정의한 {len(LABEL_DEFINING_FEATURES)}개는 타깃 누수 방지를 위해 제외합니다 (보고서 3.3절)."
    )

    st.subheader("생성된 특성 값 미리보기 (상위 10행)")
    preview_cols = ["drug_name", "rating", "review_length", "word_count", "exclamation_count",
                    "uppercase_ratio", "positive_keyword_count", "rating_iqr_low_outlier",
                    "drug_review_count", "drug_avg_rating", "severe_keyword_count",
                    "symptom_keyword_count", "low_rating_flag", "risk_label"]
    st.dataframe(filtered[[c for c in preview_cols if c in filtered.columns]].head(10),
                 use_container_width=True, hide_index=True)
    st.caption("→ 전체 데이터에서 특성별로 검색·필터링하려면 '데이터 조회' 페이지를 이용하세요.")
