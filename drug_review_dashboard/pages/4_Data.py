from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_cache import get_prepared_data


st.title("데이터 조회")
st.caption("원본 컬럼과 특성 엔지니어링 결과를 검색하고 내려받을 수 있습니다.")

with st.sidebar:
    max_rows = st.number_input("최대 로딩 행 수", min_value=500, max_value=200_000, value=50_000, step=5_000)

df, source, _ = get_prepared_data(int(max_rows), False)

with st.form("search_form"):
    col1, col2, col3 = st.columns(3)
    with col1:
        keyword = st.text_input("리뷰 검색어", placeholder="chest, nausea, panic ...")
    with col2:
        drug_keyword = st.text_input("약물명 검색", placeholder="Sertraline ...")
    with col3:
        risk_choice = st.multiselect("위험 라벨", ["안전군", "위험군"], default=["안전군", "위험군"])
    submitted = st.form_submit_button("검색", use_container_width=True)

risk_map = {"안전군": 0, "위험군": 1}
filtered = df[df["risk_label"].isin([risk_map[name] for name in risk_choice])].copy()
if keyword:
    filtered = filtered[filtered["review"].str.contains(keyword, case=False, na=False)]
if drug_keyword:
    filtered = filtered[filtered["drug_name"].str.contains(drug_keyword, case=False, na=False)]

st.write(f"검색 결과: **{len(filtered):,}건** / 전체 {len(df):,}건")

default_cols = [
    "drug_name",
    "condition",
    "rating",
    "risk_label",
    "rating_iqr_low_outlier",
    "severe_keyword_count",
    "symptom_keyword_count",
    "review",
]
selected_cols = st.multiselect("표시 컬럼", df.columns.tolist(), default=[c for c in default_cols if c in df.columns])

if selected_cols:
    st.dataframe(
        filtered[selected_cols],
        use_container_width=True,
        height=520,
        column_config={
            "review": st.column_config.TextColumn("리뷰", width="large"),
            "risk_label": st.column_config.NumberColumn("위험 라벨", help="1=위험군, 0=안전군"),
            "rating_iqr_low_outlier": st.column_config.CheckboxColumn("IQR 낮은 이상치"),
        },
    )
else:
    st.warning("표시할 컬럼을 1개 이상 선택하세요.")

csv = filtered[selected_cols].to_csv(index=False).encode("utf-8-sig") if selected_cols else b""
st.download_button(
    "검색 결과 CSV 다운로드",
    data=csv,
    file_name="drug_review_filtered.csv",
    mime="text/csv",
    disabled=not selected_cols,
)
