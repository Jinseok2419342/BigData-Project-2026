from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loader import load_drug_reviews
from src.features import add_features, make_prediction_frame
from src.llm_helper import build_rule_based_report, try_ollama_report
from src.modeling import predict_risk, train_risk_model


@st.cache_data(show_spinner="모델용 데이터 로딩 중...")
def get_data(max_rows: int):
    df, source = load_drug_reviews(max_rows=max_rows)
    return add_features(df), source


@st.cache_resource(show_spinner="RandomForest 모델 학습 중...")
def get_model(df: pd.DataFrame, sample_size: int):
    return train_risk_model(df, sample_size=sample_size)


def guess_drug_from_filename(file_name: str, options: list[str]) -> str | None:
    stem = Path(file_name).stem.lower().replace("_", " ").replace("-", " ")
    for drug in options:
        if drug.lower() in stem:
            return drug
    return None


st.title("모델/서비스 페이지")
st.caption("입력한 약물명과 복용 리뷰를 바탕으로 부작용 위험도를 예측합니다.")

with st.sidebar:
    max_rows = st.number_input("최대 데이터 행 수", min_value=500, max_value=200_000, value=50_000, step=5_000)
    sample_size = st.number_input("모델 학습 샘플 수", min_value=500, max_value=50_000, value=12_000, step=1_000)
    use_ollama = st.toggle("Ollama 리포트 시도", value=False)
    ollama_model = st.text_input("Ollama 모델명", value="gemma3", disabled=not use_ollama)

df, source = get_data(int(max_rows))
bundle = get_model(df, int(sample_size))

metric_cols = st.columns(5)
metric_cols[0].metric("학습 데이터", f"{bundle.train_rows:,}건")
metric_cols[1].metric("검증 데이터", f"{bundle.test_rows:,}건")
metric_cols[2].metric("Accuracy", f"{bundle.metrics['accuracy'] * 100:.1f}%")
metric_cols[3].metric("Recall", f"{bundle.metrics['recall'] * 100:.1f}%")
metric_cols[4].metric("F1", f"{bundle.metrics['f1'] * 100:.1f}%")

with st.expander("모델 성능과 특성 중요도 보기", expanded=False):
    col1, col2 = st.columns([1, 1.2])
    with col1:
        cm = bundle.metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["실제 안전군", "실제 위험군"], columns=["예측 안전군", "예측 위험군"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", labels={"color": "건수"})
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig = px.bar(
            bundle.feature_importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "중요도", "feature": "특성"},
        )
        st.plotly_chart(fig, use_container_width=True)

st.divider()

drug_options = sorted(df["drug_name"].dropna().astype(str).unique().tolist())
default_drug = drug_options[0] if drug_options else "Unknown"

left, right = st.columns([0.95, 1.05])

with left:
    st.subheader("사용자 입력")
    uploaded = st.file_uploader("약통 이미지 업로드", type=["png", "jpg", "jpeg"])
    guessed = None
    if uploaded is not None:
        image = Image.open(uploaded)
        st.image(image, caption="업로드한 이미지", use_container_width=True)
        guessed = guess_drug_from_filename(uploaded.name, drug_options)
        if guessed:
            st.caption(f"파일명에서 약물 후보를 찾았습니다: {guessed}")
        else:
            st.caption("현재 버전은 이미지 미리보기와 파일명 기반 후보 매칭까지 지원합니다.")

    selected = st.selectbox(
        "약물명 선택",
        drug_options if drug_options else [default_drug],
        index=drug_options.index(guessed) if guessed in drug_options else 0,
    )
    manual_drug = st.text_input("직접 입력 약물명", value=selected)
    review_text = st.text_area(
        "복용 리뷰 또는 증상 설명",
        value="I had chest pain and my heart was racing after taking this medicine.",
        height=170,
    )
    rating = st.slider("사용자 평점", 1.0, 10.0, 2.0, step=0.5)
    useful_count = st.number_input("유사한 경험을 참고한 사람 수(usefulCount)", min_value=0, value=0, step=1)
    submitted = st.button("위험도 예측", type="primary", use_container_width=True)

with right:
    st.subheader("예측 결과")
    if not submitted:
        st.info("왼쪽 입력값을 확인한 뒤 버튼을 누르면 예측 결과가 표시됩니다.")
    else:
        pred_row = make_prediction_frame(df, manual_drug, review_text, rating, useful_count)
        result = predict_risk(bundle, pred_row)
        probability = result["risk_probability"]
        iqr_flag = bool(pred_row["rating_iqr_low_outlier"].iloc[0])

        r1, r2, r3 = st.columns(3)
        r1.metric("위험도 점수", f"{probability * 100:.1f}%")
        r2.metric("분류 결과", result["label"])
        r3.metric("IQR 이상치", "감지" if iqr_flag else "미감지")

        st.progress(probability)
        if probability >= 0.7:
            st.error("심각한 부작용 위험군으로 예측되었습니다. 긴급 증상이 있으면 의료진 상담이 우선입니다.")
        elif probability >= 0.4:
            st.warning("중간 위험으로 예측되었습니다. 증상 지속 여부와 과거 사례를 함께 확인하세요.")
        else:
            st.success("현재 입력만으로는 위험도가 낮게 예측되었습니다. 단, 실제 증상이 심하면 의료 상담이 필요합니다.")

        similar = df[df["drug_name"].str.lower() == manual_drug.lower()].copy()
        if similar.empty:
            similar = df[df["drug_name"] == selected].copy()
        similar = similar.sort_values(["risk_label", "useful_count"], ascending=[False, False]).head(8)

        st.markdown(
            build_rule_based_report(
                drug_name=manual_drug,
                review=review_text,
                probability=probability,
                iqr_flag=iqr_flag,
                similar_cases=similar,
            )
        )

        if use_ollama:
            prompt = f"""
약물명: {manual_drug}
사용자 증상/리뷰: {review_text}
모델 위험도: {probability * 100:.1f}%
IQR 이상치 여부: {iqr_flag}

의학적 진단은 피하고, 수업용 서비스 리포트 형식으로 위험 신호와 다음 행동을 한국어로 정리해줘.
"""
            ollama_text = try_ollama_report(prompt, model=ollama_model)
            if ollama_text:
                st.markdown("### Ollama 리포트")
                st.markdown(ollama_text)
            else:
                st.info("Ollama 호출에 실패했습니다. 로컬 Ollama 서버와 모델 설치 상태를 확인하세요.")

        st.subheader("과거 유사 리뷰")
        if similar.empty:
            st.info("선택한 약물의 과거 리뷰가 충분하지 않습니다.")
        else:
            display = similar[["drug_name", "condition", "rating", "risk_label", "useful_count", "review"]].copy()
            display["risk_label"] = display["risk_label"].map({0: "안전군", 1: "위험군"})
            st.dataframe(
                display,
                use_container_width=True,
                hide_index=True,
                column_config={
                    "rating": st.column_config.NumberColumn("평점", format="%.1f"),
                    "review": st.column_config.TextColumn("리뷰", width="large"),
                },
            )
