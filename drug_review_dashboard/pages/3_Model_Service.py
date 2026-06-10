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

from src.data_cache import get_prepared_data
from src.features import make_prediction_frame
from src.llm_helper import (
    PROVIDER_OFFLINE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    build_rule_based_report,
    generate_report,
    openai_available,
    try_ollama_vision,
)
from src.modeling import predict_risk, train_risk_model

# Ollama-first priority chain (Ollama -> OpenAI backup -> rule-based).
PROVIDER_LABELS = {
    "로컬 Ollama 우선 (자동: Ollama→OpenAI→규칙)": PROVIDER_OLLAMA,
    "OpenAI 우선 (OpenAI→Ollama→규칙)": PROVIDER_OPENAI,
    "오프라인(규칙 기반)": PROVIDER_OFFLINE,
}


@st.cache_resource(show_spinner="RandomForest 모델 학습 중... (최초 1회만 실행)")
def get_model(max_rows: int, sample_size: int):
    # Keyed on scalar params (not the DataFrame) so Streamlit doesn't re-hash
    # 50k+ rows on every rerun. Loads the shared cached data internally.
    df, _, _ = get_prepared_data(max_rows, False)
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
    st.markdown("---")
    st.subheader("LLM 리포트 엔진")
    provider_label = st.radio("백엔드 선택", list(PROVIDER_LABELS.keys()), index=0)
    report_provider = PROVIDER_LABELS[provider_label]
    st.caption("기본은 로컬 Ollama 우선, 실패 시 OpenAI(키 있을 때) → 규칙 기반 순으로 자동 대체됩니다.")
    if openai_available():
        st.caption("✅ OPENAI_API_KEY 감지됨 (백업 사용 가능)")
    else:
        st.caption("⚠️ OPENAI_API_KEY 없음 — Ollama 실패 시 바로 규칙 기반으로 대체")
    ollama_model = st.text_input("Ollama 모델명", value="gemma3", disabled=report_provider == PROVIDER_OFFLINE)
    openai_model = st.text_input("OpenAI 모델명(백업)", value="gpt-4o-mini", disabled=report_provider == PROVIDER_OFFLINE)

df, source, _ = get_prepared_data(int(max_rows), False)
bundle = get_model(int(max_rows), int(sample_size))

metric_cols = st.columns(5)
metric_cols[0].metric("학습 데이터", f"{bundle.train_rows:,}건")
metric_cols[1].metric("검증 데이터", f"{bundle.test_rows:,}건")
metric_cols[2].metric("Accuracy", f"{bundle.metrics['accuracy'] * 100:.1f}%")
metric_cols[3].metric("Recall", f"{bundle.metrics['recall'] * 100:.1f}%")
metric_cols[4].metric("F1", f"{bundle.metrics['f1'] * 100:.1f}%")

st.caption(
    "지표는 **누수(leakage) 제거** 평가입니다. 라벨을 정의하는 키워드 카운트(severe/symptom)와 "
    "low_rating_flag는 학습 특성에서 제외했고, 모델은 리뷰 원문(TF-IDF)과 비누수 특성만으로 위험군을 예측합니다."
)

with st.expander("모델 성능 · 비교 · 특성 중요도 보기", expanded=False):
    if bundle.comparison is not None:
        st.markdown("**모델 비교 (검증셋, 누수 제거)** — 단순 규칙(rating≤3) 대비 성능 향상 확인")
        comp = bundle.comparison.copy()
        for c in ["accuracy", "precision", "recall", "f1", "roc_auc"]:
            comp[c] = (comp[c] * 100).round(1)
        st.dataframe(
            comp,
            width="stretch",
            hide_index=True,
            column_config={
                "model": "모델",
                "accuracy": st.column_config.NumberColumn("Acc(%)", format="%.1f"),
                "precision": st.column_config.NumberColumn("Prec(%)", format="%.1f"),
                "recall": st.column_config.NumberColumn("Recall(%)", format="%.1f"),
                "f1": st.column_config.NumberColumn("F1(%)", format="%.1f"),
                "roc_auc": st.column_config.NumberColumn("AUC(%)", format="%.1f"),
            },
        )

    col1, col2 = st.columns([1, 1.2])
    with col1:
        cm = bundle.metrics["confusion_matrix"]
        cm_df = pd.DataFrame(cm, index=["실제 안전군", "실제 위험군"], columns=["예측 안전군", "예측 위험군"])
        fig = px.imshow(cm_df, text_auto=True, color_continuous_scale="Blues", labels={"color": "건수"})
        fig.update_layout(title=f"혼동행렬 (배포 모델: {bundle.model_name})")
        st.plotly_chart(fig, width="stretch")
    with col2:
        fig = px.bar(
            bundle.feature_importance.sort_values("importance"),
            x="importance",
            y="feature",
            orientation="h",
            labels={"importance": "중요도", "feature": "특성"},
        )
        st.plotly_chart(fig, width="stretch")
    st.caption(
        "해석: 혼동행렬에서 놓친 위험군(False Negative)이 주된 오류이며, 누수 제거 후 상위 특성은 "
        "rating과 hospital·er·swelling 같은 의미 있는 위험 토큰입니다 — 모델이 규칙 암기가 아니라 텍스트 신호를 학습했다는 근거입니다."
    )

st.divider()

drug_options = sorted(df["drug_name"].dropna().astype(str).unique().tolist())
default_drug = drug_options[0] if drug_options else "Unknown"

left, right = st.columns([0.95, 1.05])

with left:
    st.subheader("사용자 입력")
    uploaded = st.file_uploader("약통 이미지 업로드", type=["png", "jpg", "jpeg"])
    use_vision = st.toggle("이미지 멀티모달 인식(Ollama) 시도", value=False,
                           help="로컬 Ollama의 비전 모델(gemma3 등)로 약통 글자를 읽어 약물명을 매칭합니다. 실패 시 파일명 기반으로 대체합니다.")
    guessed = None
    if uploaded is not None:
        image_bytes = uploaded.getvalue()
        image = Image.open(uploaded)
        st.image(image, caption="업로드한 이미지", width="stretch")

        if use_vision:
            with st.spinner("Ollama 비전 모델로 약통을 인식하는 중..."):
                vision = try_ollama_vision(image_bytes, drug_options, model="gemma3")
            if vision is None:
                st.caption("Ollama 비전 호출 실패 → 파일명 기반 매칭으로 대체합니다.")
            else:
                guessed = vision.get("matched")
                with st.popover("모델이 읽은 내용 보기"):
                    st.text(vision.get("raw", ""))
                if guessed:
                    st.success(f"멀티모달 인식 결과 약물 후보: {guessed}")
                else:
                    st.caption("이미지에서 데이터셋 약물과 일치하는 이름을 찾지 못했습니다.")

        if guessed is None:
            guessed = guess_drug_from_filename(uploaded.name, drug_options)
            if guessed:
                st.caption(f"파일명에서 약물 후보를 찾았습니다: {guessed}")
            elif not use_vision:
                st.caption("이미지 미리보기 + 파일명 기반 후보 매칭을 지원합니다. (멀티모달은 위 토글)")

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
    submitted = st.button("위험도 예측", type="primary", width="stretch")

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

        if report_provider != PROVIDER_OFFLINE:
            prompt = f"""
약물명: {manual_drug}
사용자 증상/리뷰: {review_text}
모델 위험도: {probability * 100:.1f}%
IQR 이상치 여부: {iqr_flag}

의학적 진단은 피하고, 수업용 서비스 리포트 형식으로 위험 신호와 다음 행동을 한국어로 정리해줘.
"""
            with st.spinner("LLM 심층 리포트 생성 중..."):
                llm_text = generate_report(
                    prompt,
                    provider=report_provider,
                    openai_model=openai_model,
                    ollama_model=ollama_model,
                )
            if llm_text:
                st.markdown(f"### LLM 심층 리포트 ({provider_label})")
                st.markdown(llm_text)
            else:
                st.info("선택한 LLM 호출에 실패했습니다. OpenAI 키 또는 로컬 Ollama 상태를 확인하세요. (위 규칙 기반 리포트는 항상 제공됩니다.)")

        st.subheader("과거 유사 리뷰")
        if similar.empty:
            st.info("선택한 약물의 과거 리뷰가 충분하지 않습니다.")
        else:
            display = similar[["drug_name", "condition", "rating", "risk_label", "useful_count", "review"]].copy()
            display["risk_label"] = display["risk_label"].map({0: "안전군", 1: "위험군"})
            st.dataframe(
                display,
                width="stretch",
                hide_index=True,
                column_config={
                    "rating": st.column_config.NumberColumn("평점", format="%.1f"),
                    "review": st.column_config.TextColumn("리뷰", width="large"),
                },
            )
