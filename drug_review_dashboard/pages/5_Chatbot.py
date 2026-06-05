from __future__ import annotations

import sys
from pathlib import Path

import plotly.express as px
import streamlit as st


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_cache import get_prepared_data
from src.features import extract_keywords
from src.llm_helper import (
    PROVIDER_OFFLINE,
    PROVIDER_OLLAMA,
    PROVIDER_OPENAI,
    answer_drug_question,
    build_drug_context,
    openai_available,
)


@st.cache_data(show_spinner=False)
def get_drug_view(max_rows: int, drug_name: str):
    """Per-drug grounding context + keyword chart data, cached by (rows, drug).

    Keeps each chat message instant: the drug subset + keyword extraction run
    once per drug instead of on every rerun.
    """
    df, _, _ = get_prepared_data(max_rows, False)
    drug_df = df[df["drug_name"] == drug_name]
    ctx = build_drug_context(drug_df, drug_name)
    risky = drug_df[drug_df["risk_label"] == 1]["review"]
    source_reviews = risky if not risky.empty else drug_df["review"]
    top_kw = extract_keywords(source_reviews, top_n=10)
    return ctx, top_kw


# Ollama-first priority chain (Ollama -> OpenAI backup -> rule-based).
PROVIDER_LABELS = {
    "로컬 Ollama 우선 (자동: Ollama→OpenAI→규칙)": PROVIDER_OLLAMA,
    "OpenAI 우선 (OpenAI→Ollama→규칙)": PROVIDER_OPENAI,
    "오프라인(규칙 기반)": PROVIDER_OFFLINE,
}


st.title("AI 상담 챗봇")
st.caption("선택한 약물의 리뷰 데이터에 근거해 부작용·주의사항을 질의응답합니다. (참고용, 의학적 진단 아님)")

with st.sidebar:
    max_rows = st.number_input("최대 로딩 행 수", min_value=500, max_value=200_000, value=50_000, step=5_000)
    st.markdown("---")
    st.subheader("답변 엔진")
    provider_label = st.radio("LLM 백엔드 선택", list(PROVIDER_LABELS.keys()), index=0)
    provider = PROVIDER_LABELS[provider_label]
    st.caption("기본은 로컬 Ollama 우선, 실패 시 OpenAI(키 있을 때) → 규칙 기반 순으로 자동 대체됩니다.")
    if openai_available():
        st.caption("✅ OPENAI_API_KEY 감지됨 (백업 사용 가능)")
    else:
        st.caption("⚠️ OPENAI_API_KEY 없음 — Ollama 실패 시 바로 규칙 기반으로 대체")
    ollama_model = st.text_input("Ollama 모델명", value="gemma3", disabled=provider == PROVIDER_OFFLINE)
    openai_model = st.text_input("OpenAI 모델명(백업)", value="gpt-4o-mini", disabled=provider == PROVIDER_OFFLINE)

df, source, _ = get_prepared_data(int(max_rows), False)


@st.cache_data(show_spinner=False)
def get_drug_options(max_rows: int):
    data, _, _ = get_prepared_data(max_rows, False)
    counts = data["drug_name"].value_counts()
    return counts[counts >= 5].index.tolist() or sorted(data["drug_name"].unique().tolist())


drug_options = get_drug_options(int(max_rows))
selected_drug = st.selectbox("상담할 약물 선택", drug_options, index=0)

ctx, top_kw = get_drug_view(int(max_rows), selected_drug)

# --- lightweight monitoring summary (problem def: monitoring + chatbot 탭) ---
m1, m2, m3 = st.columns(3)
m1.metric("리뷰 수", f"{ctx['reviews']:,}건")
m2.metric("평균 평점", f"{ctx['avg_rating']:.2f}" if ctx["avg_rating"] is not None else "N/A")
m3.metric("위험군 비율", f"{ctx['risk_ratio'] * 100:.1f}%" if ctx["risk_ratio"] is not None else "N/A")

with st.expander("이 약물의 모니터링 요약 (챗봇 근거 데이터)", expanded=False):
    if ctx["keywords"]:
        if not top_kw.empty:
            fig = px.bar(
                top_kw.sort_values("count"),
                x="count", y="keyword", orientation="h",
                color_discrete_sequence=["#dc2626"],
                labels={"count": "빈도", "keyword": "키워드"},
            )
            fig.update_layout(height=320, margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
    st.caption("가장 많이 보고된 부작용/증상 키워드는 챗봇 답변의 근거로 사용됩니다.")

st.divider()

# --- chat state, reset when drug changes ---
if st.session_state.get("chat_drug") != selected_drug:
    st.session_state["chat_drug"] = selected_drug
    st.session_state["chat_history"] = []

history = st.session_state["chat_history"]

st.markdown("#### 예시 질문")
example_qs = [
    f"{selected_drug}의 흔한 부작용은?",
    "복용 시 주의할 점이 있나요?",
    "환자 평점은 어떤가요?",
]
cols = st.columns(len(example_qs))
clicked = None
for col, q in zip(cols, example_qs):
    if col.button(q, use_container_width=True):
        clicked = q

# render history
for turn in history:
    with st.chat_message(turn["role"]):
        st.markdown(turn["content"])

user_input = st.chat_input("부작용·주의사항을 물어보세요")
question = clicked or user_input

if question:
    history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        with st.spinner("답변 생성 중..."):
            answer = answer_drug_question(
                question=question,
                ctx=ctx,
                history=history[:-1],
                provider=provider,
                openai_model=openai_model,
                ollama_model=ollama_model,
            )
        st.markdown(answer)
    history.append({"role": "assistant", "content": answer})
    st.session_state["chat_history"] = history

if history and st.button("대화 초기화"):
    st.session_state["chat_history"] = []
    st.rerun()
