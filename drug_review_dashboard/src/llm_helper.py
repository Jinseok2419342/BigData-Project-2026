from __future__ import annotations

import os

import pandas as pd

from .features import extract_keywords


# --- LLM provider routing -------------------------------------------------
PROVIDER_OPENAI = "openai"
PROVIDER_OLLAMA = "ollama"
PROVIDER_OFFLINE = "offline"

DEFAULT_SYSTEM_PROMPT = (
    "You are a cautious medication-review assistant for a class project. "
    "Answer only from the provided review data, do not diagnose, and encourage "
    "urgent care for emergency symptoms."
)
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"
DEFAULT_OLLAMA_MODEL = "gemma3"
_API_KEY_PLACEHOLDER = "your_api_key_here"

_ENV_LOADED = False


def _load_env_once() -> None:
    """Load variables from a nearby .env file exactly once (best effort)."""
    global _ENV_LOADED
    if _ENV_LOADED:
        return
    _ENV_LOADED = True
    try:
        from dotenv import find_dotenv, load_dotenv

        load_dotenv(find_dotenv(usecwd=True))
    except Exception:
        pass


def get_openai_api_key() -> str | None:
    """Resolve the OpenAI API key from streamlit secrets, then .env / environment.

    Returns ``None`` when missing or still set to the placeholder, so callers can
    treat "no key" uniformly.
    """
    _load_env_once()
    key = None
    try:  # streamlit secrets (only if a secrets file exists)
        import streamlit as st

        key = st.secrets.get("OPENAI_API_KEY")  # type: ignore[attr-defined]
    except Exception:
        key = None
    if not key:
        key = os.environ.get("OPENAI_API_KEY")
    if not key or str(key).strip() in ("", _API_KEY_PLACEHOLDER):
        return None
    return str(key).strip()


def openai_available() -> bool:
    return get_openai_api_key() is not None


def _openai_chat(prompt: str, system: str, model: str, api_key: str) -> str | None:
    try:
        from openai import OpenAI

        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def _ollama_chat(prompt: str, system: str, model: str) -> str | None:
    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception:
        return None


def route_chat(
    prompt: str,
    provider: str = PROVIDER_OLLAMA,
    *,
    system: str = DEFAULT_SYSTEM_PROMPT,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    api_key: str | None = None,
) -> str | None:
    """Route a single-prompt chat request following an **Ollama-first** chain.

    Priority / fallback:
      1. Local Ollama (gemma3) is always tried first.
      2. OpenAI API (gpt-4o-mini) is the backup when a key is available — used
         if Ollama is off/unavailable/failing.
      3. Offline rule-based (caller handles): returned as ``None`` here.

    Provider selection only changes which engine *leads*:
      - provider == ollama (default): Ollama -> OpenAI backup.
      - provider == openai: OpenAI -> Ollama backup (explicit OpenAI-first).
      - provider == offline: skip all LLMs (returns None).

    [한국어] LLM 라우팅의 단일 관문 — 앱의 모든 LLM 호출이 이 함수를 거친다.
    우선순위: 로컬 Ollama(gemma3, 무료·오프라인) → OpenAI(gpt-4o-mini, 키 있을 때만)
    → None 반환(호출부가 규칙 기반 답변으로 대체). 각 단계는 예외를 삼키고
    다음 단계로 넘어가므로, 모델이 꺼져 있거나 키가 없어도 앱은 절대 죽지 않는다.
    이 폴백 체인은 tests/test_llm_routing.py의 18개 단위 테스트로 검증된다.
    """
    if provider == PROVIDER_OFFLINE:
        return None

    # Ollama-first by default; only an explicit OpenAI choice leads with OpenAI.
    if provider == PROVIDER_OPENAI:
        order = [PROVIDER_OPENAI, PROVIDER_OLLAMA]
    else:
        order = [PROVIDER_OLLAMA, PROVIDER_OPENAI]

    for p in order:
        if p == PROVIDER_OLLAMA:
            text = _ollama_chat(prompt, system, ollama_model)
            if text:
                return text
        elif p == PROVIDER_OPENAI:
            key = api_key or get_openai_api_key()
            if not key:
                continue
            text = _openai_chat(prompt, system, openai_model, key)
            if text:
                return text
    return None


def build_rule_based_report(
    drug_name: str,
    review: str,
    probability: float,
    iqr_flag: bool,
    similar_cases: pd.DataFrame,
) -> str:
    risk_level = "높음" if probability >= 0.7 else "중간" if probability >= 0.4 else "낮음"
    keywords = extract_keywords([review], top_n=6)
    keyword_text = ", ".join(keywords["keyword"].tolist()) if not keywords.empty else "뚜렷한 키워드 없음"
    case_count = len(similar_cases)
    avg_rating = similar_cases["rating"].mean() if case_count else None
    avg_rating_text = f"{avg_rating:.1f}점" if avg_rating is not None else "비교 데이터 부족"

    iqr_text = (
        "입력 평점이 해당 약물의 기존 평점 분포에서 낮은 이상치로 감지되었습니다."
        if iqr_flag
        else "입력 평점은 해당 약물의 기존 평점 분포에서 뚜렷한 낮은 이상치로 보이지 않습니다."
    )

    return f"""
### AI 상담 리포트 초안

- 예측 위험도: **{probability * 100:.1f}% ({risk_level})**
- 입력 증상 핵심어: **{keyword_text}**
- 과거 유사 약물 리뷰 수: **{case_count:,}건**
- 유사 약물 평균 평점: **{avg_rating_text}**
- IQR 통계 판단: {iqr_text}

**해석**  
모델은 리뷰 문장의 심각 증상 표현, 낮은 평점, 약물별 평점 이상치, 과거 리뷰 패턴을 함께 사용해 위험도를 계산했습니다.
위험도가 높게 나오면 복용 중단 여부를 스스로 판단하기보다 의료진이나 약사에게 즉시 상담하는 흐름으로 연결하는 것이 이 서비스의 목적입니다.

**주의**  
이 결과는 수업용 데이터 분석 서비스의 예측값이며 의학적 진단이 아닙니다.
호흡곤란, 흉통, 얼굴/목 부종, 자살사고, 발작처럼 긴급 신호가 있으면 실제 의료기관 또는 응급 서비스를 우선해야 합니다.
"""


def try_ollama_report(prompt: str, model: str = DEFAULT_OLLAMA_MODEL) -> str | None:
    """Backward-compatible Ollama-only report helper (delegates to _ollama_chat)."""
    return _ollama_chat(prompt, DEFAULT_SYSTEM_PROMPT, model)


def generate_report(
    prompt: str,
    provider: str = PROVIDER_OLLAMA,
    *,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    api_key: str | None = None,
) -> str | None:
    """Generate an LLM consultation report via the selected provider.

    Returns the report text, or ``None`` when offline / no LLM is reachable
    (the rule-based report from build_rule_based_report is always shown anyway).
    """
    return route_chat(
        prompt,
        provider,
        openai_model=openai_model,
        ollama_model=ollama_model,
        api_key=api_key,
    )


def build_drug_context(drug_df: pd.DataFrame, drug_name: str, max_examples: int = 3) -> dict:
    """Summarize one drug's review history into a grounding context for the chatbot.

    Returns numeric stats, the most-reported side-effect keywords (from risky
    reviews), and a few short example reviews. Everything is derived from the
    dataset so the chatbot's answers stay grounded in real reviews.
    """
    if drug_df.empty:
        return {
            "drug_name": drug_name,
            "reviews": 0,
            "avg_rating": None,
            "risk_ratio": None,
            "keywords": [],
            "examples": [],
        }

    risky = drug_df[drug_df.get("risk_label", 0) == 1]
    keywords = extract_keywords(risky["review"], top_n=8) if not risky.empty else extract_keywords(drug_df["review"], top_n=8)
    examples = (
        risky.sort_values("useful_count", ascending=False)["review"].head(max_examples).tolist()
        if not risky.empty
        else drug_df["review"].head(max_examples).tolist()
    )
    return {
        "drug_name": drug_name,
        "reviews": int(len(drug_df)),
        "avg_rating": float(drug_df["rating"].mean()) if "rating" in drug_df else None,
        "risk_ratio": float(drug_df["risk_label"].mean()) if "risk_label" in drug_df else None,
        "keywords": keywords["keyword"].tolist() if not keywords.empty else [],
        "examples": [str(e)[:300] for e in examples],
    }


def _context_block(ctx: dict) -> str:
    avg = f"{ctx['avg_rating']:.1f}" if ctx["avg_rating"] is not None else "N/A"
    risk = f"{ctx['risk_ratio'] * 100:.1f}%" if ctx["risk_ratio"] is not None else "N/A"
    kw = ", ".join(ctx["keywords"]) if ctx["keywords"] else "특이 키워드 없음"
    examples = "\n".join(f"  - {e}" for e in ctx["examples"]) if ctx["examples"] else "  - (예시 없음)"
    return (
        f"약물명: {ctx['drug_name']}\n"
        f"리뷰 수: {ctx['reviews']}건 / 평균 평점: {avg} / 위험군 비율: {risk}\n"
        f"가장 많이 보고된 부작용/증상 키워드: {kw}\n"
        f"대표 위험 리뷰 발췌:\n{examples}"
    )


def answer_drug_question(
    question: str,
    ctx: dict,
    history: list[dict] | None = None,
    provider: str = PROVIDER_OFFLINE,
    *,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    api_key: str | None = None,
) -> str:
    """Answer a question about a drug's cautions / side-effect history.

    Routes to the selected LLM provider (OpenAI -> Ollama priority) via
    route_chat(); if every LLM path is unavailable or fails, returns a
    rule-based answer built from the grounded dataset context so the chatbot
    always responds (offline-safe for demos).
    """
    if provider != PROVIDER_OFFLINE:
        history_text = ""
        for turn in (history or [])[-4:]:
            role = "사용자" if turn.get("role") == "user" else "상담봇"
            history_text += f"{role}: {turn.get('content', '')}\n"
        prompt = (
            "아래는 한 약물의 환자 리뷰 데이터 요약이다. 이 데이터에 근거해서만 한국어로 "
            "간결히 답하라. 데이터에 없는 내용은 추측하지 말고 모른다고 말하라. 의학적 진단은 "
            "하지 말고, 긴급 증상은 의료기관 상담을 권하라.\n\n"
            f"[약물 데이터 요약]\n{_context_block(ctx)}\n\n"
            f"[이전 대화]\n{history_text or '(없음)'}\n"
            f"[사용자 질문]\n{question}"
        )
        answer = route_chat(
            prompt,
            provider,
            openai_model=openai_model,
            ollama_model=ollama_model,
            api_key=api_key,
        )
        if answer:
            return answer
        # fall through to rule-based if the selected LLM is unavailable

    return _rule_based_answer(question, ctx)


def _rule_based_answer(question: str, ctx: dict) -> str:
    if ctx["reviews"] == 0:
        return (
            f"**{ctx['drug_name']}** 에 대한 리뷰 데이터가 충분하지 않아 데이터 기반 답변이 어렵습니다. "
            "다른 약물을 선택하거나 약물명을 확인해 주세요."
        )

    avg = f"{ctx['avg_rating']:.1f}점" if ctx["avg_rating"] is not None else "N/A"
    risk = f"{ctx['risk_ratio'] * 100:.1f}%" if ctx["risk_ratio"] is not None else "N/A"
    kw = ", ".join(ctx["keywords"][:6]) if ctx["keywords"] else "뚜렷한 키워드 없음"
    example = ctx["examples"][0] if ctx["examples"] else None

    # [한국어] 질문 의도를 간단한 키워드 매칭으로 분류해 통계 기반 답변을 고른다.
    # 영어 토큰도 잡히도록 소문자로 정규화한 q를 사용한다(한글은 영향 없음).
    q = (question or "").lower()
    if any(token in q for token in ["부작용", "증상", "side", "효과"]):
        focus = f"환자들이 가장 많이 보고한 부작용/증상 키워드는 **{kw}** 입니다."
    elif any(token in q for token in ["주의", "조심", "위험", "caution", "warn"]):
        focus = f"이 약물의 위험군 리뷰 비율은 **{risk}** 이며, 자주 등장하는 위험 신호 키워드는 **{kw}** 입니다."
    elif any(token in q for token in ["평점", "효능", "좋", "rating", "효과있"]):
        focus = f"전체 리뷰 평균 평점은 **{avg}** 입니다(10점 만점)."
    else:
        focus = f"평균 평점 **{avg}**, 위험군 비율 **{risk}**, 주요 키워드 **{kw}** 로 요약됩니다."

    example_text = f"\n\n> 대표 위험 리뷰: \"{example}\"" if example else ""
    return (
        f"**{ctx['drug_name']}** 리뷰 데이터({ctx['reviews']:,}건) 기준으로 답변드립니다.\n\n"
        f"{focus}{example_text}\n\n"
        "※ 이 답변은 리뷰 데이터 통계에 기반한 참고용이며 의학적 진단이 아닙니다. "
        "긴급 증상(호흡곤란·흉통·부종·자살사고·발작 등)이 있으면 즉시 의료기관에 상담하세요."
    )


def _vision_prompt(options: list[str]) -> str:
    # [한국어] 비전 모델 공용 프롬프트 — 이미지에서 약물명을 읽고,
    # 데이터셋에 존재하는 약물 목록과 정확히 일치하는 이름을 고르게 한다.
    option_hint = ", ".join(options[:60]) if options else "(no list provided)"
    return (
        "You are reading a photo of a medicine bottle or pill packaging for a "
        "class project. Identify the drug/brand name visible in the image. "
        "Then, if it matches one of these known drug names, return that exact "
        f"name; otherwise say UNKNOWN.\nKnown names: {option_hint}\n"
        "Answer on two lines:\nNAME: <drug name you read or UNKNOWN>\n"
        "MATCH: <one exact name from the list, or NONE>"
    )


def _match_from_vision_text(text: str, options: list[str]) -> str | None:
    # [한국어] 모델 답변에서 "MATCH:" 줄을 우선 파싱하고,
    # 실패하면 답변 전체에서 약물명 부분 문자열을 탐색한다(관대한 매칭).
    matched = None
    lower_opts = {opt.lower(): opt for opt in options}
    for line in text.splitlines():
        if line.upper().startswith("MATCH:"):
            candidate = line.split(":", 1)[1].strip()
            matched = lower_opts.get(candidate.lower())
    if matched is None:
        low = text.lower()
        for opt in options:
            if opt.lower() in low:
                matched = opt
                break
    return matched


def _ollama_vision(image_bytes: bytes, prompt: str, model: str) -> str | None:
    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[{"role": "user", "content": prompt, "images": [image_bytes]}],
        )
        return response["message"]["content"]
    except Exception:
        return None


def _openai_vision(image_bytes: bytes, prompt: str, model: str, api_key: str) -> str | None:
    # [한국어] OpenAI 멀티모달 호출 — 이미지를 base64 data URL로 전달한다.
    # gpt-4o-mini 등 비전 지원 모델이면 동작하며, 실패 시 None(폴백 계속).
    try:
        import base64

        from openai import OpenAI

        media = "image/png" if image_bytes[:4] == b"\x89PNG" else "image/jpeg"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        client = OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:{media};base64,{b64}"}},
                    ],
                }
            ],
            temperature=0,
        )
        return response.choices[0].message.content
    except Exception:
        return None


def recognize_drug_image(
    image_bytes: bytes,
    options: list[str],
    provider: str = PROVIDER_OLLAMA,
    *,
    ollama_model: str = DEFAULT_OLLAMA_MODEL,
    openai_model: str = DEFAULT_OPENAI_MODEL,
    api_key: str | None = None,
) -> dict | None:
    """[한국어] 약통 이미지 → 약물명 인식 라우터 (route_chat과 동일한 폴백 철학).

    우선순위: 선택 엔진에 따라 Ollama 비전 ↔ OpenAI 비전(gpt-4o-mini, 키 있을 때)
    순서로 시도하고, 둘 다 실패하면 None을 반환한다 — 호출부(서비스 페이지)는
    파일명 기반 매칭으로 폴백하므로 어떤 환경에서도 화면이 깨지지 않는다.
    반환: {"matched": 목록과 일치한 약물명 | None, "raw": 모델 원문, "engine": 사용 엔진}
    """
    if provider == PROVIDER_OFFLINE:
        return None

    prompt = _vision_prompt(options)
    order = [PROVIDER_OPENAI, PROVIDER_OLLAMA] if provider == PROVIDER_OPENAI else [PROVIDER_OLLAMA, PROVIDER_OPENAI]

    for p in order:
        text = None
        if p == PROVIDER_OLLAMA:
            text = _ollama_vision(image_bytes, prompt, ollama_model)
        elif p == PROVIDER_OPENAI:
            key = api_key or get_openai_api_key()
            if not key:
                continue
            text = _openai_vision(image_bytes, prompt, openai_model, key)
        if text:
            return {"matched": _match_from_vision_text(text, options), "raw": text, "engine": p}
    return None


def try_ollama_vision(image_bytes: bytes, options: list[str], model: str = "gemma3") -> dict | None:
    """Multimodal pill-bottle recognition via a local Ollama vision model.

    Sends the uploaded image to a vision-capable model (e.g. gemma3) and asks it
    to read any visible drug name and match it to the closest dataset option.
    Returns ``{"matched": <option|None>, "raw": <model text>}`` or ``None`` if
    Ollama / the model is unavailable. The filename heuristic is the fallback.

    [한국어] 하위 호환용 Ollama 전용 헬퍼 — 신규 코드는 recognize_drug_image 사용.
    """
    text = _ollama_vision(image_bytes, _vision_prompt(options), model)
    if not text:
        return None
    return {"matched": _match_from_vision_text(text, options), "raw": text}
