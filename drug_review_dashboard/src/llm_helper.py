from __future__ import annotations

import pandas as pd

from .features import extract_keywords


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


def try_ollama_report(prompt: str, model: str = "gemma3") -> str | None:
    try:
        import ollama

        response = ollama.chat(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a cautious medication-review assistant for a class project. Do not diagnose. Encourage urgent care for emergency symptoms.",
                },
                {"role": "user", "content": prompt},
            ],
        )
        return response["message"]["content"]
    except Exception:
        return None
