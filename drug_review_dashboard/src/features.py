"""특성 엔지니어링 핵심 모듈 — EDA 발견을 모델 특성으로 변환한다.

[EDA 발견 → 특성 설계 흐름 요약]  (상세: docs/특성_엔지니어링.md)
  1. 평점이 J자형(10점·1점 양극단)            → rating, low_rating_flag, 약물별 IQR 이상치
  2. 위험군 리뷰가 길고 감정 표현이 강함        → review_length, word_count,
                                               exclamation_count, uppercase_ratio
  3. 위험/안전군의 어휘가 뚜렷이 갈림           → severe/symptom/positive_keyword_count,
                                               리뷰 원문 TF-IDF (modeling.py)
  4. 약물 3,671종 — 리뷰 수·평점 분포가 제각각  → drug_review_count, drug_avg_rating,
                                               약물별 IQR 기준 rating_iqr_low_outlier
  5. 정답(ADE) 라벨이 없음                     → 약한 라벨 build_target()
     단, 라벨을 정의한 특성은 학습에서 제외      → LABEL_DEFINING_FEATURES / MODEL_FEATURES 분리

원본 컬럼(7개): review_id, drug_name, condition, review, rating, date, useful_count
나머지 수치 특성은 전부 이 모듈에서 새로 만든(가공·결합·도출한) 특성이다.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


# 약한 라벨의 clause A를 구성하는 "심각(응급) 증상" 키워드 사전.
# 응급실(er/hospital/emergency), 호흡·심장(breathing/chest pain), 자살사고 등
# 즉시 조치가 필요한 신호만 모았다. 하나라도 등장하면 평점과 무관하게 위험군.
SEVERE_KEYWORDS = {
    "anaphylaxis",
    "breathing",
    "chest pain",
    "heart racing",
    "irregular heartbeat",
    "palpitation",
    "seizure",
    "suicidal",
    "suicide",
    "hallucination",
    "faint",
    "passed out",
    "emergency",
    "er",
    "hospital",
    "swelling",
    "throat",
    "rash",
    "fever",
    "confusion",
    "dehydration",
}

# 약한 라벨의 clause B에 쓰이는 "일반 부작용 증상" 키워드 사전.
# 메스꺼움·어지러움처럼 흔한 증상이라 단독으로는 위험이 아니고,
# "낮은 평점(rating ≤ 3)과 동시에" 등장할 때만 위험 신호로 본다.
SYMPTOM_KEYWORDS = {
    "nausea",
    "vomiting",
    "dizzy",
    "dizziness",
    "headache",
    "fatigue",
    "insomnia",
    "anxiety",
    "panic",
    "pain",
    "bleeding",
    "rash",
    "swelling",
    "diarrhea",
    "cramps",
    "blurred",
    "withdrawal",
}

# 긍정 어휘 사전 — 라벨 정의에는 쓰지 않는 "비누수" 특성용.
# EDA 키워드 비교에서 안전군 상위 단어(worked/helped/improved...)가 뚜렷해
# 모델이 안전군 방향의 신호로 활용할 수 있다 (positive_keyword_count).
POSITIVE_KEYWORDS = {
    "worked",
    "helped",
    "improved",
    "better",
    "stable",
    "relief",
    "effective",
    "great",
    "good",
    "benefit",
}

STOPWORDS = {
    # common English function words (so keyword charts surface real symptoms)
    "the", "and", "was", "had", "for", "but", "not", "you", "are", "she", "her",
    "his", "him", "its", "our", "out", "who", "how", "why", "all", "any", "can",
    "did", "has", "get", "got", "now", "one", "two", "too", "off", "per", "use",
    "used", "than", "them", "into", "over", "very", "just", "only", "more",
    "most", "some", "such", "even", "also", "much", "many", "back", "down", "still",
    "this", "that", "with", "from", "have", "after", "before", "about", "were",
    "been", "they", "then", "when", "will", "would", "could", "should", "there",
    "their", "what", "which", "while", "again", "ever", "every", "your", "yours",
    "mine", "myself", "because", "though", "since", "until", "being", "doing",
    "having", "does", "day", "days", "week", "weeks", "month", "months", "year",
    "years", "time", "times", "first", "last", "feel", "felt", "feels", "really",
    # domain-generic words that aren't symptoms
    "medicine", "medication", "drug", "drugs", "dose", "doses", "dosage", "taking",
    "took", "take", "takes", "pill", "pills", "mg", "started", "start", "stopped",
    # pronoun/verb contractions
    "i'm", "i've", "i'll", "i'd", "it's", "that's", "don't", "didn't", "doesn't",
    "can't", "wasn't", "isn't", "haven't", "hadn't", "won't", "they're", "you're",
}


# All engineered numeric columns add_features() produces. Used for EDA,
# correlation analysis, and the data-explorer page.
# [한국어] add_features()가 만드는 수치 컬럼 전체 목록 (rating·useful_count만 원본,
# 나머지 11개는 새로 만든 특성). EDA·상관분석·데이터 조회 페이지에서 사용한다.
NUMERIC_FEATURES = [
    "rating",
    "useful_count",
    "review_length",
    "word_count",
    "exclamation_count",
    "uppercase_ratio",
    "severe_keyword_count",
    "symptom_keyword_count",
    "positive_keyword_count",
    "low_rating_flag",
    "rating_iqr_low_outlier",
    "drug_review_count",
    "drug_avg_rating",
]

# Columns that the weak label build_target() is *defined from*. Feeding these to
# the classifier lets it reconstruct the label deterministically (target
# leakage), which is why early runs scored ~99% / AUC 1.0. They are excluded
# from the model so reported metrics reflect honest predictive skill.
#   label = (severe_keyword_count > 0) OR (rating <= 3 AND symptom_keyword_count > 0)
# [한국어·핵심] 아래 3개는 약한 라벨을 "정의"한 컬럼이다. 이걸 그대로 학습에 넣으면
# 모델이 라벨 생성 규칙을 암기해 Acc 99.6%/AUC 1.0이라는 가짜 성능이 나온다(타깃 누수).
# 그래서 학습 특성에서 완전히 제외하고, EDA/데이터 조회 화면에서만 보여준다.
LABEL_DEFINING_FEATURES = [
    "severe_keyword_count",
    "symptom_keyword_count",
    "low_rating_flag",
]

# Numeric features actually given to the model. `rating` is kept on purpose: it
# is a genuine user-provided signal and, on its own (without the symptom-keyword
# count), cannot reconstruct the label. The TF-IDF of the raw review text is
# added on top of these inside the modeling pipeline.
# [한국어] 모델에 실제로 들어가는 수치 특성 = 전체 − 라벨 정의 3개.
# rating은 사용자가 직접 준 신호이고, 증상 키워드 수 없이 단독으로는 라벨을
# 복원할 수 없으므로 의도적으로 유지했다 (보고서 3.3절 참고).
MODEL_FEATURES = [c for c in NUMERIC_FEATURES if c not in LABEL_DEFINING_FEATURES]


@dataclass(frozen=True)
class DrugStats:
    review_count: int
    avg_rating: float
    q1: float
    q3: float
    iqr: float
    low_outlier_cutoff: float


def normalize_text(text: str) -> str:
    text = str(text or "").lower()
    text = re.sub(r"\s+", " ", text)
    return text.strip()


@lru_cache(maxsize=None)
def _keyword_pattern(keywords: frozenset[str]) -> re.Pattern:
    """One compiled, word-boundaried alternation for a keyword set (cached).

    Longer phrases first so e.g. "chest pain" wins over "pain" in findall.
    """
    ordered = sorted(keywords, key=len, reverse=True)
    alternation = "|".join(re.escape(k) for k in ordered)
    return re.compile(r"\b(?:" + alternation + r")\b")


def count_keywords(text: str, keywords) -> int:
    """Number of *distinct* keywords from the set present in the text.

    Uses a single precompiled regex per set (instead of recompiling one regex
    per keyword per row), which is the dominant cost in add_features over tens
    of thousands of reviews.

    [한국어] 텍스트에 등장하는 키워드의 "서로 다른 종류 수"를 센다.
    키워드 세트당 정규식 1개를 미리 컴파일해 재사용(lru_cache)하므로,
    행마다 키워드별로 re.search를 도는 방식보다 5만 행 기준 약 4배 빠르다.
    단어 경계(\\b)를 쓰므로 "er"가 "better" 안에서 오탐되지 않는다.
    """
    if not keywords:
        return 0
    pattern = _keyword_pattern(frozenset(keywords))
    return len(set(pattern.findall(normalize_text(text))))


def build_target(df: pd.DataFrame) -> pd.Series:
    """Create a weak ADE-risk label from review text and rating.

    The source dataset has ratings but no official ADE severity label, so this
    project transparently defines a proxy target:
    - severe medical keyword appears, or
    - low rating plus at least one symptom keyword.

    [한국어] 약한 라벨(weak label) 생성 — 이 프로젝트의 target.
    원본에 정답 ADE 라벨이 없으므로, 교수님 피드백에 따라 투명한 규칙으로 정의한다:
      위험군(1) = (심각 키워드 등장) OR (평점 ≤ 3 AND 일반 증상 키워드 등장)
    전체 데이터 기준 clause A 9.6% + clause B 11.3% → 합집합 18.6%가 위험군이 된다.
    """
    text = df["review"].fillna("").astype(str)
    severe = text.map(lambda value: count_keywords(value, SEVERE_KEYWORDS))
    symptoms = text.map(lambda value: count_keywords(value, SYMPTOM_KEYWORDS))
    rating = pd.to_numeric(df["rating"], errors="coerce")

    return ((severe > 0) | ((rating <= 3) & (symptoms > 0))).astype(int)


def compute_drug_stats(df: pd.DataFrame) -> pd.DataFrame:
    """[한국어] 약물별 평점 통계(리뷰 수·평균·Q1·Q3·IQR·낮은 이상치 기준) 계산.

    EDA에서 약물마다 평점 분포가 크게 다름을 확인했기 때문에(Box Plot),
    "낮은 평점"의 기준을 전체 평균이 아니라 **약물별 IQR**(Q1 − 1.5×IQR)로 잡는다.
    리뷰 5건 미만 약물은 분위수가 불안정하므로 전체 분포 기준으로 보정한다.
    """
    base = df.dropna(subset=["rating"]).copy()
    if base.empty:
        return pd.DataFrame(
            columns=[
                "drug_name",
                "drug_review_count",
                "drug_avg_rating",
                "drug_q1",
                "drug_q3",
                "drug_iqr",
                "drug_low_outlier_cutoff",
            ]
        )

    grouped = base.groupby("drug_name")["rating"]
    stats = grouped.agg(
        drug_review_count="count",
        drug_avg_rating="mean",
        drug_q1=lambda s: s.quantile(0.25),
        drug_q3=lambda s: s.quantile(0.75),
    ).reset_index()
    stats["drug_iqr"] = stats["drug_q3"] - stats["drug_q1"]
    stats["drug_low_outlier_cutoff"] = stats["drug_q1"] - 1.5 * stats["drug_iqr"]

    global_q1 = base["rating"].quantile(0.25)
    global_q3 = base["rating"].quantile(0.75)
    global_iqr = global_q3 - global_q1
    sparse = stats["drug_review_count"] < 5
    stats.loc[sparse, "drug_q1"] = global_q1
    stats.loc[sparse, "drug_q3"] = global_q3
    stats.loc[sparse, "drug_iqr"] = global_iqr
    stats.loc[sparse, "drug_low_outlier_cutoff"] = global_q1 - 1.5 * global_iqr
    return stats


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """[한국어] 특성 엔지니어링 메인 함수 — 원본 7컬럼에서 13개 수치 특성 + 약한 라벨 생성.

    노트북(R&D)과 Streamlit 앱(Production)이 이 함수를 그대로 공유하므로
    (Single Source of Truth), 양쪽의 특성 값이 항상 동일하다.
    각 특성의 EDA 근거·계산식·예측 유용성은 docs/특성_엔지니어링.md 표 참고.
    """
    out = df.copy()
    stat_cols = [
        "drug_review_count",
        "drug_avg_rating",
        "drug_q1",
        "drug_q3",
        "drug_iqr",
        "drug_low_outlier_cutoff",
        "rating_iqr_low_outlier",
    ]
    out = out.drop(columns=[col for col in stat_cols if col in out.columns], errors="ignore")

    # ① 기본 정제: 결측 보정(평점은 중앙값, 공감 수는 0) — EDA의 결측 분석 반영
    out["review"] = out["review"].fillna("").astype(str)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(out["rating"].median())
    out["useful_count"] = pd.to_numeric(out["useful_count"], errors="coerce").fillna(0)

    # ② 언어/감정 신호 특성 — EDA에서 "위험군 리뷰가 길고 표현이 격하다"는 발견 반영
    out["review_length"] = out["review"].str.len()                      # 글자 수
    out["word_count"] = out["review"].str.findall(r"[A-Za-z']+").map(len)  # 단어 수
    out["exclamation_count"] = out["review"].str.count("!")             # 느낌표 = 감정 강도
    out["uppercase_ratio"] = out["review"].map(_uppercase_ratio)        # 대문자 비율(강조/분노)

    # ③ 키워드 사전 특성 — severe/symptom은 라벨 정의용(학습 제외), positive는 비누수 특성
    out["severe_keyword_count"] = out["review"].map(lambda value: count_keywords(value, SEVERE_KEYWORDS))
    out["symptom_keyword_count"] = out["review"].map(lambda value: count_keywords(value, SYMPTOM_KEYWORDS))
    out["positive_keyword_count"] = out["review"].map(lambda value: count_keywords(value, POSITIVE_KEYWORDS))

    # ④ 약한 라벨 생성 — low_rating_flag(평점≤3)와 키워드 수로 target 정의 (보고서 3.1절)
    out["low_rating_flag"] = (out["rating"] <= 3).astype(int)
    out["risk_label"] = build_target(out)

    # ⑤ 약물 단위 통계 특성 — EDA의 "약물 3,671종, 분포 제각각" 발견 반영.
    #    같은 2점이라도 평균 9점 약물에서는 이상치, 평균 3점 약물에서는 평범하므로
    #    약물별 IQR 기준으로 낮은 평점 이상치(rating_iqr_low_outlier)를 판정한다.
    stats = compute_drug_stats(out)
    out = out.merge(stats, how="left", on="drug_name")
    global_count = len(out)
    global_avg = float(out["rating"].mean()) if len(out) else 0.0
    out["drug_review_count"] = out["drug_review_count"].fillna(global_count)
    out["drug_avg_rating"] = out["drug_avg_rating"].fillna(global_avg)
    out["drug_low_outlier_cutoff"] = out["drug_low_outlier_cutoff"].fillna(
        out["rating"].quantile(0.25) - 1.5 * (out["rating"].quantile(0.75) - out["rating"].quantile(0.25))
    )
    out["rating_iqr_low_outlier"] = (out["rating"] < out["drug_low_outlier_cutoff"]).astype(int)

    for col in NUMERIC_FEATURES:
        out[col] = pd.to_numeric(out[col], errors="coerce").fillna(0)
    return out


def ensure_features(df: pd.DataFrame) -> pd.DataFrame:
    """Return df unchanged if it is already feature-engineered, else add features.

    Lets callers accept an already-prepared (cached) DataFrame without paying the
    cost of running add_features a second time.
    """
    return df if "risk_label" in df.columns else add_features(df)


def _uppercase_ratio(text: str) -> float:
    letters = re.findall(r"[A-Za-z]", str(text))
    if not letters:
        return 0.0
    upper = sum(1 for char in letters if char.isupper())
    return upper / len(letters)


def make_prediction_frame(
    base_df: pd.DataFrame,
    drug_name: str,
    review: str,
    rating: float,
    useful_count: float = 0,
) -> pd.DataFrame:
    """Build a single feature row for one user input.

    Only the user's row is feature-engineered (cheap); the drug-level statistics
    (review count, avg rating, IQR cutoff) are looked up from the base dataset so
    we never re-featurize the whole 50k+ row frame on each prediction click.

    [한국어] 서비스 페이지에서 사용자 입력 1건을 모델 입력 형태로 변환한다.
    텍스트 특성은 입력 행 1개만 계산(저렴)하고, 약물별 통계는 전체 데이터에서
    조회한다 — 클릭할 때마다 5만 행 전체를 다시 특성화하지 않기 위한 설계.
    데이터에 없는 새 약물이면 전체 분포 통계로 대체한다(콜드 스타트 처리).
    """
    drug_name = drug_name or "Unknown"
    row = pd.DataFrame(
        [
            {
                "review_id": "user-input",
                "drug_name": drug_name,
                "condition": "User input",
                "review": review or "",
                "rating": rating,
                "date": pd.Timestamp.today().normalize(),
                "useful_count": useful_count,
            }
        ]
    )
    featured = add_features(row)  # 1 row: text/linguistic features computed correctly

    # Drug-level stats must come from the full dataset, not the single row.
    base = ensure_features(base_df)
    stats = compute_drug_stats(base)
    match = stats[stats["drug_name"] == drug_name]
    if not match.empty:
        s = match.iloc[0]
        featured["drug_review_count"] = s["drug_review_count"]
        featured["drug_avg_rating"] = s["drug_avg_rating"]
        featured["drug_low_outlier_cutoff"] = s["drug_low_outlier_cutoff"]
    else:  # unseen drug -> fall back to global distribution
        q1 = base["rating"].quantile(0.25)
        q3 = base["rating"].quantile(0.75)
        featured["drug_review_count"] = float(len(base))
        featured["drug_avg_rating"] = float(base["rating"].mean()) if len(base) else 0.0
        featured["drug_low_outlier_cutoff"] = q1 - 1.5 * (q3 - q1)

    cutoff = float(featured["drug_low_outlier_cutoff"].iloc[0])
    featured["rating_iqr_low_outlier"] = int(float(featured["rating"].iloc[0]) < cutoff)

    for col in NUMERIC_FEATURES:
        featured[col] = pd.to_numeric(featured[col], errors="coerce").fillna(0)
    return featured.tail(1).reset_index(drop=True)


def get_drug_summary(df: pd.DataFrame, min_reviews: int = 5) -> pd.DataFrame:
    featured = ensure_features(df)
    summary = (
        featured.groupby("drug_name")
        .agg(
            reviews=("review", "count"),
            avg_rating=("rating", "mean"),
            risk_ratio=("risk_label", "mean"),
            avg_useful=("useful_count", "mean"),
            condition=("condition", lambda s: s.mode().iat[0] if not s.mode().empty else "Unknown"),
        )
        .reset_index()
    )
    summary = summary[summary["reviews"] >= min_reviews]
    summary["risk_ratio"] = summary["risk_ratio"] * 100
    return summary.sort_values(["risk_ratio", "reviews"], ascending=[False, False])


def extract_keywords(texts, top_n: int = 20) -> pd.DataFrame:
    words: list[str] = []
    for text in texts:
        tokens = re.findall(r"[A-Za-z][A-Za-z']{2,}", normalize_text(text))
        words.extend([token for token in tokens if token not in STOPWORDS])

    if not words:
        return pd.DataFrame(columns=["keyword", "count"])

    counts = pd.Series(words).value_counts().head(top_n)
    return counts.rename_axis("keyword").reset_index(name="count")
