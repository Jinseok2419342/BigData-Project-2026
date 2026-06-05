from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import pandas as pd


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
LABEL_DEFINING_FEATURES = [
    "severe_keyword_count",
    "symptom_keyword_count",
    "low_rating_flag",
]

# Numeric features actually given to the model. `rating` is kept on purpose: it
# is a genuine user-provided signal and, on its own (without the symptom-keyword
# count), cannot reconstruct the label. The TF-IDF of the raw review text is
# added on top of these inside the modeling pipeline.
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
    """
    text = df["review"].fillna("").astype(str)
    severe = text.map(lambda value: count_keywords(value, SEVERE_KEYWORDS))
    symptoms = text.map(lambda value: count_keywords(value, SYMPTOM_KEYWORDS))
    rating = pd.to_numeric(df["rating"], errors="coerce")

    return ((severe > 0) | ((rating <= 3) & (symptoms > 0))).astype(int)


def compute_drug_stats(df: pd.DataFrame) -> pd.DataFrame:
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

    out["review"] = out["review"].fillna("").astype(str)
    out["rating"] = pd.to_numeric(out["rating"], errors="coerce").fillna(out["rating"].median())
    out["useful_count"] = pd.to_numeric(out["useful_count"], errors="coerce").fillna(0)
    out["review_length"] = out["review"].str.len()
    out["word_count"] = out["review"].str.findall(r"[A-Za-z']+").map(len)
    out["exclamation_count"] = out["review"].str.count("!")
    out["uppercase_ratio"] = out["review"].map(_uppercase_ratio)
    out["severe_keyword_count"] = out["review"].map(lambda value: count_keywords(value, SEVERE_KEYWORDS))
    out["symptom_keyword_count"] = out["review"].map(lambda value: count_keywords(value, SYMPTOM_KEYWORDS))
    out["positive_keyword_count"] = out["review"].map(lambda value: count_keywords(value, POSITIVE_KEYWORDS))
    out["low_rating_flag"] = (out["rating"] <= 3).astype(int)
    out["risk_label"] = build_target(out)

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
