"""Standalone, documented EDA for the UCI Drug Review project.

Run with the project venv:

    ../venv/Scripts/python.exe eda_report.py

It loads the full dataset (local CSV -> KaggleHub fallback), prints a text
summary, and saves figures to ``eda_outputs/``. The findings here feed the
report (보고서.md) and the in-app EDA page. Figures use matplotlib/seaborn so
they can be embedded as static PNGs in the report.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.data_loader import load_drug_reviews
from src.features import (
    SEVERE_KEYWORDS,
    SYMPTOM_KEYWORDS,
    add_features,
    build_target,
    extract_keywords,
)

sys.stdout.reconfigure(encoding="utf-8")
sns.set_theme(style="whitegrid")
OUT = ROOT / "eda_outputs"
OUT.mkdir(exist_ok=True)


def section(title: str) -> None:
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)


def main(max_rows: int | None = None) -> None:
    section("1. LOAD")
    df, source = load_drug_reviews(max_rows=max_rows)
    print(f"source={source}  shape={df.shape}")
    print("columns:", list(df.columns))

    section("2. SCHEMA / MISSINGNESS (post-standardization)")
    info = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing": df.isna().sum(),
            "missing_%": (df.isna().mean() * 100).round(2),
            "n_unique": df.nunique(dropna=True),
        }
    )
    print(info)

    section("3. RATING DISTRIBUTION")
    print(df["rating"].describe().round(2))
    vc = df["rating"].value_counts().sort_index()
    print("\nrating value counts:\n", vc)
    print("\n>>> insight: ratings are strongly bimodal/J-shaped (10 and 1 dominate).")

    section("4. usefulCount")
    print(df["useful_count"].describe().round(2))
    print("skew:", round(df["useful_count"].skew(), 2), "(heavy right tail)")

    section("5. TOP DRUGS / CONDITIONS")
    print("top drugs:\n", df["drug_name"].value_counts().head(10))
    print("\ntop conditions:\n", df["condition"].value_counts().head(10))
    weird = df["condition"].astype(str).str.contains("</span>", na=False).sum()
    print(f"\n>>> data-quality note: condition contains HTML junk like "
          f"'</span> users found this comment helpful' in {weird:,} rows.")

    section("6. WEAK ADE LABEL")
    feat = add_features(df)
    rate = feat["risk_label"].mean()
    print(f"risk_label positive rate = {rate:.3f}")
    print(feat.groupby("risk_label")["rating"].describe().round(2))

    # How much of the label is driven by each clause (leakage diagnosis)
    text = df["review"].fillna("").astype(str)
    from src.features import count_keywords

    severe = text.map(lambda v: count_keywords(v, SEVERE_KEYWORDS)) > 0
    symptom = text.map(lambda v: count_keywords(v, SYMPTOM_KEYWORDS)) > 0
    low = pd.to_numeric(df["rating"], errors="coerce") <= 3
    label = build_target(df).astype(bool)
    print(f"\nclause A (severe keyword present)         : {severe.mean():.3f}")
    print(f"clause B (rating<=3 AND symptom keyword)  : {(low & symptom).mean():.3f}")
    print(f"label = A or B                            : {label.mean():.3f}")
    print(f"label fully reconstructable from A,B      : {(label == (severe | (low & symptom))).mean():.3f}")
    print(">>> LEAKAGE: feeding severe/symptom counts + low_rating_flag lets a model "
          "reconstruct the label exactly. Those columns must be excluded from training.")

    section("7. CORRELATIONS (engineered numeric features)")
    num_cols = [
        "rating", "useful_count", "review_length", "word_count",
        "exclamation_count", "uppercase_ratio", "severe_keyword_count",
        "symptom_keyword_count", "positive_keyword_count", "risk_label",
    ]
    corr = feat[num_cols].corr(numeric_only=True)
    print(corr["risk_label"].sort_values(ascending=False).round(3))

    # ---- FIGURES ----
    section("8. SAVING FIGURES -> eda_outputs/")

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.countplot(x="rating", data=df, color="#2a9d8f", ax=ax)
    ax.set_title("Rating distribution (J-shaped)")
    fig.tight_layout(); fig.savefig(OUT / "01_rating_dist.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    feat["review_length"].clip(upper=2000).hist(bins=60, color="#264653", ax=ax)
    ax.set_title("Review length distribution (chars, clipped at 2000)")
    fig.tight_layout(); fig.savefig(OUT / "02_review_length.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(5, 4))
    feat["risk_label"].map({0: "safe(0)", 1: "risk(1)"}).value_counts().plot.bar(
        color=["#2a9d8f", "#e76f51"], ax=ax)
    ax.set_title(f"Weak ADE label balance (pos={rate:.1%})")
    ax.tick_params(axis="x", rotation=0)
    fig.tight_layout(); fig.savefig(OUT / "03_label_balance.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="vlag", center=0, ax=ax)
    ax.set_title("Feature correlation (note rating/keyword leakage into label)")
    fig.tight_layout(); fig.savefig(OUT / "04_corr_heatmap.png", dpi=110); plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.boxplot(x="risk_label", y="rating", hue="risk_label", data=feat,
                palette={0: "#2a9d8f", 1: "#e76f51"}, legend=False, ax=ax)
    ax.set_xticks([0, 1]); ax.set_xticklabels(["safe(0)", "risk(1)"])
    ax.set_title("Rating by risk label")
    fig.tight_layout(); fig.savefig(OUT / "05_rating_by_label.png", dpi=110); plt.close(fig)

    # time trend
    dated = feat.dropna(subset=["date"]).copy()
    if not dated.empty:
        dated["month"] = dated["date"].dt.to_period("M").dt.to_timestamp()
        monthly = dated.groupby("month").agg(
            risk_ratio=("risk_label", "mean"), reviews=("review", "count")).reset_index()
        fig, ax = plt.subplots(figsize=(9, 4))
        ax.plot(monthly["month"], monthly["risk_ratio"] * 100, color="#e76f51")
        ax.set_title("Monthly risk-label ratio (%)")
        fig.tight_layout(); fig.savefig(OUT / "06_risk_trend.png", dpi=110); plt.close(fig)

    print("saved:", sorted(p.name for p in OUT.glob("*.png")))
    section("DONE")


if __name__ == "__main__":
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else None
    main(rows)
