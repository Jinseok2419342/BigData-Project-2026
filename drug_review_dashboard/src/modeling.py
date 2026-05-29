from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import NUMERIC_FEATURES, add_features


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metrics: dict
    train_rows: int
    test_rows: int
    positive_rate: float
    feature_importance: pd.DataFrame


def train_risk_model(df: pd.DataFrame, sample_size: int = 20_000) -> ModelBundle:
    data = add_features(df)
    data = data.dropna(subset=["review", "risk_label"]).copy()
    data = data[data["review"].str.strip().astype(bool)]

    if sample_size and len(data) > sample_size:
        sampled_groups = []
        for _, group in data.groupby("risk_label"):
            group_size = min(len(group), max(1, int(sample_size * len(group) / len(data))))
            sampled_groups.append(group.sample(group_size, random_state=42))
        data = pd.concat(sampled_groups, ignore_index=True).sample(frac=1, random_state=42).reset_index(drop=True)

    y = data["risk_label"].astype(int)
    if y.nunique() < 2:
        raise ValueError("Model training requires both risk and safe examples.")

    X = data[["review"] + NUMERIC_FEATURES]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(max_features=1_500, ngram_range=(1, 2), min_df=1, stop_words="english"),
                "review",
            ),
            ("num", StandardScaler(), NUMERIC_FEATURES),
        ],
        remainder="drop",
    )
    model = RandomForestClassifier(
        n_estimators=160,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )
    pipeline = Pipeline([("features", preprocessor), ("model", model)])
    pipeline.fit(X_train, y_train)

    pred = pipeline.predict(X_test)
    proba = pipeline.predict_proba(X_test)[:, 1]
    cm = confusion_matrix(y_test, pred)
    metrics = {
        "accuracy": accuracy_score(y_test, pred),
        "precision": precision_score(y_test, pred, zero_division=0),
        "recall": recall_score(y_test, pred, zero_division=0),
        "f1": f1_score(y_test, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan,
        "confusion_matrix": cm,
    }

    return ModelBundle(
        pipeline=pipeline,
        metrics=metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
        positive_rate=float(y.mean()),
        feature_importance=_feature_importance(pipeline),
    )


def predict_risk(bundle: ModelBundle, row: pd.DataFrame) -> dict:
    X = row[["review"] + NUMERIC_FEATURES]
    proba = float(bundle.pipeline.predict_proba(X)[0, 1])
    pred = int(proba >= 0.5)
    return {
        "risk_probability": proba,
        "prediction": pred,
        "label": "심각한 부작용 위험군" if pred else "일반 리뷰/안전군",
    }


def _feature_importance(pipeline: Pipeline, top_n: int = 20) -> pd.DataFrame:
    model = pipeline.named_steps["model"]
    preprocessor = pipeline.named_steps["features"]
    names = preprocessor.get_feature_names_out()
    importance = getattr(model, "feature_importances_", None)
    if importance is None:
        return pd.DataFrame(columns=["feature", "importance"])

    result = pd.DataFrame({"feature": names, "importance": importance})
    result["feature"] = result["feature"].str.replace("text__", "", regex=False).str.replace("num__", "", regex=False)
    return result.sort_values("importance", ascending=False).head(top_n).reset_index(drop=True)
