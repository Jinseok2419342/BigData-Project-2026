from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

from .features import MODEL_FEATURES, ensure_features


@dataclass
class ModelBundle:
    pipeline: Pipeline
    metrics: dict
    train_rows: int
    test_rows: int
    positive_rate: float
    feature_importance: pd.DataFrame
    model_name: str = "RandomForest"
    comparison: pd.DataFrame | None = None
    baseline: dict | None = None


def _build_preprocessor() -> ColumnTransformer:
    # [한국어] 모델 입력 전처리기.
    #  - 리뷰 원문 → TF-IDF (1~2gram, 상위 1,500개, 영어 불용어 제거, 2회 미만 등장 제외)
    #  - 수치 특성 → 표준화(StandardScaler). MODEL_FEATURES만 사용 — 라벨 정의
    #    컬럼(severe/symptom 키워드 수, low_rating_flag)은 여기 들어오지 않는다(누수 차단).
    return ColumnTransformer(
        transformers=[
            (
                "text",
                TfidfVectorizer(max_features=1_500, ngram_range=(1, 2), min_df=2, stop_words="english"),
                "review",
            ),
            ("num", StandardScaler(), MODEL_FEATURES),
        ],
        remainder="drop",
    )


def _score(y_true, pred, proba=None) -> dict:
    metrics = {
        "accuracy": accuracy_score(y_true, pred),
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
    }
    if proba is not None and pd.Series(y_true).nunique() == 2:
        metrics["roc_auc"] = roc_auc_score(y_true, proba)
    else:
        metrics["roc_auc"] = np.nan
    return metrics


def train_risk_model(df: pd.DataFrame, sample_size: int = 20_000) -> ModelBundle:
    """[한국어] 위험군 분류 모델 학습 파이프라인.

    1) 라벨 비율을 유지한 층화 샘플링(기본 2만 행) — 수업 PC에서도 1분 내 학습
    2) train/test 75:25 분리(stratify) 후 RandomForest 학습 (배포 모델)
    3) HistGradientBoosting(부스팅 계열)과 단순 규칙(rating≤3) 베이스라인을
       같은 검증셋에서 비교 — "ML이 단순 규칙을 정말 이기는가"를 보여주는 표
    RF를 배포 모델로 둔 이유: 단건 추론이 빠르고 feature_importances_로
    어떤 단어·특성이 위험 판정에 기여했는지 해석할 수 있어서다.
    """
    data = ensure_features(df)
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

    # Leakage-free input: raw review text (TF-IDF) + non-label-defining numerics.
    # [한국어] 누수 차단의 실제 지점: 입력은 "리뷰 원문 + 비누수 수치 특성"뿐이다.
    X = data[["review"] + MODEL_FEATURES]
    stratify = y if y.value_counts().min() >= 2 else None
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        random_state=42,
        stratify=stratify,
    )

    # --- primary model: RandomForest ---
    rf = Pipeline(
        [
            ("features", _build_preprocessor()),
            (
                "model",
                RandomForestClassifier(
                    n_estimators=200,
                    min_samples_leaf=2,
                    max_features="sqrt",
                    class_weight="balanced_subsample",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    rf_proba = rf.predict_proba(X_test)[:, 1]
    rf_metrics = _score(y_test, rf_pred, rf_proba)
    rf_metrics["confusion_matrix"] = confusion_matrix(y_test, rf_pred)

    comparison_rows = [{"model": "RandomForest", **{k: rf_metrics[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}}]

    # --- comparison model: HistGradientBoosting (XGBoost-style, no extra dep) ---
    try:
        densify = FunctionTransformer(_to_dense, accept_sparse=True)
        hgb = Pipeline(
            [
                ("features", _build_preprocessor()),
                ("dense", densify),
                ("model", HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1, random_state=42)),
            ]
        )
        hgb.fit(X_train, y_train)
        hgb_pred = hgb.predict(X_test)
        hgb_proba = hgb.predict_proba(X_test)[:, 1]
        hgb_metrics = _score(y_test, hgb_pred, hgb_proba)
        comparison_rows.append({"model": "HistGradientBoosting", **hgb_metrics})
    except Exception:
        pass

    # --- naive baseline: "rating <= 3 => risk" (the trivial rule a model must beat) ---
    # [한국어] "ML 없이 평점만 보면?"이라는 가장 단순한 규칙. 모델이 이걸 크게
    # 이겨야(F1 0.62 vs 0.90) 텍스트에 실제 예측 신호가 있다고 말할 수 있다.
    base_pred = (X_test["rating"] <= 3).astype(int)
    baseline = _score(y_test, base_pred)
    baseline["name"] = "Baseline (rating<=3)"
    comparison_rows.append({"model": "Baseline (rating<=3)", **{k: baseline[k] for k in ("accuracy", "precision", "recall", "f1", "roc_auc")}})

    comparison = pd.DataFrame(comparison_rows)

    return ModelBundle(
        pipeline=rf,
        metrics=rf_metrics,
        train_rows=len(X_train),
        test_rows=len(X_test),
        positive_rate=float(y.mean()),
        feature_importance=_feature_importance(rf),
        model_name="RandomForest",
        comparison=comparison,
        baseline=baseline,
    )


def _to_dense(x):
    return x.toarray() if hasattr(x, "toarray") else x


def predict_risk(bundle: ModelBundle, row: pd.DataFrame) -> dict:
    # [한국어] 단건 추론: 학습 때와 동일한 입력 형태(리뷰 원문 + 비누수 특성)로
    # 위험 확률을 계산하고, 0.5 기준으로 위험군/안전군을 판정한다.
    X = row[["review"] + MODEL_FEATURES]
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
