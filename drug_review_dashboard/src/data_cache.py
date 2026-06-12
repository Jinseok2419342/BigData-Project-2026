"""Shared Streamlit cache layer.

The heavy work in this app is (1) loading the Kaggle CSV and (2) running
``add_features`` (regex keyword counting + per-drug IQR stats over tens of
thousands of rows). Previously every page defined its *own* ``get_data`` so
Streamlit cached each one separately — the dataset was loaded and feature-
engineered up to 5 times and re-run on every page switch.

This module centralizes that work behind a single set of cached functions.
Because every page imports the *same* function object, ``@st.cache_data`` keys
on the function + args and returns the one shared result, so navigation only
pays a cheap copy of an already-prepared DataFrame.

[한국어] 중앙 캐시 계층 — 이 앱에서 가장 비싼 작업 두 가지(① CSV 5만 행 로딩,
② add_features의 정규식 키워드 카운트 + 약물별 IQR 통계)를 단 한 번만 실행한다.
예전에는 페이지마다 자기만의 get_data를 정의해 같은 작업이 최대 5번 반복됐지만,
지금은 5개 페이지가 모두 "같은 함수 객체" get_prepared_data를 같은 인자
(max_rows, False)로 호출하므로 Streamlit 캐시 항목 1개를 공유한다
→ 페이지 전환이 즉시 이루어진다.
"""
from __future__ import annotations

import streamlit as st

from .data_loader import load_drug_reviews
from .features import add_features, get_drug_summary


def prepare_data(max_rows: int = 50_000, prefer_kaggle: bool = False):
    """Load + clean + feature-engineer the dataset. Returns (featured_df, source, warning).

    This is the **single shared core pipeline** proven in the Jupyter notebook
    (notebooks/EDA_and_Preprocessing_Analysis.ipynb) and reused, unchanged, by the
    Streamlit app. It is a plain function (no Streamlit dependency) so the notebook
    can call the exact same preprocessing/feature-engineering the app serves.

    [한국어] 노트북(R&D)과 앱(Production)이 공유하는 핵심 파이프라인:
    load_drug_reviews(적재) → add_features(특성 생성). Streamlit 의존성이 없는
    순수 함수라 노트북에서도 그대로 호출 가능 — 단일 진실 원천(SSOT)의 실체.
    """
    df, source = load_drug_reviews(max_rows=max_rows, prefer_kaggle=prefer_kaggle)
    featured = add_features(df)
    warning = df.attrs.get("load_warning", "")
    return featured, source, warning


@st.cache_data(show_spinner="데이터 로딩·전처리 중... (최초 1회만 실행)")
def get_prepared_data(max_rows: int = 50_000, prefer_kaggle: bool = False):
    """Streamlit-cached wrapper around prepare_data() — identical output, cached once."""
    return prepare_data(max_rows, prefer_kaggle)


@st.cache_data(show_spinner=False)
def get_summary(max_rows: int = 50_000, prefer_kaggle: bool = False, min_reviews: int = 5):
    """Per-drug summary (risk ratio, avg rating, ...), cached by its parameters."""
    featured, _, _ = get_prepared_data(max_rows, prefer_kaggle)
    return get_drug_summary(featured, min_reviews=min_reviews)
