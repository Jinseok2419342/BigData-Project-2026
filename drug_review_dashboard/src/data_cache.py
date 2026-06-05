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
