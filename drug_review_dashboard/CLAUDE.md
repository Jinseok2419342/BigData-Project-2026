# CLAUDE.md — System Guide

Definitive guide for Claude Code sessions on this project. Read this first.

This file lives in the **project root** (`drug_review_dashboard/`), alongside
`app.py` and `README.md`. **Run all commands from this folder.** The shared
virtual environment `venv/` lives one level up at the workspace root (`../venv`),
which also holds unrelated weekly course exercises (`../NN_week/`).

---

## 1. Project Overview & Tech Stack

**Big Data Drug Review Pipeline** — detects "serious adverse drug event (ADE)
risk" reviews from the UCI ML Drug Review dataset (~215k rows) and serves the
result through an interactive dashboard.

- **App / UI**: Streamlit multipage app (`app.py` + `pages/`)
- **ML**: scikit-learn — RandomForest (deployed) + HistGradientBoosting (benchmark), TF-IDF text features
- **LLM (Ollama-first)**: local **Ollama `gemma3`** → **OpenAI `gpt-4o-mini`** backup → offline rule-based
- **R&D / Lab**: **JupyterLab** notebook (`notebooks/EDA_and_Preprocessing_Analysis.ipynb`)
- **Data**: KaggleHub `jessicali9530/kuc-hackathon-winter-2018`, cached under `.kagglehub_cache/`

---

## 2. Build & Run Commands

A virtual environment (`venv/`) already exists at the **workspace root** (`../venv`).
Create/activate it:

```bash
# Windows (PowerShell) — from the workspace root (one level up)
python -m venv ..\venv                  # only if it does not exist yet
..\venv\Scripts\Activate.ps1            # if blocked: Set-ExecutionPolicy -Scope Process RemoteSigned

# macOS / Linux — from the workspace root
python3 -m venv ../venv                 # only if it does not exist yet
source ../venv/bin/activate
```

Then, from **this** folder (`drug_review_dashboard/`):

```bash
pip install -r requirements.txt

streamlit run app.py        # launch the dashboard
jupyter lab                 # open the R&D notebook (notebooks/…ipynb)
pytest -q                   # run unit tests (tests/)
pytest tests/test_llm_routing.py -q   # just the LLM-routing tests
```

> Without activating the venv, prefix Python directly, e.g.
> `..\venv\Scripts\python.exe -m pytest -q` (Windows) — `src/` imports require the venv.

---

## 3. Core Architecture (R&D → Production)

A **single source of truth**: the EDA/preprocessing/feature-engineering logic is
defined and proven in the notebook, and the Streamlit app reuses the **exact same**
`src/` core — no duplicated logic.

```
src/  (Single Source of Truth)
  data_loader.py  load_drug_reviews()   # deterministic, offline-first cache load
  features.py     add_features()        # cleaning + 13 engineered features + weak label
  data_cache.py   prepare_data()        # load_drug_reviews -> add_features (plain function)
                  get_prepared_data()   # @st.cache_data wrapper of prepare_data (app only)
        │                                   │
   notebooks/ (R&D: define/prove)       pages/ (Production: serve, cached)
```

- **`prepare_data()`** is the shared, Streamlit-free core. **`get_prepared_data()`**
  is only its cache wrapper — identical output. The notebook calls the same
  functions, so notebook and app yield identical shapes/dtypes/feature values.
- **`data_loader.py`** is **deterministic & offline-first**: it reads the
  already-downloaded KaggleHub CSVs directly from `.kagglehub_cache/`
  (`_load_kaggle_cache`) *before* any network call, so a flaky connection never
  silently degrades to demo data. Order: `local data/ → kaggle cache → kagglehub download → demo`.
- **Caching is centralized** in `data_cache.py`; all 5 pages call `get_prepared_data`
  with identical args so they share one cache entry (instant navigation).

---

## 4. Professor's Feedback & Compliance Rules

These four constraints are non-negotiable — preserve them in any future edit:

1. **Data-driven Weak Labeling** — no ground-truth ADE label exists, so the target
   is a transparent rule: `(severe keyword) OR (rating ≤ 3 AND symptom keyword)`.
   Keep weak labeling; do not invent a fake "true" label.
2. **Target Leakage Isolation** — the columns that *define* the label
   (`LABEL_DEFINING_FEATURES`: `severe_keyword_count`, `symptom_keyword_count`,
   `low_rating_flag`) are **excluded from training**. Models use only
   `MODEL_FEATURES` + raw-text TF-IDF. This keeps metrics honest (~90–97%, not a
   leaky ~99.6%). **Never feed label-defining features to a model.**
3. **Image OCR is decoupled** — the text pipeline + chatbot are the core. Pill-image
   multimodal recognition is an isolated **bonus** path with graceful fallback
   (filename match) so the dashboard never crashes if no vision model is running.
4. **License clarity** — the data is the UCI **Drug Review Dataset (Drugs.com)**
   (UCI #462, Gräßer et al. 2018; *not* the Druglib.com dataset, UCI #461): **CC BY 4.0**
   at UCI, while the Kaggle redistribution (`kuc-hackathon-winter-2018`) lists the stricter
   **CC BY-NC-SA 4.0** — the project follows the stricter terms. Keep this accurate and
   identical across README/보고서/문제정의/발표자료.

---

## 5. Code Style & Guidelines

- **Keep the centralized caching layer.** New pages must load data via
  `src.data_cache.get_prepared_data(max_rows, prefer_kaggle)` with the **same arg
  shape** as existing pages (positional `(max_rows, False)`) — do not add per-page
  `@st.cache_data def get_data(...)`. Key model caches on scalars, not DataFrames.
- **Zero target leakage in model edits.** When adding features, decide whether each
  belongs in `MODEL_FEATURES` (safe) or `LABEL_DEFINING_FEATURES` (excluded). If a
  feature is a deterministic component of `build_target`, it must be excluded.
- **Vectorized keyword extraction.** `count_keywords` uses one precompiled,
  `lru_cache`d regex alternation per keyword set — do not regress to per-keyword
  `re.search` loops (it makes `add_features` ~4× slower on 50k rows).
- **LLM calls go through `route_chat()`** (Ollama-first → OpenAI → rule-based). Every
  LLM path must fail gracefully to the rule-based answer; never let the app crash on
  a missing model/key.
- **Keep notebook ↔ app parity.** If you change `add_features`/`prepare_data`, the
  notebook reflects it automatically (shared import) — re-run the notebook and
  confirm identical outputs.
- **Verify before done**: `pytest -q` (18 tests) green, and `streamlit run app.py`
  boots with zero errors.
