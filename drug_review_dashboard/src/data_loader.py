from __future__ import annotations

from pathlib import Path
from typing import Iterable
from zipfile import ZipFile, is_zipfile
import os

import numpy as np
import pandas as pd


PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
DATASET_SLUG = "jessicali9530/kuc-hackathon-winter-2018"
KAGGLE_FILES = ("drugsComTrain_raw.csv", "drugsComTest_raw.csv")
LOCAL_CANDIDATES = (
    "drug_reviews.csv",
    "uci_drug_reviews.csv",
    "kuc_drug_reviews.csv",
    "drugsComTrain_raw.csv",
    "drugsComTest_raw.csv",
)
LOCAL_ARCHIVES = (
    "kuc-hackathon-winter-2018.zip",
    "drug_reviews.zip",
    "uci_drug_reviews.zip",
)


STANDARD_COLUMNS = {
    "uniqueid": "review_id",
    "uniqueID": "review_id",
    "drugName": "drug_name",
    "drug_name": "drug_name",
    "condition": "condition",
    "review": "review",
    "rating": "rating",
    "date": "date",
    "usefulCount": "useful_count",
    "useful_count": "useful_count",
}


def _read_csv(path: Path, max_rows: int | None = None) -> pd.DataFrame:
    if is_zipfile(path):
        with ZipFile(path) as archive:
            csv_members = [name for name in archive.namelist() if Path(name).suffix.lower() == ".csv"]
            exact_members = [name for name in csv_members if Path(name).name == path.name]
            member_name = exact_members[0] if exact_members else csv_members[0]
        return _read_csv_from_zip(path, member_name, max_rows)

    try:
        return pd.read_csv(path, nrows=max_rows)
    except UnicodeDecodeError:
        return pd.read_csv(path, encoding="latin-1", nrows=max_rows)


def _read_csv_from_zip(zip_path: Path, member_name: str, max_rows: int | None = None) -> pd.DataFrame:
    with ZipFile(zip_path) as archive:
        with archive.open(member_name) as file:
            try:
                return pd.read_csv(file, nrows=max_rows)
            except UnicodeDecodeError:
                file.seek(0)
                return pd.read_csv(file, encoding="latin-1", nrows=max_rows)


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    lower_map = {str(col).lower(): col for col in df.columns}
    for source, target in STANDARD_COLUMNS.items():
        if source in df.columns:
            rename[source] = target
        elif source.lower() in lower_map:
            rename[lower_map[source.lower()]] = target

    df = df.rename(columns=rename).copy()

    required_defaults = {
        "review_id": np.arange(len(df)),
        "drug_name": "Unknown",
        "condition": "Unknown",
        "review": "",
        "rating": np.nan,
        "date": pd.NaT,
        "useful_count": 0,
    }
    for col, default in required_defaults.items():
        if col not in df.columns:
            df[col] = default

    df["drug_name"] = df["drug_name"].fillna("Unknown").astype(str).str.strip()
    df["condition"] = df["condition"].fillna("Unknown").astype(str).str.strip()
    df["review"] = df["review"].fillna("").astype(str)
    df["review"] = (
        df["review"]
        .str.replace("&quot;", '"', regex=False)
        .str.replace("&#039;", "'", regex=False)
        .str.strip()
    )
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df["useful_count"] = pd.to_numeric(df["useful_count"], errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"], errors="coerce", format="mixed")

    df = df[df["review"].str.len() > 0].reset_index(drop=True)
    return df


def _load_local_csvs(max_rows: int | None = None) -> tuple[pd.DataFrame, str]:
    frames: list[pd.DataFrame] = []
    used_files: list[str] = []

    for name in LOCAL_CANDIDATES:
        path = DATA_DIR / name
        if path.exists():
            remaining = None if max_rows is None else max(max_rows - sum(len(f) for f in frames), 0)
            if remaining == 0:
                break
            frames.append(_read_csv(path, remaining))
            used_files.append(name)

    for archive_name in LOCAL_ARCHIVES:
        archive_path = DATA_DIR / archive_name
        if not archive_path.exists():
            continue

        with ZipFile(archive_path) as archive:
            csv_members = [
                name
                for name in archive.namelist()
                if Path(name).name in LOCAL_CANDIDATES or Path(name).suffix.lower() == ".csv"
            ]

        for member_name in csv_members:
            remaining = None if max_rows is None else max(max_rows - sum(len(f) for f in frames), 0)
            if remaining == 0:
                break
            frames.append(_read_csv_from_zip(archive_path, member_name, remaining))
            used_files.append(f"{archive_name}:{member_name}")

    if not frames:
        raise FileNotFoundError("No local drug review CSV files were found in data/.")

    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        df = df.head(max_rows)
    return _standardize_columns(df), "local: " + ", ".join(used_files)


def _kagglehub_versions_dir() -> Path:
    cache_root = Path(os.environ.get("KAGGLEHUB_CACHE", str(PROJECT_DIR / ".kagglehub_cache")))
    return cache_root / "datasets" / DATASET_SLUG / "versions"


def _load_kaggle_cache(max_rows: int | None = None) -> tuple[pd.DataFrame, str]:
    """Read the already-downloaded KaggleHub CSVs directly from the local cache.

    KaggleHub's dataset_download() makes a network call on every load even when
    the file is cached, and a transient connection error would otherwise drop the
    app to demo data. Reading the cached files directly makes loading fast,
    offline-robust, and deterministic, so the notebook and app never diverge.
    """
    versions = _kagglehub_versions_dir()
    if not versions.exists():
        raise FileNotFoundError("No KaggleHub cache directory found.")

    frames: list[pd.DataFrame] = []
    used_files: list[str] = []
    for version_dir in sorted((p for p in versions.iterdir() if p.is_dir()), reverse=True):
        present = [version_dir / name for name in KAGGLE_FILES if (version_dir / name).exists()]
        if not present:
            continue
        for path in present:
            remaining = None if max_rows is None else max(max_rows - sum(len(f) for f in frames), 0)
            if remaining == 0:
                break
            frames.append(_read_csv(path, remaining))
            used_files.append(path.name)
        break  # use the newest version that has the files

    if not frames:
        raise FileNotFoundError("No cached KaggleHub CSV files found.")

    df = pd.concat(frames, ignore_index=True)
    if max_rows is not None:
        df = df.head(max_rows)
    return _standardize_columns(df), "kaggle cache: " + ", ".join(used_files)


def _load_kagglehub(max_rows: int | None = None) -> tuple[pd.DataFrame, str]:
    os.environ.setdefault("KAGGLEHUB_CACHE", str(PROJECT_DIR / ".kagglehub_cache"))
    _patch_kagglesdk_for_kagglehub()

    import kagglehub

    frames: list[pd.DataFrame] = []
    for file_path in KAGGLE_FILES:
        remaining = None if max_rows is None else max(max_rows - sum(len(f) for f in frames), 0)
        if remaining == 0:
            break

        downloaded = kagglehub.dataset_download(DATASET_SLUG, path=file_path)
        df = _read_csv(Path(downloaded), remaining)
        frames.append(df)

    if not frames:
        raise RuntimeError("KaggleHub returned no frames.")
    return _standardize_columns(pd.concat(frames, ignore_index=True)), "kagglehub"


def _patch_kagglesdk_for_kagglehub() -> None:
    """Patch a known kagglehub/kagglesdk compatibility mismatch.

    Some environments install kagglehub 1.0.x with a kagglesdk build that
    exposes get_endpoint(), while kagglehub imports get_web_endpoint().
    Adding the alias before importing kagglehub prevents an import crash.
    """
    try:
        import kagglesdk.kaggle_env as kaggle_env
    except Exception:
        return

    if hasattr(kaggle_env, "get_web_endpoint"):
        return

    if hasattr(kaggle_env, "get_endpoint"):
        kaggle_env.get_web_endpoint = kaggle_env.get_endpoint


def _generate_demo_data(rows: int = 480) -> tuple[pd.DataFrame, str]:
    rng = np.random.default_rng(42)
    drugs = [
        ("Sertraline", "Depression"),
        ("Lexapro", "Anxiety"),
        ("Metformin", "Diabetes, Type 2"),
        ("Lisinopril", "High Blood Pressure"),
        ("Ibuprofen", "Pain"),
        ("Prednisone", "Inflammation"),
        ("Gabapentin", "Neuropathic Pain"),
        ("Amitriptyline", "Migraine Prevention"),
    ]
    safe_reviews = [
        "Worked well after two weeks. Mild nausea at first but it settled down.",
        "My symptoms improved and I could sleep better. No major side effects.",
        "A little dizziness in the morning, otherwise the medicine helped.",
        "The dose was adjusted and I felt stable. I would discuss follow up with my doctor.",
        "Some dry mouth and fatigue, but the benefit was worth it for me.",
        "No serious reaction. I only noticed a headache on the first day.",
    ]
    risky_reviews = [
        "Severe chest pain and my heart was racing. I felt like I might pass out.",
        "I had trouble breathing, swelling in my face, and went to the emergency room.",
        "After taking it I had suicidal thoughts and panic attacks that scared me.",
        "A rash spread quickly with fever and dizziness. I stopped and called a doctor.",
        "I had a seizure and confusion after the new dose. It felt dangerous.",
        "Extreme vomiting and dehydration. I needed urgent medical help.",
    ]

    records = []
    for i in range(rows):
        drug, condition = drugs[i % len(drugs)]
        risky = rng.random() < 0.28
        review = rng.choice(risky_reviews if risky else safe_reviews)
        rating = rng.integers(1, 4) if risky else rng.integers(5, 11)
        useful = int(rng.poisson(12 if risky else 5))
        records.append(
            {
                "review_id": i + 1,
                "drug_name": drug,
                "condition": condition,
                "review": review,
                "rating": float(rating),
                "date": pd.Timestamp("2023-01-01") + pd.Timedelta(days=int(i * 3)),
                "useful_count": useful,
            }
        )

    return _standardize_columns(pd.DataFrame(records)), "demo sample"


def load_drug_reviews(
    max_rows: int | None = 50_000,
    prefer_kaggle: bool = False,
) -> tuple[pd.DataFrame, str]:
    """Load UCI Drug Review data.

    Priority:
    1. local CSV files in data/
    2. already-downloaded KaggleHub cache (read directly, no network)
    3. KaggleHub network download (first-time fetch)
    4. generated demo data, so the app still opens during development
    """
    loaders: Iterable = (
        (_load_kaggle_cache, _load_kagglehub, _load_local_csvs)
        if prefer_kaggle
        else (_load_local_csvs, _load_kaggle_cache, _load_kagglehub)
    )

    last_error = None
    for loader in loaders:
        try:
            return loader(max_rows)
        except Exception as exc:  # keep the app usable even without network/data
            last_error = exc

    df, source = _generate_demo_data()
    df.attrs["load_warning"] = str(last_error) if last_error else ""
    return df, source
