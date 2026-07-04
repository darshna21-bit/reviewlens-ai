"""
Data pipeline for the Amazon Reviews dataset (Kaggle).

Expected input: data/raw/*.csv with at least these columns:
  - review_body   (the actual review text)
  - star_rating   (integer 1–5)
  - review_headline (optional, used as summary target for T5 training)

Run: python training/data_pipeline.py
"""

import argparse
import logging
import re
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# add project root to path so imports work regardless of where you run from
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.utils.text_utils import clean_review_text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")

# label mapping — using 3 classes instead of binary pos/neg because early
# experiments with just pos/neg had the model confidently mislabeling mediocre
# 3-star reviews as positive (they are positive relative to 1-star but the
# language is clearly hedged)
LABEL_MAP = {1: 0, 2: 0, 3: 1, 4: 2, 5: 2}
LABEL_NAMES = {0: "negative", 1: "neutral", 2: "positive"}


def find_raw_csv() -> Path:
    """
    Looks for any CSV in data/raw/. The Kaggle dataset comes with a bunch
    of category-specific files; we just take the first one found. If you've
    merged them into a single file, that works too.
    """
    csvs = list(RAW_DIR.glob("*.csv")) + list(RAW_DIR.glob("*.tsv"))
    if not csvs:
        raise FileNotFoundError(
            f"No CSV/TSV files found in {RAW_DIR}. "
            "Download the Amazon Reviews dataset from Kaggle and place it there."
        )
    logger.info(f"Found raw data file: {csvs[0]}")
    return csvs[0]


def load_raw_data(filepath: Path, max_rows: int | None = None) -> pd.DataFrame:
    """
    Loads the CSV. The Kaggle Amazon dataset is tab-separated in some versions,
    so we try both delimiters. The on_bad_lines='skip' saved me from a crash
    when a few rows had unescaped quotes in the review text.
    """
    try:
        df = pd.read_csv(
            filepath,
            sep="\t",
            on_bad_lines="skip",
            nrows=max_rows,
        )
        if df.shape[1] < 3:
            raise ValueError("Too few columns — trying comma separator")
    except Exception:
        df = pd.read_csv(
            filepath,
            sep=",",
            on_bad_lines="skip",
            nrows=max_rows,
        )

    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    logger.info(f"Columns: {list(df.columns)}")
    return df


def normalize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Kaggle Amazon datasets have different column name conventions depending on
    which version you download. Normalizing here so downstream code doesn't
    need to know which variant it got.
    """
    rename_map = {}
    col_lower = {c.lower(): c for c in df.columns}

    # review body
    for candidate in ["review_body", "reviewtext", "review text", "body", "text"]:
        if candidate in col_lower:
            rename_map[col_lower[candidate]] = "review_body"
            break

    # star rating
    for candidate in ["star_rating", "overall", "rating", "stars"]:
        if candidate in col_lower:
            rename_map[col_lower[candidate]] = "star_rating"
            break

    # headline / title (used as summary ground truth for T5)
    for candidate in ["review_headline", "summary", "review_title", "title"]:
        if candidate in col_lower:
            rename_map[col_lower[candidate]] = "review_headline"
            break

    df = df.rename(columns=rename_map)

    if "review_body" not in df.columns:
        raise ValueError(
            f"Couldn't find a review text column. Available columns: {list(df.columns)}"
        )
    if "star_rating" not in df.columns:
        raise ValueError(
            f"Couldn't find a star rating column. Available columns: {list(df.columns)}"
        )

    return df


def clean_and_filter(df: pd.DataFrame) -> pd.DataFrame:
    initial_count = len(df)

    # drop rows where review_body or star_rating is missing
    df = df.dropna(subset=["review_body", "star_rating"])
    after_null = len(df)

    # clean the review text
    df["review_body"] = df["review_body"].apply(clean_review_text)

    # remove empty reviews after cleaning
    df = df[df["review_body"].str.len() > 20]
    after_empty = len(df)

    # drop exact duplicates — the dataset has some copy-pasted reviews
    df = df.drop_duplicates(subset=["review_body"])
    after_dedup = len(df)

    # coerce star_rating to int, drop anything that doesn't parse
    df["star_rating"] = pd.to_numeric(df["star_rating"], errors="coerce")
    df = df.dropna(subset=["star_rating"])
    df["star_rating"] = df["star_rating"].astype(int)
    df = df[df["star_rating"].between(1, 5)]
    after_rating = len(df)

    logger.info(f"Cleaning stats:")
    logger.info(f"  Initial rows:          {initial_count:>8,}")
    logger.info(f"  After null removal:    {after_null:>8,} (removed {initial_count - after_null:,})")
    logger.info(f"  After empty removal:   {after_empty:>8,} (removed {after_null - after_empty:,})")
    logger.info(f"  After deduplication:   {after_dedup:>8,} (removed {after_empty - after_dedup:,})")
    logger.info(f"  After rating filter:   {after_rating:>8,} (removed {after_dedup - after_rating:,})")

    return df


def add_sentiment_labels(df: pd.DataFrame) -> pd.DataFrame:
    df["label"] = df["star_rating"].map(LABEL_MAP)
    df = df.dropna(subset=["label"])
    df["label"] = df["label"].astype(int)

    logger.info("\nClass distribution:")
    dist = df["label"].value_counts().sort_index()
    for label_id, count in dist.items():
        pct = count / len(df) * 100
        logger.info(f"  {LABEL_NAMES[label_id]:8s} (label {label_id}): {count:>7,} ({pct:.1f}%)")

    return df


def log_text_stats(df: pd.DataFrame):
    lengths = df["review_body"].str.split().str.len()
    logger.info(f"\nReview length stats (words):")
    logger.info(f"  Mean:   {lengths.mean():.1f}")
    logger.info(f"  Median: {lengths.median():.1f}")
    logger.info(f"  P95:    {lengths.quantile(0.95):.1f}")
    logger.info(f"  Max:    {lengths.max()}")


def split_and_save(df: pd.DataFrame):
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    # stratified split to maintain class balance across train/val/test
    train_df, temp_df = train_test_split(
        df, test_size=0.30, stratify=df["label"], random_state=42
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.50, stratify=temp_df["label"], random_state=42
    )

    # keep only the columns we need downstream
    columns_to_keep = ["review_body", "star_rating", "label"]
    if "review_headline" in df.columns:
        columns_to_keep.append("review_headline")

    train_df[columns_to_keep].to_csv(PROCESSED_DIR / "train.csv", index=False)
    val_df[columns_to_keep].to_csv(PROCESSED_DIR / "val.csv", index=False)
    test_df[columns_to_keep].to_csv(PROCESSED_DIR / "test.csv", index=False)

    logger.info(f"\nSplit sizes:")
    logger.info(f"  Train: {len(train_df):,}")
    logger.info(f"  Val:   {len(val_df):,}")
    logger.info(f"  Test:  {len(test_df):,}")
    logger.info(f"\nProcessed data saved to {PROCESSED_DIR}/")


def balance_classes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Downsamples the dataset so that negative, neutral, and positive classes
    have an equal number of samples, preventing training bias.
    """
    min_class_size = df["label"].value_counts().min()
    logger.info(f"Balancing classes. Downsampling all classes to match smallest class size: {min_class_size:,}")
    
    balanced_df = (
        df.groupby("label", group_keys=False)
        .apply(lambda x: x.sample(min_class_size, random_state=42))
        .reset_index(drop=True)
    )
    
    logger.info("New balanced class distribution:")
    dist = balanced_df["label"].value_counts().sort_index()
    for label_id, count in dist.items():
        pct = count / len(balanced_df) * 100
        logger.info(f"  {LABEL_NAMES[label_id]:8s} (label {label_id}): {count:>7,} ({pct:.1f}%)")
        
    return balanced_df


def run_pipeline(max_rows: int | None = None):
    logger.info("=" * 60)
    logger.info("ReviewLens AI — Data Pipeline")
    logger.info("=" * 60)

    filepath = find_raw_csv()
    df = load_raw_data(filepath, max_rows=max_rows)
    df = normalize_column_names(df)
    df = clean_and_filter(df)
    df = add_sentiment_labels(df)
    df = balance_classes(df)
    log_text_stats(df)
    split_and_save(df)

    logger.info("\nPipeline complete.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="ReviewLens data preprocessing pipeline")
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Limit rows for quick testing (e.g. --max-rows 10000)",
    )
    args = parser.parse_args()
    run_pipeline(max_rows=args.max_rows)
