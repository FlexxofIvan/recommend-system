import json
import random
from pathlib import Path
from typing import Union

import polars as pl


def build_review_dataset(
    reviews_path: Union[str, Path], metadata: pl.DataFrame
) -> Path:
    """
    Build a clean review dataset by merging review JSONL data
    with product metadata (Parquet). Produces a Parquet file.

    Parameters
    ----------
    reviews_path : str | Path
        Path to the reviews JSON Lines file (.jsonl / .ndjson).
    metadata : pl.DataFrame
        product metadata Parquet file.
    output_path : str | Path
        Path where the resulting Parquet will be written.

    Returns
    -------
    Path
        Path to the written Parquet file.

    Notes
    -----
    This function:
    - Loads reviews in vectorized Polars mode
    - Fills missing fields with defaults
    - Joins onto metadata via ASIN
    - Produces a deduplicated, enriched dataset
    - Writes it to `output_path`
    """

    reviews_path = Path(reviews_path)

    if not reviews_path.exists():
        raise FileNotFoundError(f"Reviews file not found: {reviews_path}")

    if not isinstance(metadata, pl.DataFrame):
        raise FileNotFoundError(f"Metadata not dataframe: {metadata}")

    reviews = pl.read_ndjson(reviews_path)

    df = (
        reviews.select(
            [
                pl.col("reviewerID").alias("reviewerID"),
                "asin",
                pl.col("overall").alias("score"),
                pl.col("reviewText").fill_null("none").alias("comment"),
                pl.col("verified").fill_null(False).alias("verified"),
            ]
        )
        .join(metadata, on="asin", how="inner")
        .select(
            [
                "reviewerID",
                "asin",
                pl.col("name").alias("product_name"),
                pl.col("description").alias("product_info"),
                "comment",
                "verified",
                "score",
            ]
        )
        .unique()
    )
    return df


def validate_path(path: Path) -> None:
    """Ensure file exists and is a file."""
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"Path does not exist or is not a file: {path}")


def load_metadata(
    mus_file: Path, max_title_length: int, max_desc_length: int, meta_file: Path
) -> pl.DataFrame:
    """
    Load metadata for instruments found in mus_file.

    Parameters
    ----------
    mus_file : Path
        Path to NDJSON file with music items (contains `asin` field).
    meta_file : Path
        Path to NDJSON metadata file.
    max_title_length : int
        Define maximum length of title in output.
    max_desc_length : int
        Define maximum length of description in output.
    Returns
    -------
    pl.DataFrame
        Joined metadata with truncated title/description.
    """
    validate_path(mus_file)
    validate_path(meta_file)

    mus = pl.scan_ndjson(mus_file).select("asin").unique()

    meta = pl.scan_ndjson(meta_file)

    result = (
        meta.join(mus, on="asin", how="inner")
        .select(
            [
                "asin",
                pl.col("title")
                .str.split(" ")
                .list.slice(0, max_title_length)
                .list.join(" ")
                .alias("name"),
                pl.col("description")
                .list.join(" ")
                .str.split(" ")
                .list.slice(0, max_desc_length)
                .list.join(" ")
                .fill_null("none")
                .alias("description"),
            ]
        )
        .collect()
    )

    return result

def split_jsonl_by_user(
    input_path,
    train_path,
    test_path,
    user_field,
    test_ratio=0.1,
    seed=42,
):
    random.seed(seed)
    user_split = {}

    with open(input_path, "r", encoding="utf-8") as fin, \
         open(train_path, "w", encoding="utf-8") as ftrain, \
         open(test_path, "w", encoding="utf-8") as ftest:

        for line in fin:
            obj = json.loads(line)
            user = obj[user_field]

            if user not in user_split:
                user_split[user] = random.random() < test_ratio

            if user_split[user]:
                ftest.write(line)
            else:
                ftrain.write(line)