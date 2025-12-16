from typing import Callable, List

import numpy as np
import polars as pl
from tqdm import tqdm


def batch_iter(data: List[str], batch_size: int) -> List[List[str]]:
    """
    Yield successive batches from a list.

    Parameters
    ----------
    data : list[str]
        List of items to split into batches.
    batch_size : int
        Batch size.

    Yields
    ------
    list[str]
        Next batch of data.
    """
    n = len(data)
    for i in range(0, n, batch_size):
        yield data[i : i + batch_size]


def text_to_vec(
    model_call: Callable[[List[str]], np.ndarray], words: List[str], batch_size: int
) -> np.ndarray:
    """
    Convert a list of strings into embeddings using the provided model callable.

    Parameters
    ----------
    model_call : Callable
        Model callable that converts list of strings to embeddings.
    words : list[str]
        List of text strings to vectorize.
    batch_size : int
        Batch size for model.

    Returns
    -------
    np.ndarray
        Array of embeddings.
    """
    return model_call(words, batch_size=batch_size)


def vectorize_df(
    df: pl.DataFrame,
    vec_model_call: Callable[[List[str]], np.ndarray],
    words_fields: List[str],
    batch_size: int,
) -> pl.DataFrame:
    """
    Vectorize specified text columns in a Polars DataFrame.

    Non-text columns are preserved.

    Parameters
    ----------
    df : pl.DataFrame
        Input DataFrame containing text columns.
    vec_model_call : Callable
        Model callable (e.g., SentenceTransformer.encode) to vectorize text.
    words_fields : list[str]
        List of columns to vectorize.
    batch_size : int
        Batch size for vectorization.

    Returns
    -------
    pl.DataFrame
        DataFrame with vectorized columns replacing original text columns.
    """
    data_dict = dict()

    # Preserve non-text columns
    text_fields = [col for col in df.columns if col not in words_fields]
    for field in text_fields:
        data_dict[field] = df[field].to_list()

    # Vectorize text fields
    for field in words_fields:
        if field not in df.columns:
            raise ValueError(f"Missing field in DataFrame: {field}")

        df_field_list = df[field].to_list()
        vectorized_field = []

        total_batches = (len(df_field_list) + batch_size - 1) // batch_size
        for batch in tqdm(
            batch_iter(df_field_list, batch_size),
            desc=f"Vectorizing {field}",
            total=total_batches,
        ):
            embeddings = text_to_vec(vec_model_call, batch, batch_size)
            vectorized_field.extend(embeddings)

        data_dict[field] = vectorized_field

    return pl.DataFrame(data_dict)
