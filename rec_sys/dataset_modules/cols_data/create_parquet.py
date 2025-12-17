from sentence_transformers import SentenceTransformer

from rec_sys.dataset_modules.cols_data.parquet_data_utils import (
    build_review_dataset,
    load_metadata,
)
from rec_sys.dataset_modules.cols_data.vector_data_utils import vectorize_df

from pathlib import Path


def preprocess_reviews_to_vectorized_df(
    mus_file: str,
    metadata_file: str,
    max_name_len: int,
    max_desc_len: int,
    model_name: str,
    batch_size: int,
    words_fields: str = None,
):
    """
    Загружает сырой датасет, строит векторизованные данные.
    Возвращает DataFrame с векторными представлениями.

    Args:
        mus_file: путь к исходным отзывам
        metadata_file: путь к метаданным
        max_name_len: макс. длина названия продукта
        max_desc_len: макс. длина описания продукта
        model_name: модель для векторизации текстов
        words_fields: список полей с текстами
    """
    metadata = load_metadata(
        mus_file=mus_file,
        meta_file=metadata_file,
        max_title_length=max_name_len,
        max_desc_length=max_desc_len,
    )

    review_df = build_review_dataset(reviews_path=mus_file, metadata=metadata)

    sent_model = SentenceTransformer(model_name)
    sent_model_call = sent_model.encode

    vectorized_df = vectorize_df(
        review_df,
        sent_model_call,
        words_fields if words_fields is not None else words_fields,
        batch_size
    )

    return vectorized_df


def save_user_parquet(vectorized_df, user_field: str, unique_user_dir: Path):
    """
    Разбивает vectorized_df на пользователей и сохраняет по user_id в parquet.

    Args:
        vectorized_df: DataFrame с векторными представлениями
        user_field: колонка с идентификатором пользователя
        unique_user_dir: директория для сохранения parquet
    """
    unique_user_dir.mkdir(parents=True, exist_ok=True)

    user_groups = vectorized_df.partition_by(user_field, as_dict=True)

    for user_id, df_user in user_groups.items():
        user_path = unique_user_dir / f"{user_id[0]}.parquet"
        df_user.write_parquet(user_path)

