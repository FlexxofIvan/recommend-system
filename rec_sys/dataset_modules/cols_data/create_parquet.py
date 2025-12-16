from sentence_transformers import SentenceTransformer

from rec_sys.dataset_modules.cols_data.parquet_data_utils import (
    build_review_dataset,
    load_metadata,
)
from rec_sys.dataset_modules.cols_data.vector_data_utils import vectorize_df


def preprocess_to_parquet(
    mus_file: str,
    metadata_file: str,
    max_name_len: int,
    max_desc_len: int,
    model_name: str,
    unique_user_dir,
    words_fields: str,
    user_field: str,
    batch_size: int,
):
    """
    Загружает сырой датасет, строит векторизованные данные и сохраняет
    по пользователям в parquet.

    Args:
        mus_file: путь к исходным отзывам
        metadata_file: путь к метаданным
        max_name_len: максимальная длина названия продукта
        max_desc_len: максимальная длина описания продукта
        unique_user_dir: директория для сохранения parquet по пользователям
        user_field: колонка с идентификатором пользователя
        model_name: функция или модель для векторизации текстов
        words_fields: список полей с текстами для обработки
        batch_size: размер батча для обработки
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
        sent_model_call if sent_model is None else sent_model,
        words_fields if words_fields is None else words_fields,
        batch_size,
    )

    user_groups = vectorized_df.partition_by(user_field, as_dict=True)

    for user_id, df_user in user_groups.items():
        user_path = unique_user_dir / f"{user_id[0]}.parquet"
        df_user.write_parquet(user_path)
