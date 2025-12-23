from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from rec_sys.dataset_modules.cols_data.create_parquet import (
    preprocess_reviews_to_vectorized_df,
    save_user_parquet,
)
from rec_sys.dataset_modules.dataset import HeteroDataLoader
from rec_sys.dataset_modules.graph_data.create_graphs import build_user_graphs


def prepare_data(
    raw_mus_file: str | Path,
    metadata_file: str | Path,
    unique_user_dir: str | Path,
    graphs_dir: str | Path,
    max_name_len: int,
    max_desc_len: int,
    model_name: str,
    batch_size: int,
    words_fields: list[str],
    user_field: str,
    eval_ratio: float,
):
    """Создает parquet и графы, если их нет."""

    # parquet
    unique_user_dir = Path(unique_user_dir)
    unique_user_dir.mkdir(parents=True, exist_ok=True)
    if not any(unique_user_dir.iterdir()):
        vectorized_df = preprocess_reviews_to_vectorized_df(
            mus_file=Path(raw_mus_file),
            metadata_file=Path(metadata_file),
            max_name_len=max_name_len,
            max_desc_len=max_desc_len,
            model_name=model_name,
            batch_size=batch_size,
            words_fields=words_fields,
        )
        save_user_parquet(
            vectorized_df,
            user_field=user_field,
            unique_user_dir=unique_user_dir,
        )

    # графы
    graphs_dir = Path(graphs_dir)
    graphs_dir.mkdir(parents=True, exist_ok=True)
    if not any(graphs_dir.iterdir()):
        build_user_graphs(
            user_dir=unique_user_dir, save_dir=graphs_dir, eval_ratio=eval_ratio
        )


def create_loaders(cfg: DictConfig):
    """Создает train/test DataLoader’ы из готовых графов."""
    out_graph_file = Path(cfg.dirs.graphs)
    graphs_list = [torch.load(f, weights_only=False) for f in out_graph_file.iterdir()]

    full_size = len(graphs_list)
    test_size = int(full_size * cfg.dataset.test_size)
    train_size = full_size - test_size
    train_dataset, test_dataset = random_split(graphs_list, [train_size, test_size])

    train_loader = HeteroDataLoader(
        train_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=True,
    )
    test_loader = HeteroDataLoader(
        test_dataset,
        batch_size=cfg.dataset.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False,
    )

    return train_loader, test_loader
