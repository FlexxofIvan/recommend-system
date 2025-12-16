from pathlib import Path

import torch
from omegaconf import DictConfig
from torch.utils.data import random_split

from rec_sys.dataset_modules.cols_data.create_parquet import preprocess_to_parquet
from rec_sys.dataset_modules.dataset import HeteroDataLoader
from rec_sys.dataset_modules.graph_data.create_graphs import build_user_graphs


def create_loader(cfg: DictConfig):
    out_parquet_file = Path(cfg.dirs.unique_user_data)
    out_parquet_file.mkdir(parents=True, exist_ok=True)
    if not any(out_parquet_file.iterdir()):
        parquet_kwargs = {
            "mus_file": Path(cfg.files.raw_mus_file),
            "metadata_file": Path(cfg.files.raw_metadata_file),
            "max_name_len": cfg.dataset.max_name_len,
            "max_desc_len": cfg.dataset.max_desc_len,
            "model_name": cfg.dataset.model_name,
            "words_fields": cfg.dataset.words_fields,
            "user_field": cfg.dataset.user_field,
            "batch_size": cfg.dataset.batch_size,
            "unique_user_dir": Path(cfg.dirs.unique_user_data),
        }
        preprocess_to_parquet(**parquet_kwargs)

    out_graph_file = Path(cfg.dirs.graphs)
    out_graph_file.mkdir(parents=True, exist_ok=True)
    if not any(out_graph_file.iterdir()):
        graph_kwargs = {
            "user_dir": Path(cfg.dirs.unique_user_data),
            "save_dir": out_graph_file,
            "eval_ratio": cfg.dataset.eval_ratio,
        }
        build_user_graphs(**graph_kwargs)

    graphs_list = []
    for graph_file in out_graph_file.iterdir():
        graph = torch.load(graph_file, weights_only=False)
        graphs_list.append(graph)

    full_size = len(graphs_list)

    test_size = int(full_size * cfg.dataset.test_size)
    train_size = full_size - test_size

    train_dataset, test_dataset = random_split(graphs_list, [train_size, test_size])
    train_loader = HeteroDataLoader(
        train_dataset, batch_size=cfg.dataset.batch_size, shuffle=True
    )
    test_loader = HeteroDataLoader(
        test_dataset, batch_size=cfg.dataset.batch_size, shuffle=False
    )

    return train_loader, test_loader
