from pathlib import Path

import polars as pl
import torch
from rec_sys.dataset_modules.graph_data.graph_data_utils import split_tensors
from rec_sys.dataset_modules.graph_data.graph_data_utils import make_edges
from rec_sys.dataset_modules.graph_data.schemas import UserRow, HeteroGraphBuilder, NodeData, EdgeData

from rec_sys.dataset_modules.graph_data.graph_dataset_constants import (FEAT_COMMENT,
                                     FEAT_PRODUCT_NAME, FEAT_PRODUCT_INFO,
                                     TARGET, USER_EMB,
                                     PRODUCT_TRAIN_NODE, PRODUCT_TEST_NODE,
                                     USER_NODE, USER_REL_PRODUCT)


def build_user_graphs(
        user_dir: Path,
        save_dir: Path,
        eval_ratio: float,
        target_threshold: int = 3,
        max_class_diff: int = 2
):
    """
    Строит графы для пользователей из parquet файлов и сохраняет их.

    Args:
        user_dir: директория с parquet файлами пользователей
        save_dir: директория для сохранения графов
        eval_ratio: доля данных для теста
        target_threshold: порог для бинаризации таргета (> threshold = 1)
        max_class_diff: макс. разница между количеством классов для фильтрации
    """
    for num, file in enumerate(user_dir.iterdir()):
        user_df = pl.read_parquet(file)
        data_dict = {key: user_df[key] for key in user_df.columns}

        user_row = UserRow(**data_dict)
        features_tensor_dict = user_row.to_tensors()

        n = features_tensor_dict[FEAT_PRODUCT_INFO].shape[0]
        if any(t.shape[0] != n for t in features_tensor_dict.values()):
            print(f"[WARN] size mismatch in {file}")
            continue

        train, test, (train_n, test_n) = split_tensors(
            features_tensor_dict,
            eval_ratio
        )
        if train is None:
            continue

        avg_user_vec = torch.mean(train[FEAT_COMMENT], dim=0).unsqueeze(0)

        train_num_nodes = train[FEAT_PRODUCT_INFO].shape[0]
        test_num_nodes = test[FEAT_PRODUCT_INFO].shape[0]

        binary_marks = (train[TARGET] > target_threshold).long()
        count = torch.bincount(binary_marks, minlength=2)
        if count[1] - count[0] > max_class_diff or count[0] == 0:
            continue

        nodes_data = {
            USER_NODE: NodeData(features={USER_EMB: avg_user_vec}, num_nodes=1),
            PRODUCT_TRAIN_NODE: NodeData(
                features={
                    FEAT_PRODUCT_NAME: train[FEAT_PRODUCT_NAME],
                    FEAT_PRODUCT_INFO: train[FEAT_PRODUCT_INFO],
                    TARGET: train[TARGET]
                },
                num_nodes=train_num_nodes
            ),
            PRODUCT_TEST_NODE: NodeData(
                features={
                    FEAT_PRODUCT_NAME: test[FEAT_PRODUCT_NAME],
                    FEAT_PRODUCT_INFO: test[FEAT_PRODUCT_INFO],
                    TARGET: test[TARGET]
                },
                num_nodes=test_num_nodes
            )
        }

        edge_index = make_edges(train_num_nodes)

        edges_data = [
            EdgeData(
                src=PRODUCT_TRAIN_NODE,
                dst=USER_NODE,
                rel=USER_REL_PRODUCT,
                edge_index=edge_index,
                edge_features={USER_REL_PRODUCT: train[FEAT_COMMENT]}
            )
        ]

        builder = HeteroGraphBuilder()
        bipart_data = builder.build_graph(nodes_data, edges_data)
        save_path = save_dir / f'graph_{num}.pt'
        torch.save(bipart_data, save_path)
