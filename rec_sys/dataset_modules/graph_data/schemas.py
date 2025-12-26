from dataclasses import dataclass, field
from typing import Dict

import polars as pl
import torch
from pydantic import BaseModel, ConfigDict, field_validator
from torch_geometric.data import HeteroData

from rec_sys.dataset_modules.graph_data.graph_dataset_constants import (
    ASIN,
    FEAT_COMMENT,
    FEAT_PRODUCT_INFO,
    FEAT_PRODUCT_NAME,
    TARGET,
    VER_FLAG,
    template,
)


class UserRow(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    asin: pl.Series
    product_info: pl.Series
    comment: pl.Series
    product_name: pl.Series
    verified: pl.Series
    score: pl.Series

    @field_validator("score")
    def check_score_series(cls, v):
        if not isinstance(v, pl.Series):
            raise TypeError(f"Expected Polars Series for score, got {type(v)}")
        if v.dtype not in [pl.Float32, pl.Float64]:
            raise TypeError(f"Expected float Series, got {v.dtype}")
        return v

    @field_validator("verified")
    def check_bool_series(cls, v):
        if not isinstance(v, pl.Series):
            raise TypeError(f"Expected Polars Series, got {type(v)}")
        if v.dtype != pl.Boolean:
            raise TypeError(f"Expected Series with Boolean dtype, got {v.dtype}")
        return v

    def to_tensors(self):
        def series_to_tensor(s: pl.Series):
            if str(s.dtype).startswith("Array(Float32") or str(s.dtype).startswith(
                "Array(Float64"
            ):
                return torch.tensor(s, dtype=torch.float32)
            elif s.dtype == pl.Boolean:
                return torch.tensor(s, dtype=torch.bool)
            else:
                return torch.tensor(s, dtype=torch.float32)

        return {
            ASIN: self.asin,
            FEAT_PRODUCT_INFO: series_to_tensor(self.product_info),
            FEAT_COMMENT: series_to_tensor(self.comment),
            FEAT_PRODUCT_NAME: series_to_tensor(self.product_name),
            VER_FLAG: series_to_tensor(self.verified),
            TARGET: series_to_tensor(self.score),
        }


@dataclass
class EdgeData:
    src: str
    dst: str
    rel: str
    edge_index: torch.Tensor
    edge_features: Dict[str, torch.Tensor] = field(default_factory=dict)


@dataclass
class NodeData:
    features: Dict[str, torch.Tensor]
    num_nodes: int


class HeteroGraphBuilder:
    def __init__(self, template: dict = template):
        self.graph = HeteroData()
        self.template = template

    def add_node(self, node_type: str, node_data: NodeData):
        """Добавляет ноду с фичами и num_nodes"""
        for feat_name, tensor in node_data.features.items():
            self.graph[node_type][feat_name] = tensor
        self.graph[node_type].num_nodes = node_data.num_nodes

    def add_edge(self, edge_data: EdgeData):
        """Добавляет ребро с edge_index и edge_features"""
        self.graph[edge_data.src, edge_data.rel, edge_data.dst].edge_index = (
            edge_data.edge_index
        )
        for feat_name, tensor in edge_data.edge_features.items():
            self.graph[edge_data.src, edge_data.rel, edge_data.dst][feat_name] = tensor

    def build_graph(self, nodes_data: Dict[str, NodeData], edges_data: list[EdgeData]):
        """Построение графа из списка NodeData и EdgeData"""
        for node_type, node in nodes_data.items():
            self.add_node(node_type, node)
        for edge in edges_data:
            self.add_edge(edge)
        return self.graph

    def check_structure(self):
        for key, sub_template in self.template.items():
            if key not in self.graph:
                print(f"Missing key: {key}")
                return False
            if isinstance(sub_template, dict):
                sub_graph = self.graph[key]
                for sub_key in sub_template:
                    if sub_key not in sub_graph:
                        print(f"Missing sub-key: {sub_key} in {key}")
                        return False
        return True

    def get_graph(self):
        if not self.check_structure():
            raise ValueError("does not template")
        return self.graph
