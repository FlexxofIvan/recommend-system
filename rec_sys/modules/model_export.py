from collections import OrderedDict

import fire
import torch
from omegaconf import OmegaConf
from torch.export import Dim

from rec_sys.dataset_modules.graph_data.graph_data_utils import make_edges
from rec_sys.modules.neural_models import GraphModel
from pathlib import Path
import random

from triton_utils.triton_wrapper import TritonWrapper

hidden_dim = 384
max_products = 60

def export_onnx(checkpoint_path: str, model_config_path: Path, output_path: str):

    model_cfg = OmegaConf.load(model_config_path)
    model_config = OmegaConf.to_container(model_cfg, resolve=True)

    checkpoint = torch.load(checkpoint_path)
    state_dict = checkpoint.get("state_dict", checkpoint)
    state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}

    model = GraphModel(**model_config)
    model.load_state_dict(state_dict)
    model.eval()

    product_amount = random.randint(1, max_products)
    dummy_product_info_features = torch.randn(product_amount, hidden_dim)
    dummy_product_name_features = torch.randn(product_amount, hidden_dim)
    dummy_user_features = torch.randn(1, hidden_dim)
    dummy_edge_index = make_edges(product_amount)
    dummy_edge_attr = torch.randn(product_amount, hidden_dim)

    wrapper = TritonWrapper(model)
    torch.onnx.export(
        wrapper,
        (dummy_user_features, dummy_product_info_features, dummy_product_name_features, dummy_edge_index,
         dummy_edge_attr),
        output_path,
        input_names=["user_features", "product_info_features", "product_name_features", "edge_index", "edge_attr"],
        output_names=["user_emb", "prod_emb"],
        opset_version=18,
        dynamic_shapes={
            "user_features": {0: Dim("num_users", min=1, max=None)},
            "product_info_features": {0: Dim("num_products", min=1, max=max_products)},
            "product_name_features": {0: Dim("num_products", min=1, max=max_products)},
            "edge_index": {1: Dim("num_edges", min=1, max=max_products * (max_products - 1))},
            "edge_attr": {0: Dim("num_products", min=1, max=max_products)}
        }
    )

    print(f"ONNX модель сохранена в {output_path}")


if __name__ == "__main__":
    fire.Fire(export_onnx)