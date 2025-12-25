import fire
import torch
from omegaconf import OmegaConf

from rec_sys.dataset_modules.graph_data.graph_data_utils import make_edges
from rec_sys.modules.neural_models import GraphModel
from rec_sys.modules.trainer import GraphRecSysPL
from pathlib import Path
import random

hidden_dim = 384
max_products = 5

def export_torchscript(
    checkpoint_path: str,
    model_config_path: Path,
    output_path: str = "graph_model.pt",
):
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

    traced_model = torch.jit.trace(
        model,
        (
            dummy_user_features,
            dummy_product_info_features,
            dummy_product_name_features,
            dummy_edge_index,
            dummy_edge_attr,
            False
        )
    )
    traced_model.save(output_path)
    print(f"TorchScript модель сохранена в {output_path}, product_amount={product_amount}")

if __name__ == "__main__":
    fire.Fire(export_torchscript)