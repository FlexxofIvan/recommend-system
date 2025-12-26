from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from rec_sys.dataset_modules.graph_data.graph_dataset_constants import ASIN
from rec_sys.dataset_modules.make_dataset import create_loaders
from rec_sys.modules.losses import modified_cos_loss
from rec_sys.modules.neural_models import GraphModel
from rec_sys.modules.trainer import GraphRecSysPL
from rec_sys.vector_database.vectorbase_utils import load_index, search_vector


@hydra.main(
    version_base=None, config_path="../../config", config_name="infer_config.yaml"
)
def main(cfg: DictConfig):
    _, test_loader = create_loaders(cfg.data)
    model = GraphModel(**cfg.model)

    pl_model = GraphRecSysPL(model=model, loss_fn=modified_cos_loss)
    trainer = pl.Trainer(
        accelerator="auto",
        devices=1,
    )
    prediction = trainer.predict(pl_model, dataloaders=test_loader)

    user_embs = torch.cat([e for e, _ in prediction], dim=0).cpu().numpy()
    index_path = Path(cfg.vector_db.index_path)
    mapping_path = Path(cfg.vector_db.mapping_path)

    dim = user_embs.shape[1]

    index, prod_idxs = load_index(index_path, mapping_path, dim)
    rec_products = []
    for query_vec in user_embs:
        results = search_vector(query_vec, index, prod_idxs, top_k=cfg.vector_db.top_k)
        user_rec = [result[ASIN] for result in results]
        rec_products.append(user_rec)
    return prediction


if __name__ == "__main__":
    main()
