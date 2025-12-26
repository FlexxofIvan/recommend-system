import pickle
from pathlib import Path

import hydra
import polars as pl
import torch
from annoy import AnnoyIndex
from omegaconf import DictConfig

from rec_sys.dataset_modules.outer_data_utils import ensure_data_available
from rec_sys.modules.neural_models import GraphModel


@hydra.main(version_base=None, config_path="../../config", config_name="infer_config")
def main(cfg: DictConfig):
    parquet_path = Path(cfg.vector_db.source_metadata_path)
    ensure_data_available(parquet_path)
    meta_df = pl.read_parquet(parquet_path)
    name_feat = torch.tensor(meta_df["name"].to_list(), dtype=torch.float32)
    desc_feat = torch.tensor(meta_df["description"].to_list(), dtype=torch.float32)
    prod_idxs = meta_df["asin"].to_list()
    checkpoint_path = Path(cfg.checkpoint)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path)

    if "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        state_dict = {k.replace("model.", ""): v for k, v in state_dict.items()}
    else:
        state_dict = checkpoint

    model = GraphModel(**cfg.model)
    model.load_state_dict(state_dict)
    model.eval()
    with torch.no_grad():
        _ , prod_vecs = model(
            prods_only=True,
            product_info_features=desc_feat,
            product_name_features=name_feat,
        )

    dim = prod_vecs.shape[1]
    index = AnnoyIndex(dim, "angular")
    for i, vec in enumerate(prod_vecs.cpu().numpy()):
        index.add_item(i, vec.tolist())
    index.build(10)

    save_dir = Path(cfg.vector_db.index_path).parent
    save_dir.mkdir(parents=True, exist_ok=True)

    index_path = Path(cfg.vector_db.index_path)
    index.save(str(index_path))

    mapping_path = Path(cfg.vector_db.mapping_path)
    with open(mapping_path, "wb") as f:
        pickle.dump(prod_idxs, f)

    print(f"Annoy index saved to {index_path}")
    print(f"Mapping ASINs saved to {save_dir / 'prod_idx_map.pkl'}")


if __name__ == "__main__":
    main()
