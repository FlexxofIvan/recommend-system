import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from rec_sys.dataset_modules.make_dataset import create_loaders
from rec_sys.modules.losses import modified_cos_loss
from rec_sys.modules.neural_models import GraphModel
from rec_sys.modules.trainer import GraphRecSysPL


@hydra.main(
    version_base=None, config_path="../../config", config_name="infer_config.yaml"
)
def main(cfg: DictConfig):
    _, test_loader = create_loaders(cfg.data)
    model = GraphModel(**cfg.model)

    pl_model = GraphRecSysPL(model=model, loss_fn=modified_cos_loss)
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
    )
    predictions = trainer.predict(pl_model, dataloaders=test_loader)

    return predictions


if __name__ == "__main__":
    main()
