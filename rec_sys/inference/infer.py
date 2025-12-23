import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig

from rec_sys.dataset_modules.make_dataset import create_loader
from rec_sys.modules.losses import modified_cos_loss
from rec_sys.modules.neural_models import GraphModel
from rec_sys.modules.trainer import GraphRecSysPL
from pytorch_lightning.loggers import MLFlowLogger


@hydra.main(version_base=None, config_path="../../config", config_name="config")
def main(cfg: DictConfig):
    train_loader, test_loader = create_loader(cfg.data)
    model = GraphModel(**cfg.model)

    pl_model = GraphRecSysPL(model=model, loss_fn=modified_cos_loss, lr=cfg.train.lr)
    mlf_logger = MLFlowLogger(
        experiment_name="graph_rec",
        tracking_uri="file:./mlruns"
    )
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        logger=mlf_logger
    )
    trainer.validate(pl_model, dataloaders=test_loader)
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)


if __name__ == "__main__":
    main()