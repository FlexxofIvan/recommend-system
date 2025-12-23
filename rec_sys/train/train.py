from datetime import datetime
from pathlib import Path

import hydra
import pytorch_lightning as pl
import torch
from omegaconf import DictConfig
from pytorch_lightning.loggers import MLFlowLogger

from rec_sys.dataset_modules.make_dataset import create_loaders
from rec_sys.modules.losses import modified_cos_loss
from rec_sys.modules.neural_models import GraphModel
from rec_sys.modules.trainer import GraphRecSysPL


@hydra.main(version_base=None, config_path="../../config", config_name="train_config")
def main(cfg: DictConfig):
    train_loader, test_loader = create_loaders(cfg.data)
    model = GraphModel(**cfg.model)

    pl_model = GraphRecSysPL(model=model, loss_fn=modified_cos_loss, lr=cfg.train.lr)
    mlf_logger = MLFlowLogger(experiment_name="graph_rec", tracking_uri="file:./mlruns")
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1 if torch.cuda.is_available() else None,
        log_every_n_steps=10,
        logger=mlf_logger,
    )
    trainer.validate(pl_model, dataloaders=test_loader)
    trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=test_loader)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    save_dir = Path(cfg.train.save_path)
    save_dir.mkdir(parents=True, exist_ok=True)
    trainer.save_checkpoint(
        save_dir / f"graph_recsys_{timestamp}.ckpt", weights_only=True
    )


if __name__ == "__main__":
    main()
