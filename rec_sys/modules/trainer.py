import pytorch_lightning as pl
import torch
import torch.nn as nn

from rec_sys.dataset_modules.graph_data.graph_dataset_constants import (FEAT_COMMENT,
                                     FEAT_PRODUCT_NAME, FEAT_PRODUCT_INFO,
                                     TARGET, USER_EMB,
                                     PRODUCT_TRAIN_NODE, PRODUCT_TEST_NODE,
                                     USER_NODE, USER_REL_PRODUCT)



class GraphRecSysPL(pl.LightningModule):
    def __init__(self, model: nn.Module, loss_fn, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn

    def forward(self, batch, prods_only: bool):
        input_kwargs = {
            "user_features": batch[USER_NODE][USER_EMB],
            "product_info_features": batch[PRODUCT_TRAIN_NODE][FEAT_PRODUCT_INFO],
            "product_name_features": batch[PRODUCT_TRAIN_NODE][FEAT_PRODUCT_NAME],
            "edge_index": batch[PRODUCT_TRAIN_NODE, USER_REL_PRODUCT, USER_NODE]["edge_index"],
            "edge_attr": batch[PRODUCT_TRAIN_NODE, USER_REL_PRODUCT, USER_NODE][USER_REL_PRODUCT],
        }
        return self.model(prods_only, **input_kwargs)

    def training_step(self, batch, batch_idx):
        mask = batch[PRODUCT_TRAIN_NODE].batch
        batch_size = torch.max(mask) + 1
        user_emb, prod_emb = self(batch, prods_only=False)

        user_emb_exp = user_emb[mask]
        target = batch[PRODUCT_TRAIN_NODE][TARGET]

        loss = self.loss_fn(user_emb_exp, prod_emb, target)
        self.log("train_loss", loss, batch_size=batch_size, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        mask = batch[PRODUCT_TRAIN_NODE].batch
        user_emb, _ = self(batch, prods_only=False)
        prod_emb = self(batch, prods_only=True)
        user_emb_exp = user_emb[mask]
        target = batch[PRODUCT_TRAIN_NODE][TARGET]
        batch_size = torch.max(mask) + 1
        loss = self.loss_fn(user_emb_exp, prod_emb, target)
        self.log("val_loss", loss, on_step=False, batch_size=batch_size, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
