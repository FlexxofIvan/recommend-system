import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics import Accuracy, F1Score

from rec_sys.dataset_modules.graph_data.graph_dataset_constants import (
    ASIN,
    EDGE_INDEX,
    FEAT_PRODUCT_INFO,
    FEAT_PRODUCT_NAME,
    PRODUCT_TEST_NODE,
    PRODUCT_TRAIN_NODE,
    TARGET,
    USER_EMB,
    USER_NODE,
    USER_REL_PRODUCT,
)
from rec_sys.modules.neural_models import GraphModel


def _build_model_kwargs(batch, train: bool):
    if train:
        product_node = PRODUCT_TRAIN_NODE
    else:
        product_node = PRODUCT_TEST_NODE

    return dict(
        user_features=batch[USER_NODE][USER_EMB],
        product_info_features=batch[product_node][FEAT_PRODUCT_INFO],
        product_name_features=batch[product_node][FEAT_PRODUCT_NAME],
        edge_index=batch[PRODUCT_TRAIN_NODE, USER_REL_PRODUCT, USER_NODE][EDGE_INDEX],
        edge_attr=batch[PRODUCT_TRAIN_NODE, USER_REL_PRODUCT, USER_NODE][
            USER_REL_PRODUCT
        ],
    )


class GraphRecSysPL(pl.LightningModule):
    def __init__(self, model: GraphModel, loss_fn, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr
        self.loss_fn = loss_fn
        self.val_f1 = F1Score(task="binary")
        self.val_acc = Accuracy(task="binary")

    def _build_model_kwargs(self, batch, user_mode: bool):
        if user_mode:
            product_node = PRODUCT_TRAIN_NODE
        else:
            product_node = PRODUCT_TEST_NODE

        return dict(
            user_features=batch[USER_NODE][USER_EMB],
            product_info_features=batch[product_node][FEAT_PRODUCT_INFO],
            product_name_features=batch[product_node][FEAT_PRODUCT_NAME],
            edge_index=batch[PRODUCT_TRAIN_NODE, USER_REL_PRODUCT, USER_NODE][
                EDGE_INDEX
            ],
            edge_attr=batch[PRODUCT_TRAIN_NODE, USER_REL_PRODUCT, USER_NODE][
                USER_REL_PRODUCT
            ],
        )

    def forward(self, batch, prods_only: bool, user_mode: bool):
        kwargs = self._build_model_kwargs(batch, user_mode=user_mode)

        if prods_only:
            _, prod_embs = self.model(
                prods_only=True,
                product_info_features=kwargs["product_info_features"],
                product_name_features=kwargs["product_name_features"],
            )
            return prod_embs
        user_emb, prod_emb = self.model(prods_only=False, **kwargs)
        return user_emb, prod_emb

    def training_step(self, batch, batch_idx):
        mask = batch[PRODUCT_TRAIN_NODE].batch
        batch_size = torch.max(mask) + 1

        user_emb, prod_emb = self(
            batch,
            prods_only=False,
            user_mode=True,
        )

        user_emb_exp = user_emb[mask]
        target = batch[PRODUCT_TRAIN_NODE][TARGET]

        predicted_score = 2.5 * (F.cosine_similarity(user_emb_exp, prod_emb) + 1)
        loss = self.loss_fn(predicted_score, target)

        self.log(
            "train_loss", loss, on_epoch=True, prog_bar=True, batch_size=batch_size
        )
        return loss

    def validation_step(self, batch, batch_idx):
        mask = batch[PRODUCT_TEST_NODE].batch
        batch_size = torch.max(mask) + 1

        user_emb, _ = self(
            batch,
            prods_only=False,
            user_mode=True,
        )

        prod_emb = self(
            batch,
            prods_only=True,
            user_mode=False,
        )

        user_emb_exp = user_emb[mask]
        target = batch[PRODUCT_TEST_NODE][TARGET]

        predicted_score = 2.5 * (F.cosine_similarity(user_emb_exp, prod_emb) + 1)

        true_flags = target > 3
        predicted_flags = predicted_score > 3

        f1 = self.val_f1(predicted_flags, true_flags)
        acc = self.val_acc(predicted_flags, true_flags)
        loss = self.loss_fn(predicted_score, target)

        self.log("val_f1", f1, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_acc", acc, prog_bar=True, on_epoch=True, batch_size=batch_size)
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, batch_size=batch_size)

        return loss

    def predict_step(self, batch):
        user_emb, _ = self(
            batch,
            prods_only=False,
            user_mode=True,
        )
        return user_emb, batch[PRODUCT_TEST_NODE][ASIN]

    def configure_optimizers(self):
        return torch.optim.Adam(self.model.parameters(), lr=self.lr)
