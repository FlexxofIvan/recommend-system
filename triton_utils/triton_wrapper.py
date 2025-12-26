from torch import nn


class TritonWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(
        self,
        user_features,
        product_info_features,
        product_name_features,
        edge_index,
        edge_attr,
    ):
        return self.model.forward_triton(
            user_features=user_features,
            product_info_features=product_info_features,
            product_name_features=product_name_features,
            edge_index=edge_index,
            edge_attr=edge_attr,
        )
