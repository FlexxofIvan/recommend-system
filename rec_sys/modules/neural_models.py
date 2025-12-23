import torch
import torch.nn as nn
from torch_geometric.nn import TransformerConv


class NodeLayer(nn.Module):
    """
    Feed-forward block for node feature transformation.

    This module is used to encode node features (e.g., users or products)
    into a latent embedding space.

    Architecture:
        Linear → LayerNorm → ReLU → Linear

    Parameters
    ----------
    inp_dim : int
        Dimensionality of input node features.
    out_dim : int
        Dimensionality of output node embeddings.
    """

    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(inp_dim, 2 * inp_dim),
            nn.LayerNorm(2 * inp_dim),
            nn.ReLU(),
            nn.Linear(2 * inp_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the node transformation.

        Parameters
        ----------
        x : torch.Tensor
            Node feature tensor of shape (N, inp_dim).

        Returns
        -------
        torch.Tensor
            Node embedding tensor of shape (N, out_dim).
        """
        return self.layer(x)


class EdgeLayer(nn.Module):
    """
    Feed-forward block for edge feature transformation.

    Encodes edge attributes before they are passed to the
    attention-based graph convolution.

    Parameters
    ----------
    inp_dim : int
        Dimensionality of input edge features.
    out_dim : int
        Dimensionality of output edge embeddings.
    """

    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(inp_dim, 2 * inp_dim),
            nn.LayerNorm(2 * inp_dim),
            nn.ReLU(),
            nn.Linear(2 * inp_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply the edge transformation.

        Parameters
        ----------
        x : torch.Tensor
            Edge feature tensor of shape (E, inp_dim).

        Returns
        -------
        torch.Tensor
            Edge embedding tensor of shape (E, out_dim).
        """
        return self.layer(x)


class GraphModel(nn.Module):
    """
    Graph neural network for joint user–product representation learning.

    The model:
    - encodes users and products with separate MLPs,
    - fuses multiple product representations (name + info),
    - propagates information over the graph using TransformerConv.

    Nodes:
        - Users
        - Products

    Edges may carry attributes, which are incorporated into
    the attention mechanism.

    Parameters
    ----------
    inp_dim_node : int
        Dimensionality of input node features.
    out_dim_node : int
        Dimensionality of output node embeddings.
    inp_dim_edge : int
        Dimensionality of input edge features.
    out_dim_edge : int
        Dimensionality of output edge embeddings.
    heads : int
        Number of attention heads in TransformerConv.
    """

    def __init__(
        self,
        inp_dim_node: int,
        out_dim_node: int,
        inp_dim_edge: int,
        out_dim_edge: int,
        heads: int,
    ):
        super().__init__()

        self.user_layer = NodeLayer(inp_dim_node, out_dim_node)
        self.name_product_layer = NodeLayer(inp_dim_node, out_dim_node)
        self.info_product_layer = NodeLayer(inp_dim_node, out_dim_node)
        self.edge_layer = EdgeLayer(inp_dim_edge, out_dim_edge)

        self.product_proj_layer = nn.Sequential(
            nn.Linear(2 * out_dim_node, out_dim_node)
        )

        if out_dim_node % heads != 0:
            raise ValueError("out_dim_node must be divisible by heads")

        self.conv = TransformerConv(
            in_channels=out_dim_node,
            out_channels=out_dim_node // heads,
            edge_dim=out_dim_edge,
            heads=heads,
            concat=True,
        )

    def product_vec_generate(
        self,
        product_info_features: torch.Tensor,
        product_name_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Generate product embeddings without graph propagation.

        This method encodes and fuses product features only,
        without applying graph convolution.

        Parameters
        ----------
        product_info_features : torch.Tensor
            Product metadata/features of shape (P, D).
        product_name_features : torch.Tensor
            Product name features of shape (P, D).

        Returns
        -------
        torch.Tensor
            Product embedding tensor of shape (P, out_dim_node).
        """
        product_info_emb = self.info_product_layer(product_info_features)
        product_name_emb = self.name_product_layer(product_name_features)
        product_cat = torch.cat([product_name_emb, product_info_emb], dim=-1)
        return self.product_proj_layer(product_cat)

    def forward(
        self,
        *,
        prods_only: bool,
        product_info_features: torch.Tensor,
        product_name_features: torch.Tensor,
        user_features: torch.Tensor | None = None,
        edge_index: torch.Tensor | None = None,
        edge_attr: torch.Tensor | None = None,
    ):
        if prods_only:
            return self.product_vec_generate(
                product_info_features,
                product_name_features,
            )

        # --- full graph mode ---
        if user_features is None:
            raise ValueError("user_features must be provided when prods_only=False")
        if edge_index is None or edge_attr is None:
            raise ValueError(
                "edge_index and edge_attr must be provided when prods_only=False"
            )

        user_emb = self.user_layer(user_features)
        user_size = user_emb.size(0)

        product_emb = self.product_vec_generate(
            product_info_features,
            product_name_features,
        )

        full_embs = torch.cat([user_emb, product_emb], dim=0)
        edge_embs = self.edge_layer(edge_attr)

        out = self.conv(
            x=full_embs,
            edge_index=edge_index,
            edge_attr=edge_embs,
        )

        return out[:user_size], product_emb
