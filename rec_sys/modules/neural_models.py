import torch.nn as nn
import torch
from torch_geometric.nn import TransformerConv

class NodeLayer(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(inp_dim, 2 * inp_dim),
                                   nn.LayerNorm(2 * inp_dim),
                                   nn.ReLU(),
                                   nn.Linear(2 * inp_dim, out_dim)
                                  )
    def forward(self, x):
        return self.layer(x)


class EdgeLayer(nn.Module):
    def __init__(self, inp_dim: int, out_dim: int):
        super().__init__()
        self.layer = nn.Sequential(nn.Linear(inp_dim, 2 * inp_dim),
                                   nn.LayerNorm(2 * inp_dim),
                                   nn.ReLU(),
                                   nn.Linear(2 * inp_dim, out_dim)
                                  )
    def forward(self, x):
        return self.layer(x)


class GraphModel(nn.Module):
    def __init__(self,
                 inp_dim_node: int,
                 out_dim_node: int,
                 inp_dim_edge: int,
                 out_dim_edge: int,
                 heads: int
                 ):
        super().__init__()
        self.user_layer = NodeLayer(inp_dim_node, out_dim_node)
        self.name_product_layer = NodeLayer(inp_dim_node, out_dim_node)
        self.info_product_layer = NodeLayer(inp_dim_node, out_dim_node)
        self.edge_layer = EdgeLayer(inp_dim_edge, out_dim_edge)
        self.product_proj_layer = nn.Sequential(nn.Linear(2 * out_dim_node, out_dim_node))
        if out_dim_node % heads != 0:
            raise ValueError('missmatch')
        self.conv = TransformerConv(in_channels=out_dim_node,
                                    out_channels=out_dim_node // heads,
                                    edge_dim=out_dim_edge,
                                    heads=heads,
                                    concat=True)

    def product_vec_generate(self, product_info_features, product_name_features):
        product_info_emb = self.info_product_layer(product_info_features)
        product_name_emb = self.name_product_layer(product_name_features)
        product_cat = torch.cat([product_name_emb, product_info_emb], dim=-1)
        return self.product_proj_layer(product_cat)

    def forward(self, prods_only, user_features, product_info_features, product_name_features, edge_index, edge_attr):
        offset = torch.max(edge_index[1, :]) + 1
        edge_index[0, :] = edge_index[0, :] + offset
        user_emb = self.user_layer(user_features)

        user_size = user_emb.shape[0]

        product_info_emb = self.info_product_layer(product_info_features)
        product_name_emb = self.name_product_layer(product_name_features)
        product_cat = torch.cat([product_name_emb, product_info_emb], dim=-1)
        product_emb = self.product_proj_layer(product_cat)
        if prods_only:
            return product_emb
        full_embs = torch.cat([user_emb, product_emb], dim=0)
        edge_embs = self.edge_layer(edge_attr)
        out = self.conv(x=full_embs, edge_index=edge_index, edge_attr=edge_embs)
        user_emb = out[:user_size]
        return user_emb, product_emb