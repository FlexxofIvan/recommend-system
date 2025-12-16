import torch
import torch.nn.functional as F

def modified_cos_loss(x, y, tar):
    return torch.mean(torch.abs(2.5 * (F.cosine_similarity(x, y) + 1) - tar), dim=0)