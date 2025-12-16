import torch


def modified_cos_loss(x, tar):
    return torch.mean(torch.abs(x - tar), dim=0)


def f1_score_metric(y_pred, y_true, eps=1e-8):
    tp = ((y_pred == 1) & (y_true == 1)).sum().float()
    fp = ((y_pred == 1) & (y_true == 0)).sum().float()
    fn = ((y_pred == 0) & (y_true == 1)).sum().float()

    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)

    return 2 * precision * recall / (precision + recall + eps)
