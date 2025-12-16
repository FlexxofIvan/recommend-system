from typing import Dict
import torch


def split_train_test(n: int, ratio: float):
    test_n = max(1, int(n * ratio))
    train_n = n - test_n
    return train_n, test_n

def split_tensors(tensors: Dict[str, torch.Tensor], ratio: float):
    """
    Делит набор тензоров одинаковой длины по ratio.

    tensors: {"name": tensor[B, ...], ...}
    ratio: доля train (0 < ratio < 1)

    return:
        train: {"name": tensor[train_n, ...]}
        test: {"name": tensor[test_n, ...]}
        (train_n, test_n)
    """
    names = list(tensors.keys())
    n = tensors[names[0]].shape[0]

    for name, t in tensors.items():
        if t.shape[0] != n:
            raise ValueError(f"tensor '{name}' has size {t.shape[0]} but expected {n}")

    train_n, test_n = split_train_test(n, ratio)

    if train_n == 0 or test_n == 0:
        return None, None, (train_n, test_n)

    idx_train = slice(0, train_n)
    idx_test = slice(train_n, n)

    train = {name: t[idx_train] for name, t in tensors.items()}
    test = {name: t[idx_test] for name, t in tensors.items()}

    return train, test, (train_n, test_n)

def make_edges(n: int):
    return torch.cat([torch.arange(n, dtype=int).unsqueeze(0), torch.zeros(n, dtype=int).unsqueeze(0)], dim=0)

