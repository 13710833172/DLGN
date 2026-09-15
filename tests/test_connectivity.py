import torch

from src.utils.connectivity import build_connectivity


def test_same_seed_same_connectivity():
    a = build_connectivity(784, 260, seed=12345)
    b = build_connectivity(784, 260, seed=12345)
    assert a.shape == (260, 2)
    assert torch.equal(a, b)


def test_different_seed_usually_different():
    a = build_connectivity(784, 260, seed=1)
    b = build_connectivity(784, 260, seed=2)
    assert not torch.equal(a, b)
