import torch

from src.utils.output_grouping import compute_class_scores


def test_grouping_tau_beta():
    B, D, C = 4, 20, 10
    x = torch.arange(B * D, dtype=torch.float32).view(B, D)
    tau = 10.0
    beta = 0.5
    scores = compute_class_scores(x, C, tau, beta)
    assert scores.shape == (B, C)
    # group_size = 2; first row class 0 = x[0,0]+x[0,1]
    g0 = x[0, 0] + x[0, 1]
    assert torch.isclose(scores[0, 0], g0 / tau + beta)


def test_raises_if_not_divisible():
    x = torch.zeros(2, 21)
    try:
        compute_class_scores(x, 10, 1.0, 0.0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
