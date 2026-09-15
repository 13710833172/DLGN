import torch

from src.models.hard_logic_net import HardLogicNet
from src.models.logic_net import DiffLogicNet
from src.utils.metrics import soft_hard_layerwise_agreement


@torch.no_grad()
def test_near_one_hot_agrees_on_binary():
    torch.manual_seed(0)
    n_in, n_hid, L = 64, 32, 2
    model = DiffLogicNet(
        input_dim=n_in,
        num_layers=L,
        neurons_per_layer=n_hid,
        connectivity_seed=7,
        gate_init="zeros",
    )
    # Make logits very peaked toward gate 3 (A)
    for layer in model.layers:
        layer.gate_logits.zero_()
        layer.gate_logits[:, 3] = 80.0

    hard = HardLogicNet.from_soft_model(model)
    x = torch.randint(0, 2, (16, n_in)).float()
    soft_out = model(x)
    hard_out = hard(x)
    assert torch.allclose(soft_out, hard_out, atol=0.05)


@torch.no_grad()
def test_soft_hard_layerwise_agreement_length_and_high_match():
    """Regression: unpack (final, acts, probs) correctly so zip matches hard layers."""
    torch.manual_seed(1)
    n_in, n_hid, L = 32, 16, 3
    model = DiffLogicNet(
        input_dim=n_in,
        num_layers=L,
        neurons_per_layer=n_hid,
        connectivity_seed=3,
        gate_init="zeros",
    )
    for layer in model.layers:
        layer.gate_logits.zero_()
        layer.gate_logits[:, 3] = 80.0
    hard = HardLogicNet.from_soft_model(model)
    x = torch.randint(0, 2, (8, n_in)).float()
    rates = soft_hard_layerwise_agreement(model, hard, x, tol=0.05)
    assert len(rates) == L
    assert all(r > 0.99 for r in rates)
