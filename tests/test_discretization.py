import torch

from src.models.logic_layer import DiffLogicLayer
from src.utils.connectivity import build_connectivity


def test_argmax_matches_manual():
    conn = build_connectivity(32, 8, seed=99)
    layer = DiffLogicLayer(32, 8, connectivity=conn, gate_init="normal")
    with torch.no_grad():
        layer.gate_logits.copy_(torch.randn(8, 16))
    ids = layer.gate_logits.argmax(dim=-1)
    manual = torch.stack([layer.gate_logits[i].argmax() for i in range(8)])
    assert torch.equal(ids, manual)
