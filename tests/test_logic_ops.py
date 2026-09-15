"""Tests for the 16 logic operators: truth-table and range checks."""

import torch
import pytest
from src.ops.logic_ops import compute_all_16_ops, build_truth_table, NUM_OPERATORS

# Expected Boolean truth tables for all 16 operators, indexed [a][b].
EXPECTED_TRUTH_TABLES = {
    0:  [[0, 0], [0, 0]],  # FALSE
    1:  [[0, 0], [0, 1]],  # AND
    2:  [[0, 0], [1, 0]],  # A AND NOT B
    3:  [[0, 0], [1, 1]],  # A
    4:  [[0, 1], [0, 0]],  # NOT A AND B
    5:  [[0, 1], [0, 1]],  # B
    6:  [[0, 1], [1, 0]],  # XOR
    7:  [[0, 1], [1, 1]],  # OR
    8:  [[1, 0], [0, 0]],  # NOR
    9:  [[1, 0], [0, 1]],  # XNOR
    10: [[1, 0], [1, 0]],  # NOT B
    11: [[1, 0], [1, 1]],  # B=>A
    12: [[1, 1], [0, 0]],  # NOT A
    13: [[1, 1], [0, 1]],  # A=>B
    14: [[1, 1], [1, 0]],  # NAND
    15: [[1, 1], [1, 1]],  # TRUE
}


class TestTruthTables:
    """Verify all 16 operators match Boolean truth tables on binary inputs."""

    @pytest.mark.parametrize("op_id", range(NUM_OPERATORS))
    @pytest.mark.parametrize("a_val", [0, 1])
    @pytest.mark.parametrize("b_val", [0, 1])
    def test_single_binary_input(self, op_id, a_val, b_val):
        a = torch.tensor(float(a_val))
        b = torch.tensor(float(b_val))
        result = compute_all_16_ops(a, b)
        actual = result[op_id].item()
        expected = EXPECTED_TRUTH_TABLES[op_id][a_val][b_val]
        assert actual == pytest.approx(expected, abs=1e-7), (
            f"Op {op_id}: f({a_val},{b_val}) = {actual}, expected {expected}"
        )

    def test_batched_binary(self):
        """All 4 binary combos in a batch, check all 16 ops at once."""
        a = torch.tensor([0.0, 0.0, 1.0, 1.0])
        b = torch.tensor([0.0, 1.0, 0.0, 1.0])
        result = compute_all_16_ops(a, b)  # [4, 16]
        assert result.shape == (4, 16)

        for op_id in range(NUM_OPERATORS):
            for idx, (av, bv) in enumerate([(0,0),(0,1),(1,0),(1,1)]):
                actual = result[idx, op_id].item()
                expected = EXPECTED_TRUTH_TABLES[op_id][av][bv]
                assert actual == pytest.approx(expected, abs=1e-7)

    def test_build_truth_table_matches(self):
        tt = build_truth_table()
        assert tt.shape == (16, 2, 2)
        for op_id in range(NUM_OPERATORS):
            for a_val in range(2):
                for b_val in range(2):
                    assert tt[op_id, a_val, b_val].item() == EXPECTED_TRUTH_TABLES[op_id][a_val][b_val]


class TestOperatorRange:
    """Verify real-valued operators produce sensible outputs for [0,1] inputs."""

    def test_outputs_finite(self):
        torch.manual_seed(123)
        a = torch.rand(1000)
        b = torch.rand(1000)
        result = compute_all_16_ops(a, b)
        assert torch.all(torch.isfinite(result))

    def test_outputs_in_unit_interval(self):
        torch.manual_seed(456)
        a = torch.rand(1000)
        b = torch.rand(1000)
        result = compute_all_16_ops(a, b)
        assert result.min().item() >= -1e-7
        assert result.max().item() <= 1.0 + 1e-7

    def test_false_always_zero(self):
        a = torch.rand(100)
        b = torch.rand(100)
        result = compute_all_16_ops(a, b)
        assert torch.allclose(result[..., 0], torch.zeros(100))

    def test_true_always_one(self):
        a = torch.rand(100)
        b = torch.rand(100)
        result = compute_all_16_ops(a, b)
        assert torch.allclose(result[..., 15], torch.ones(100))

    def test_output_shape_multidim(self):
        a = torch.rand(8, 32)
        b = torch.rand(8, 32)
        result = compute_all_16_ops(a, b)
        assert result.shape == (8, 32, 16)
