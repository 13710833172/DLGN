"""Tests for CIFAR-10 paper binary embedding and config validation."""

import pytest
import torch

from src.data.cifar10 import (
    CIFAR10_PAPER_BINARY_INPUT_DIM,
    CIFAR10_PAPER_LARGE_INPUT_DIM,
    CIFAR10_PAPER_LARGE_THRESHOLDS,
    CIFAR10_PAPER_THRESHOLD_LEVELS,
    _embed_cifar10_paper_binary,
    _embed_cifar10_paper_large,
)
from src.data.loaders import cifar10_embedding_mode, validate_model_data_config


def test_embedding_dimension():
    x = torch.rand(3, 32, 32)
    v = _embed_cifar10_paper_binary(x)
    assert v.shape == (CIFAR10_PAPER_BINARY_INPUT_DIM,)
    assert v.min() >= 0 and v.max() <= 1


def test_threshold_monotonicity():
    """Higher channel value yields 1 on more cumulative thresholds."""
    x = torch.zeros(3, 32, 32)
    x[0, 0, 0] = 0.24
    v_lo = _embed_cifar10_paper_binary(x)
    x[0, 0, 0] = 0.26
    v_hi = _embed_cifar10_paper_binary(x)
    # First chunk of 1024 floats is channel0, first threshold 0.25
    assert v_lo[0].item() == 0.0
    assert v_hi[0].item() == 1.0


def test_validate_cifar10_config_ok():
    mcfg = {
        "input_dim": CIFAR10_PAPER_BINARY_INPUT_DIM,
        "neurons_per_layer": 12000,
        "num_classes": 10,
    }
    dcfg = {"dataset": "cifar10", "cifar10_embedding": "paper_small"}
    validate_model_data_config(mcfg, dcfg)


def test_validate_cifar10_large_config_ok():
    mcfg = {
        "input_dim": CIFAR10_PAPER_LARGE_INPUT_DIM,
        "neurons_per_layer": 256000,
        "num_classes": 10,
    }
    dcfg = {"dataset": "cifar10", "cifar10_embedding": "paper_large"}
    validate_model_data_config(mcfg, dcfg)


def test_large_embedding_dimension():
    x = torch.rand(3, 32, 32)
    v = _embed_cifar10_paper_large(x)
    assert v.shape == (CIFAR10_PAPER_LARGE_INPUT_DIM,)
    assert len(CIFAR10_PAPER_LARGE_THRESHOLDS) == 31


def test_cifar10_embedding_mode_parsing():
    assert cifar10_embedding_mode({"cifar10_embedding": "large"}) == "paper_large"
    assert cifar10_embedding_mode({}) == "paper_small"


def test_validate_cifar10_wrong_input_dim():
    mcfg = {"input_dim": 784, "neurons_per_layer": 12000, "num_classes": 10}
    dcfg = {"dataset": "cifar10", "cifar10_embedding": "paper_small"}
    with pytest.raises(ValueError, match="9216"):
        validate_model_data_config(mcfg, dcfg)


def test_validate_grouping_not_divisible():
    mcfg = {
        "input_dim": CIFAR10_PAPER_BINARY_INPUT_DIM,
        "neurons_per_layer": 12001,
        "num_classes": 10,
    }
    dcfg = {"dataset": "cifar10", "cifar10_embedding": "paper_small"}
    with pytest.raises(ValueError, match="divisible"):
        validate_model_data_config(mcfg, dcfg)


def test_three_thresholds_per_channel():
    assert len(CIFAR10_PAPER_THRESHOLD_LEVELS) == 3
