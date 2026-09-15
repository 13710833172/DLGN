"""Gate statistics, activation monitoring, soft-vs-hard diagnostics."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch import Tensor

if TYPE_CHECKING:
    from src.models.logic_net import DiffLogicNet
    from src.models.hard_logic_net import HardLogicNet


def gate_entropy(probs: Tensor, eps: float = 1e-10) -> Tensor:
    """Per-neuron entropy H = -sum_i p_i log(p_i + eps), shape [num_neurons]. 定义每个神经元的门控概率的熵"""
    logp = torch.log(probs + eps) # 对门控概率取对数，eps防止 log(0) 无穷大。  H=−∑​p_i​log(p_i) i=1,...,16
    return -(probs * logp).sum(dim=-1) # 对每个神经元的门控概率取对数，然后求和，得到每个神经元的门控概率的熵 【每个神经元算一个熵，判断稳不稳定】


def layer_gate_stats(probs: Tensor) -> tuple[float, float]:
    """Mean entropy and mean max probability over neurons. 【用于评判每一层的所有神经元的门控分布是否集中/稳定，是否由一个门主导】"""
    h = gate_entropy(probs)
    maxp = probs.max(dim=-1).values # 对每层门控概率取最大值
    return float(h.mean().item()), float(maxp.mean().item()) # 返回每层门控概率的平均熵 和 平均最大门控概率


def activation_mean_std(activations: Tensor) -> tuple[float, float]:
    """Mean and std over batch and neuron dimensions. 每层激活的平均值和标准差 【用于检查每层输出是否坍塌】"""
    return float(activations.mean().item()), float(activations.std(unbiased=False).item()) # 返回每层激活的平均值和标准差


@torch.no_grad()
def collect_soft_layer_stats(
    model: "DiffLogicNet",
    x: Tensor,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    """Return per-layer (mean_entropy, mean_max_p) and (act_mean, act_std).【给一批输入 x, 跑一遍 soft model, 然后把每层的门控统计和激活统计都提取出来（就是上面说的两个东西）。】"""
    gate_stats: list[tuple[float, float]] = []
    act_stats: list[tuple[tuple[float, float], ...]] = []
    _, layer_acts, layer_probs = model.forward_with_activations(x)
    for p in layer_probs:
        gate_stats.append(layer_gate_stats(p))
    act_list: list[tuple[float, float]] = []
    for a in layer_acts:
        act_list.append(activation_mean_std(a))
    return gate_stats, act_list


@torch.no_grad()
def soft_hard_layerwise_agreement(
    soft_model: "DiffLogicNet",
    hard_model: "HardLogicNet",
    x_binary: Tensor,
    tol: float = 0.05, # 设置一个容差，看每层有多少比例的神经元满足∣soft−hard∣≤tol，比例越高，说明离散化前后越一致。
) -> list[float]:
    """Fraction of neurons per layer where |soft - hard| <= tol (hard as float). 【比较 soft 和 hard 模型每层输出有多一致】"""
    # forward_with_activations 返回 (final_out, per_layer_acts, per_layer_probs)
    _, soft_acts, _ = soft_model.forward_with_activations(x_binary)
    hard_acts = hard_model.forward_layerwise(x_binary)
    rates: list[float] = []
    for s, h in zip(soft_acts, hard_acts, strict=True):
        hf = h.float()
        agree = (s - hf).abs() <= tol
        rates.append(float(agree.float().mean().item()))
    return rates


def discretization_gap(soft_acc: float, hard_acc: float) -> float:
    return soft_acc - hard_acc
