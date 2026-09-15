"""Differentiable logic layer: soft mixture over 16 operators. 设置每个神经元/节点如何计算"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from src.ops.logic_ops import compute_all_16_ops
from src.utils.connectivity import build_connectivity


class DiffLogicLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        connectivity: Tensor | None = None,
        connectivity_seed: int | None = None,
        gate_init: str = "normal",
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.gate_init = gate_init

        if connectivity is not None:
            conn = connectivity
        else:
            if connectivity_seed is None:
                raise ValueError("Either connectivity or connectivity_seed required")
            conn = build_connectivity(input_dim, num_neurons, connectivity_seed)
        self.register_buffer("connectivity", conn.long(), persistent=True)

        if gate_init == "zeros":
            logits = torch.zeros(num_neurons, 16)
        elif gate_init == "normal":
            logits = torch.randn(num_neurons, 16)  # 初始化门控 logits 为正态分布
        else:
            raise ValueError(f"Unknown gate_init: {gate_init}")
        self.gate_logits = nn.Parameter(logits) # 每一层的门控 logits，尺寸为[num_neurons, 16]，代表每个神经元对应16个可学习参数

    def gate_probs(self) -> Tensor:
        return F.softmax(self.gate_logits, dim=-1) # 用 softmax 将门控 logits（即16个可学习参数）转换为概率

    def forward(self, x: Tensor) -> Tensor:
        """x: [batch, input_dim] -> [batch, num_neurons]"""
        conn = self.connectivity # 随机从上一层输入中选取的索引，[num_neurons, 2]
        a = x[:, conn[:, 0]] # 获取第一个输入索引对应的输入
        b = x[:, conn[:, 1]] # 获取第二个输入索引对应的输入
        ops = compute_all_16_ops(a, b)
        p = self.gate_probs()
        return (ops * p.unsqueeze(0)).sum(dim=-1) # 将每个神经元的 16 种算子概率与对应的逻辑门概率相乘，然后求和得到最终输出
