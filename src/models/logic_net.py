"""Stacked differentiable logic layers. 可微逻辑门网络"""

from __future__ import annotations

import torch.nn as nn
from torch import Tensor

from src.models.logic_layer import DiffLogicLayer


class DiffLogicNet(nn.Module):
    """Fixed connectivity, learnable gate logits per neuron. 设置整个网络的结构和参数"""

    def __init__(
        self,
        input_dim: int,
        num_layers: int,
        neurons_per_layer: int,
        connectivity_seed: int,
        gate_init: str = "normal",
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be >= 1")
        self.input_dim = input_dim
        self.num_layers = num_layers
        self.neurons_per_layer = neurons_per_layer
        self.connectivity_seed = int(connectivity_seed)

        layers: list[DiffLogicLayer] = []
        prev = input_dim
        for li in range(num_layers): # 创建多个逻辑层
            seed = self.connectivity_seed + li * 1_000_003 # 生成每个逻辑层的连接种子，便于复现
            layers.append(
                DiffLogicLayer(
                    input_dim=prev,
                    num_neurons=neurons_per_layer,
                    connectivity_seed=seed,
                    gate_init=gate_init,
                ) # 创建一个逻辑层
            )
            prev = neurons_per_layer # 更新前一层的输出维度，作为下一层的输入维度
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x # [batch_size, neurons_per_layer]


    # --------------------------------用于分析模型内部状态---------------------------------
    def forward_with_activations(
        self, x: Tensor
    ) -> tuple[Tensor, list[Tensor], list[Tensor]]:
        """Returns final output, per-layer activations, per-layer gate probs. 返回最终输出、每层激活和每层门控概率"""
        acts: list[Tensor] = []
        probs: list[Tensor] = []
        for layer in self.layers:
            x = layer(x)
            acts.append(x)
            probs.append(layer.gate_probs()) # 记录每层门控概率，在logic_layer.py中定义如何计算
        return x, acts, probs

    def connectivity_tensors(self) -> list[Tensor]:
        return [layer.connectivity for layer in self.layers] # 返回每层的连接矩阵 尺寸为[输出神经元数（神经元数）, 2]，意思是每个神经元采样两个输入索引

    def discretized_gate_ids(self) -> list[Tensor]:
        return [layer.gate_logits.argmax(dim=-1) for layer in self.layers] # .argmax()是对每个神经元那一行 16 个值，返回每层的离散门控 ID
    # ------------------------------------------------------------------------------------


    @classmethod
    def from_checkpoint_state(
        cls,
        state_dict: dict,
        connectivity_seed: int,
        num_layers: int,
        neurons_per_layer: int,
        input_dim: int,
        gate_init: str = "normal",
    ) -> "DiffLogicNet":
        """Rebuild model and load weights; expects keys like layers.0.*. 从检查点恢复模型和权重"""
        m = cls(
            input_dim=input_dim,
            num_layers=num_layers,
            neurons_per_layer=neurons_per_layer,
            connectivity_seed=connectivity_seed,
            gate_init=gate_init,
        )
        m.load_state_dict(state_dict, strict=True)
        return m
