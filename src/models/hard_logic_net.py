"""Hard logic network with same connectivity as the soft model."""

import torch
import torch.nn as nn
from torch import Tensor

from src.models.hard_logic_layer import HardLogicLayer


class HardLogicNet(nn.Module):
    def __init__(self, input_dim, connectivities, gate_ids_per_layer, neurons_per_layer):
        super().__init__()
        if len(connectivities) != len(gate_ids_per_layer):
            raise ValueError("connectivities and gate_ids length mismatch")
        self.input_dim = input_dim
        self.neurons_per_layer = neurons_per_layer
        layers = []
        prev = input_dim
        for conn, gids in zip(connectivities, gate_ids_per_layer):
            layers.append(
                HardLogicLayer(
                    input_dim=prev,
                    num_neurons=neurons_per_layer,
                    connectivity=conn,
                    gate_ids=gids,
                )
            ) # 创建一个 HardLogicLayer 对象，并添加到 layers 列表中
            prev = neurons_per_layer
        self.layers = nn.ModuleList(layers)

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    @torch.no_grad() # 用于记录所有层的输出
    def forward_layerwise(self, x: Tensor) -> list:
        acts = []
        for layer in self.layers:
            x = layer(x)
            acts.append(x)
        return acts

    @classmethod
    def from_soft_model(cls, soft, gate_ids_per_layer=None):
        if gate_ids_per_layer is None:
            gate_ids_per_layer = soft.discretized_gate_ids()
        conns = soft.connectivity_tensors() # 从软模型拿连接结构 (尺寸为[输出神经元数（神经元数）, 2]，意思是每个神经元采样两个输入索引)
        return cls(
            input_dim=soft.input_dim,
            connectivities=conns,
            gate_ids_per_layer=[g.detach().clone() for g in gate_ids_per_layer],
            neurons_per_layer=soft.neurons_per_layer,
        )
    # 从一个已训练（或当前状态）的 DiffLogicNet（软模型）里提取
    # 每层连接关系 connectivity
    #  每层离散门编号 gate_id（默认由 argmax(gate_logits) 得到）
    # 用这些信息构造出对应的 HardLogicNet（硬模型）
    # 也就是把“可微的软门混合网络”转成“每个神经元只选一个门的离散网络”。
