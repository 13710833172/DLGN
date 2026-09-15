"""
这个类 HardLogicLayer 实现了一个只执行硬布尔逻辑门的神经网络层。每个神经元：
固定连到输入的两个位置(由 connectivity 指定)。
固定使用单一布尔逻辑门(由 gate_ids 指定,0~15 共16种二元逻辑)。
前向时，从输入张量 x 按 connectivity 选择出 [a, b]，查 truth_table 用 gate_ids 得到输出(0/1)。
每个神经元得出一个0或1输出,代表某种布尔逻辑算子的计算结果。
此层的连接模式和门类型都不可训练，只负责根据输入和门表直接查表输出。实现高度“硬”的可解释逻辑计算。
"""

import torch.nn as nn
from torch import Tensor

from src.ops.logic_ops import build_truth_table, hard_logic_lookup


class HardLogicLayer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_neurons: int,
        connectivity: Tensor, # 每个神经元连哪两个输入， 尺寸为[num_neurons, 2]
        gate_ids: Tensor, # 每个神经元对应的门控 ID， gate_ids[j] = k 表示第 j 个神经元执行第 k 个逻辑门
    ):
        super().__init__()
        self.input_dim = input_dim
        self.num_neurons = num_neurons
        self.register_buffer("connectivity", connectivity.long(), persistent=True) # 注册连接矩阵，用于索引输入 x 的第 conn[:, 0] 列和第 conn[:, 1] 列。connectivity 不是可学习参数，不应该被 optimizer 更新；
        self.register_buffer("gate_ids", gate_ids.long(), persistent=True) # 注册门控 ID，用于索引 truth_table
        tt = build_truth_table(device=connectivity.device) # 构建 16 种逻辑运算的布尔真值表
        self.register_buffer("truth_table", tt, persistent=True) # 注册真值表，用于索引 truth_table 对应选哪个门, a和b分别可能是0或1
        # buffer 会跟着模型一起 .to(device) 移动到 GPU/CPU，会被 state_dict() 保存，但不会参与梯度更新

    def forward(self, x: Tensor) -> Tensor:
        """x: [batch, input_dim] binary -> [batch, num_neurons] float {0,1}."""
        conn = self.connectivity
        a = x[:, conn[:, 0]]
        b = x[:, conn[:, 1]]
        out = hard_logic_lookup(a, b, self.gate_ids, self.truth_table) # 根据 gate_ids 和 truth_table 查找每个神经元的输出  
        return out.float() # out的形状是[batch, num_neurons]，每个神经元选一种逻辑输出一个 0/1，即每个神经元执行一个硬逻辑门
