"""
16 real-valued (differentiable) and hard (Boolean) logic operators.

Operator ordering is fixed and must be consistent everywhere:
  0: FALSE,  1: AND,  2: A AND NOT B,  3: A,
  4: NOT A AND B,  5: B,  6: XOR,  7: OR,
  8: NOR,  9: XNOR,  10: NOT B,  11: B=>A,
  12: NOT A,  13: A=>B,  14: NAND,  15: TRUE
"""

import torch
from torch import Tensor

NUM_OPERATORS = 16

OPERATOR_NAMES = [
    "FALSE", "AND", "A AND NOT B", "A",
    "NOT A AND B", "B", "XOR", "OR",
    "NOR", "XNOR", "NOT B", "B=>A",
    "NOT A", "A=>B", "NAND", "TRUE",
]

# -------------------------------以下定义的函数在logic_layer.py中调用------------------------------------
def compute_all_16_ops(a: Tensor, b: Tensor) -> Tensor:
    """Compute all 16 real-valued logic operators in parallel.

    Args:
        a: tensor of shape [...], values in [0, 1]
        b: tensor of shape [...], values in [0, 1]

    Returns:
        Tensor of shape [..., 16] with operator outputs stacked on last dim.
    """
    ab = a * b
    a_plus_b = a + b

    results = [
        torch.zeros_like(a),       # 0:  FALSE
        ab,                         # 1:  AND
        a - ab,                     # 2:  A AND NOT B  (= NOT(A=>B))
        a,                          # 3:  A
        b - ab,                     # 4:  NOT A AND B  (= NOT(B=>A))
        b,                          # 5:  B
        a_plus_b - 2.0 * ab,        # 6:  XOR
        a_plus_b - ab,              # 7:  OR
        1.0 - (a_plus_b - ab),      # 8:  NOR
        1.0 - (a_plus_b - 2.0*ab),  # 9:  XNOR
        1.0 - b,                    # 10: NOT B
        1.0 - b + ab,               # 11: B=>A
        1.0 - a,                    # 12: NOT A
        1.0 - a + ab,               # 13: A=>B
        1.0 - ab,                   # 14: NAND
        torch.ones_like(a),         # 15: TRUE
    ]
    return torch.stack(results, dim=-1)


# -------------------------------以下定义的两个函数在hard_logic_layer.py中调用------------------------------------
def build_truth_table(device: torch.device | None = None) -> Tensor: # 构建 16 种逻辑运算的布尔真值表
    """Build the [16, 2, 2] Boolean truth table for all 16 operators. 含义是truth_table[op, a, b]
        一共有 16 个二输入布尔算子, a 只可能是 0 或 1, b 只可能是 0 或 1
        软模型训练(DiffLogicLayer)不使用这个函数；它只服务于硬逻辑路径(hard eval / hard inference)。
        只是用来查表,根据ab数据查出相对应的输出。

    Returns:
        LongTensor of shape [16, 2, 2] where truth_table[op, a, b] is the output.
    """
    tt = torch.zeros(16, 2, 2, dtype=torch.long, device=device) # 初始化一个全 0 的 LongTensor，形状为 [16, 2, 2]. 即有 16 张 2×2 小表
    for a_val in range(2): # 0 和 1
        for b_val in range(2): # 0 和 1
            a_t = torch.tensor(float(a_val)) # 将 a_val 转换为浮点数
            b_t = torch.tensor(float(b_val)) # 将 b_val 转换为浮点数
            ops = compute_all_16_ops(a_t, b_t) # 计算 16 种逻辑运算的结果
            tt[:, a_val, b_val] = ops.round().long() # 将结果四舍五入为整数，防御式写法，实际上输出全是0/1，不用四舍五入
    return tt


def hard_logic_lookup(
    a: Tensor, b: Tensor, gate_ids: Tensor, truth_table: Tensor
) -> Tensor:
    """Vectorized batched truth-table lookup for hard logic.
    这个函数根据每个神经元的门编号(gate_ids)、输入 a/b(0 或 1)，在预先计算好的 truth_table 中查表，
    Args:
        a: LongTensor [..., num_neurons], values in {0, 1}
        b: LongTensor [..., num_neurons], values in {0, 1}
        gate_ids: LongTensor [num_neurons], values in {0..15}
        truth_table: LongTensor [16, 2, 2] 对应选哪个门, a和b分别可能是0或1

    Returns:
        LongTensor [..., num_neurons] with Boolean outputs.
    """
    gids = gate_ids.expand_as(a) # 将 gate_ids 扩展为与 a 和 b 相同的形状, 即把每个神经元的 gate id，复制到 batch 维度上，有多少个batch，就有多少个相同的gate_ids, 对应多少个样本
    return truth_table[gids, a, b] # 根据 gate_ids 索引 truth_table 中的结果
