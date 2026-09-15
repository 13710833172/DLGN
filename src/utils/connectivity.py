"""Fixed random connectivity: each neuron samples two inputs from the full previous layer. 固定随机连接：每个神经元从完整的上一层中采样两个输入。"""

import torch
from torch import Tensor


def build_connectivity(input_dim: int, output_dim: int, seed: int) -> Tensor:
    """Sample two input indices per output neuron, uniformly from [0, input_dim). 从[0, input_dim)中均匀采样两个输入索引，每个输出神经元采样两个输入索引

    Args:
        input_dim: size of the previous layer (or input features).
        output_dim: number of neurons in this layer. 
        seed: RNG seed for reproducibility.

    Returns:
        LongTensor of shape [output_dim, 2]. 形状为[输出神经元数, 2]，每个神经元采样两个输入索引
    """
    g = torch.Generator() # 生成器，用于生成随机数
    g.manual_seed(int(seed)) # 设置随机种子，便于复现
    idx = torch.randint(
        low=0,
        high=input_dim,
        size=(output_dim, 2), # 形状为[输出神经元数, 2]，每个神经元采样从输入维度中采样两个输入索引
        generator=g, # 生成器，用于生成随机数
        dtype=torch.long,
    ) # 生成随机数，形状为[输出神经元数, 2]，每个神经元采样两个输入索引
    return idx # 返回连接矩阵，形状为[输出神经元数, 2]，每个神经元采样两个输入索引  

    #####################################################################
    # 以上随机均匀采样有弊端：可能很多输入重复被选 有些输入完全没被选到 连接覆盖会很差
    #####################################################################
