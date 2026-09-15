"""Aggregate final-layer neuron outputs into class logits."""

from torch import Tensor


def compute_class_scores(
    outputs: Tensor,
    num_classes: int,
    tau: float,
    beta: float,
) -> Tensor:
    """Group outputs evenly across classes and apply tau/beta.

    Args:
        outputs: [batch, total_neurons], total_neurons divisible by num_classes.
        num_classes: number of classes (e.g. 10).
        tau: scale for summed group activations.
        beta: additive bias.

    Returns:
        [batch, num_classes] class scores for CrossEntropyLoss.
    """
    if outputs.shape[-1] % num_classes != 0:
        raise ValueError(
            f"outputs dim {outputs.shape[-1]} not divisible by num_classes {num_classes}"
        )
    group_size = outputs.shape[-1] // num_classes # 每个类别的神经元数
    grouped = outputs.view(outputs.shape[0], num_classes, group_size) # 将输出 reshape 成 [batch, num_classes, group_size]
    sums = grouped.sum(dim=-1) # 对每个类别的神经元求和
    return sums / tau + beta # 𝜏 主要用于归一化，𝛽 基本无关紧要
