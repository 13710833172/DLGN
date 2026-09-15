"""Evaluate soft and hard logic networks on the held-out test set (MNIST / CIFAR-10).
    读取 checkpoint 和 config
    按训练时的配置重建测试数据
    按训练时的配置重建 soft 模型，并加载权重
    从 soft 模型离散化得到 hard 模型
    对 soft / hard 模型分别在 test set 上跑前向
    把最后一层输出按类别分组，得到分类 logits
    比较 soft/hard accuracy, 并报告 discretization gap
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import get_data_loaders, validate_model_data_config
from src.models.hard_logic_net import HardLogicNet
from src.models.logic_net import DiffLogicNet
from src.utils.metrics import discretization_gap
from src.utils.output_grouping import compute_class_scores


@torch.no_grad()
def acc_model(forward_fn, loader, device, num_classes, tau, beta):
    """计算模型在测试集上的准确率"""
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = forward_fn(x) # [batch, neurons_per_layer]
        logits = compute_class_scores(out, num_classes, tau, beta) # [batch, 10]
        pred = logits.argmax(dim=-1) # [batch]
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1) # max(total,1) 只是防止极端情况下分母为 0。


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--device", type=str, default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    ckpt = torch.load(Path(args.checkpoint), map_location="cpu", weights_only=False) # 加载checkpoint文件
    cfg = ckpt["config"] # 从checkpoint中取配置
    mcfg, dcfg, ocfg = cfg["model"], cfg["data"], cfg["output"]
    ccfg = cfg.get("checkpoint", {})
    conn_seed = int(ckpt.get("connectivity_seed", ccfg.get("seed", 0)))

    validate_model_data_config(mcfg, dcfg)
    _, _, test_loader = get_data_loaders(
        dcfg,
        batch_size=int(cfg["training"]["batch_size"]),
    )

    # 重新实例化一个和训练时结构完全一致的 DiffLogicNet，用于评估软逻辑网络
    soft = DiffLogicNet(
        input_dim=int(mcfg["input_dim"]),
        num_layers=int(mcfg["num_layers"]),
        neurons_per_layer=int(mcfg["neurons_per_layer"]),
        connectivity_seed=conn_seed,
        gate_init=str(mcfg.get("gate_init", "normal")),
    )
    soft.load_state_dict(ckpt["model_state_dict"], strict=True) # 加载模型权重
    soft.to(device)
    soft.eval()

    hard = HardLogicNet.from_soft_model(soft).to(device) # 从 soft 模型构造 hard 模型
    hard.eval()

    nc = int(mcfg["num_classes"])
    tau = float(ocfg["tau"])
    beta = float(ocfg["beta"])

    # accuracy = 预测正确的样本数 ​/ 总样本数
    s_acc = acc_model(soft, test_loader, device, nc, tau, beta)
    h_acc = acc_model(hard, test_loader, device, nc, tau, beta)
    gap = discretization_gap(s_acc, h_acc)
    print(f"test_soft_acc: {s_acc:.6f}")
    print(f"test_hard_acc: {h_acc:.6f}")
    print(f"discretization_gap (test_soft_acc - test_hard_acc): {gap:.6f}")


if __name__ == "__main__":
    main()
