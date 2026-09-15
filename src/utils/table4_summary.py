"""Table 4–style MNIST summary row for experiment reports (paper small / paper normal / other)."""

from __future__ import annotations

import csv
import time
from pathlib import Path
from typing import Any

import torch

from src.models.hard_logic_net import HardLogicNet


def _is_paper_small_param_count_architecture(mcfg: dict) -> bool: # 判断是否是论文中的小参数模型
    """Paper-style 48000 / 48K row: MNIST 6x8000 or CIFAR-10 4x12000 (layersxneurons)."""
    nl = int(mcfg.get("num_layers", 0))
    npl = int(mcfg.get("neurons_per_layer", 0))
    return (nl == 6 and npl == 8000) or (nl == 4 and npl == 12000)


def _is_paper_cifar10_large_architecture(mcfg: dict) -> bool: # 判断是否是论文中的大参数模型
    """CIFAR-10 large: 5x256000 → paper-style 1280000 / 1.28M."""
    return int(mcfg.get("num_layers", 0)) == 5 and int(mcfg.get("neurons_per_layer", 0)) == 256000


def _is_paper_normal_architecture(mcfg: dict) -> bool: # 判断是否是论文中的正常参数模型
    return int(mcfg.get("num_layers", 0)) == 6 and int(mcfg.get("neurons_per_layer", 0)) == 64000


def table4_model_param_ops(mcfg: dict) -> tuple[str, str, str]: # 返回 CSV/MD Model 列、# Param. 字符串、OPs 字符串
    """Return (CSV/MD Model column, # Param. string, OPs string).

    Paper-style fixed triples for small / CIFAR large / normal; else gate-logit count.
    """
    if _is_paper_small_param_count_architecture(mcfg):
        return "Diff Logic Net (small)", "48000", "48K"
    if _is_paper_cifar10_large_architecture(mcfg):
        return "Diff Logic Net (large)", "1280000", "1.28M"
    if _is_paper_normal_architecture(mcfg):
        return "Diff Logic Net", "384000", "384K"
    nl = int(mcfg["num_layers"]) # 获取层数
    npl = int(mcfg["neurons_per_layer"]) # 获取每层神经元数
    nparams = nl * npl * 16 # 计算参数数量
    if nparams % 1000 == 0 and nparams >= 1000: # 如果参数数量是1000的倍数，则用K表示，否则用原始数字表示
        ops = f"{nparams // 1000}K"
    else:
        ops = str(nparams)
    return "Diff Logic Net", str(nparams), ops


def table4_param_and_ops_display(mcfg: dict) -> tuple[str, str]:
    """Backward-compatible (# Param., OPs) only; prefer `table4_model_param_ops` for the model name."""
    _, p, o = table4_model_param_ops(mcfg)
    return p, o


@torch.no_grad()
def benchmark_hard_inference_seconds_per_image( # 硬推理时间，返回每张图片的推理时间
    hard: HardLogicNet,
    loader,
    device: torch.device,
    warmup_batches: int = 5,
) -> tuple[float | None, str]:
    """Pure PyTorch hard forward on GPU; returns (seconds_per_image, notes).

    Uses the same DataLoader order as passed in (typically full MNIST test set)。
    """
    if device.type != "cuda":
        return None, "not_cuda_device"

    hard.eval()
    hard.to(device)

    it = iter(loader)
    for _ in range(max(1, warmup_batches)): # 预热
        try:
            xb, _ = next(it) 
        except StopIteration: 
            it = iter(loader) # 如果迭代器结束，则重新迭代
            xb, _ = next(it) # 获取下一个批次
        xb = xb.to(device, non_blocking=True)
        torch.cuda.synchronize() 
        _ = hard(xb) # 推理
        torch.cuda.synchronize() 

    total_sec = 0.0 # 总时间
    n_images = 0
    for xb, _ in loader:
        xb = xb.to(device, non_blocking=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        _ = hard(xb)
        torch.cuda.synchronize()
        total_sec += time.perf_counter() - t0 # 累加时间
        n_images += xb.size(0) # 获取批次大小

    if n_images == 0:
        return None, "empty_loader"
    return total_sec / n_images, f"full_loader_batches_n_images={n_images}" # 返回每张图片的推理时间


def write_table4_csv_md(
    run_dir: Path,
    final_test: dict[str, Any],
    train_wall_seconds: float,
    t_gpu_sec_per_image: float | None,
    t_gpu_note: str,
    model_name: str,
    param_str: str,
    ops_str: str,
    dataset_display: str = "MNIST",
) -> tuple[Path, Path]: # 写入 CSV/MD 文件
    acc = float(final_test["test_hard_acc"])
    soft = float(final_test["test_soft_acc"])
    gap = float(final_test["test_gap"])
    flops = "-"
    train_time_str = f"{train_wall_seconds:.1f} s ({train_wall_seconds / 3600:.4f} h)"

    if t_gpu_sec_per_image is not None:
        t_gpu_str = f"{t_gpu_sec_per_image * 1000:.4f} ms/img"
    else:
        t_gpu_str = "-"

    csv_path = run_dir / "mnist_table4_summary.csv"
    md_path = run_dir / "mnist_table4_summary.md"

    fieldnames = [
        "Model",
        "Acc.",
        "Soft Acc.",
        "Gap",
        "# Param.",
        "T. [GPU]",
        "OPs",
        "FLOPs",
        "Train Time",
    ]
    row = {
        "Model": model_name,
        "Acc.": f"{acc:.6f}",
        "Soft Acc.": f"{soft:.6f}",
        "Gap": f"{gap:.6f}",
        "# Param.": param_str,
        "T. [GPU]": t_gpu_str,
        "OPs": ops_str,
        "FLOPs": flops,
        "Train Time": train_time_str,
    }

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_MINIMAL)
        w.writeheader()
        w.writerow(row)

    meta_note = f"GPU timing note: {t_gpu_note}. Acc.=test_hard_acc, Soft Acc.=test_soft_acc, Gap=test_soft_acc-test_hard_acc."

    md_lines = [
        f"# {dataset_display} — Table 4 style summary (single row)",
        "",
        "| Model | Acc. | Soft Acc. | Gap | # Param. | T. [GPU] | OPs | FLOPs | Train Time |",
        "| --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        "| "
        + " | ".join(
            [
                model_name,
                f"{acc:.6f}",
                f"{soft:.6f}",
                f"{gap:.6f}",
                param_str,
                t_gpu_str.replace("|", "\\|"),
                ops_str,
                flops,
                train_time_str.replace("|", "\\|"),
            ]
        )
        + " |",
        "",
        "## Column definitions",
        "",
        "- **Acc.**: `test_hard_acc` (argmax hard model on the held-out test set).",
        "- **Soft Acc.**: `test_soft_acc` (differentiable soft gates on test).",
        "- **Gap**: `test_soft_acc - test_hard_acc`.",
        "- **# Param. / OPs**: paper-style fixed **48000 / 48K** (6x8000 MNIST or 4x12000 CIFAR-10), **1280000 / 1.28M** (5x256000 CIFAR-10 large), **384000 / 384K** (6x64000); otherwise gate-logit count `num_layers x neurons_per_layer x 16`.",
        "- **FLOPs**: not estimated here (`-`).",
        "- **Train Time**: sum of per-epoch wall times logged in `train_log.csv` (`epoch_time` column), in seconds and hours.",
        "",
        "## Inference timing (T. [GPU])",
        "",
        "- **Hard model** only, pure PyTorch forward on GPU.",
        "- Each batch is bracketed with `torch.cuda.synchronize()` before/after forward for valid GPU timing.",
        "- Warm-up: several batches before the timed pass.",
        f"- Timed pass: **full {dataset_display} test loader** (same split as `final_test.json`), batch size from training config.",
        "- Reported **T. [GPU]** is **mean wall time per image** over all timed batches.",
        "- **Not comparable** to paper **T. [GPU]** from optimized backends / custom CUDA / bit-packed inference.",
        "",
        "## Provenance",
        "",
        f"- Values from `final_test.json` in this run directory.",
        f"- {meta_note}",
        "",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))

    return csv_path, md_path
