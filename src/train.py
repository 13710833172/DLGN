"""Train differentiable logic gate network (MNIST / CIFAR-10) with structured experiment logs."""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import get_data_loaders, validate_model_data_config
from src.models.hard_logic_net import HardLogicNet
from src.models.logic_net import DiffLogicNet
from src.utils.metrics import (
    collect_soft_layer_stats,
    discretization_gap,
    soft_hard_layerwise_agreement,
)
from src.utils.output_grouping import compute_class_scores
from src.utils.table4_summary import (
    benchmark_hard_inference_seconds_per_image,
    table4_model_param_ops,
    write_table4_csv_md,
)


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


@torch.no_grad()
def evaluate_soft(
    model: DiffLogicNet,
    loader,
    device,
    num_classes: int,
    tau: float,
    beta: float,
) -> float:
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = model(x)
        logits = compute_class_scores(out, num_classes, tau, beta)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_hard(
    hard: HardLogicNet,
    loader,
    device,
    num_classes: int,
    tau: float,
    beta: float,
) -> float:
    hard.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        out = hard(x)
        logits = compute_class_scores(out, num_classes, tau, beta)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1) # 返回准确率


def _is_cuda_oom(exc: BaseException) -> bool:
    if isinstance(exc, torch.cuda.OutOfMemoryError):
        return True
    msg = str(exc).lower()
    return "out of memory" in msg


def _print_training_oom_help(cfg_path: Path, tcfg: dict, mcfg: dict) -> None: # 打印内存不足的错误信息
    print(
        "\n======== CUDA / memory: out of memory during training ========\n"
        f"Config: {cfg_path}\n"
        f"Architecture: num_layers={mcfg.get('num_layers')}, neurons_per_layer={mcfg.get('neurons_per_layer')}, "
        f"batch_size={tcfg.get('batch_size')}.\n\n"
        "This project does not silently shrink the model or batch size.\n"
        "Options: use a GPU with more memory, reduce `training.batch_size` or layer width in a **new** YAML copy, "
        "or run a one-step feasibility check:\n"
        "  python scripts/check_mnist_normal_feasibility.py\n"
        "  python scripts/check_cifar10_small_feasibility.py\n"
        "  python scripts/check_cifar10_large_feasibility.py\n"
        "================================================================\n",
        file=sys.stderr,
    )


def train_one_epoch(
    model: DiffLogicNet,
    loader,
    optimizer,
    criterion,
    device,
    num_classes: int,
    tau: float,
    beta: float,
) -> tuple[float, float]:
    """Return (train_loss, train_acc)."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = compute_class_scores(out, num_classes, tau, beta)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        bs = x.size(0)
        total_loss += loss.item() * bs
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()

    return total_loss / max(total, 1), correct / max(total, 1) # 返回损失和准确率, max(total, 1) 的作用是避免当 total 为 0 时出现除以零的错误


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=str, required=True)
    args = ap.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    ocfg = cfg["output"]
    lcfg = cfg.get("logging", {})
    ccfg = cfg.get("checkpoint", {})

    device_str = tcfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        device = torch.device("cpu")
        print("CUDA not available, using CPU")
    else:
        device = torch.device(device_str)

    conn_seed = int(ccfg.get("seed", 0))
    validate_model_data_config(mcfg, dcfg)
    train_loader, val_loader, test_loader = get_data_loaders(
        dcfg,
        batch_size=int(tcfg["batch_size"]),
    )

    # 设置软网络模型
    model = DiffLogicNet(
        input_dim=int(mcfg["input_dim"]),
        num_layers=int(mcfg["num_layers"]),
        neurons_per_layer=int(mcfg["neurons_per_layer"]),
        connectivity_seed=conn_seed,
        gate_init=str(mcfg.get("gate_init", "normal")),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
    criterion = nn.CrossEntropyLoss()
    max_epochs = int(tcfg["max_epochs"])
    num_classes = int(mcfg["num_classes"])
    tau = float(ocfg["tau"])
    beta = float(ocfg["beta"])

    evaluate_test_during_training = bool(lcfg.get("evaluate_test_during_training", False)) # 是否在每个 epoch 计算测试集的准确率
    log_soft_hard = bool(lcfg.get("log_soft_hard_consistency", False)) # 是否记录软硬模型的一致性
    cons_interval = int(lcfg.get("consistency_interval", 1)) # 每多少个 epoch 计算一次软硬模型的一致性

    save_dir = Path(ccfg.get("save_dir", "checkpoints"))
    save_dir.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path("logs") / f"run_{stamp}"
    run_dir.mkdir(parents=True, exist_ok=True)

    train_log_path = run_dir / "train_log.csv"
    layer_stats_path = run_dir / "layer_stats.csv"
    final_test_path = run_dir / "final_test.json"
    summary_path = run_dir / "run_summary.json"

    best_val_soft = -1.0
    best_epoch = -1
    best_val_hard = -1.0
    best_val_gap = 0.0
    best_path = save_dir / "best_model.pt"

    with open(train_log_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "epoch",
                "train_loss",
                "train_acc",
                "val_soft_acc",
                "val_hard_acc",
                "val_gap",
                "lr",
                "epoch_time",
            ]
        )

    with open(layer_stats_path, "w", newline="", encoding="utf-8") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(
            [
                "epoch",
                "layer_idx",
                "mean_gate_entropy",
                "mean_max_gate_probability",
                "activation_mean",
                "activation_std",
            ]
        )

    total_train_wall_seconds = 0.0

    cuda_mem_f = None
    cuda_mem_writer = None
    if device.type == "cuda":
        cuda_mem_path = run_dir / "cuda_memory.csv" # 记录每个 epoch 的内存使用情况
        cuda_mem_f = open(cuda_mem_path, "w", newline="", encoding="utf-8") # 打开文件，用于写入内存使用情况
        cuda_mem_writer = csv.writer(cuda_mem_f) # 创建一个 csv.writer 对象，用于写入内存使用情况
        cuda_mem_writer.writerow(["epoch", "peak_allocated_gib", "peak_reserved_gib"]) # 写入表头 
        torch.cuda.reset_peak_memory_stats()

    try:
    # --------------------------------------------正式开始训练---------------------------------------
        for epoch in range(1, max_epochs + 1):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats() #

            t0 = time.time()

            train_loss, train_acc = train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
                device,
                num_classes,
                tau,
                beta,
            )

            val_soft_acc = evaluate_soft(model, val_loader, device, num_classes, tau, beta) # 评估软模型在验证集上的准确率
            hard = HardLogicNet.from_soft_model(model).to(device) # 从软模型创建硬模型
            val_hard_acc = evaluate_hard(hard, val_loader, device, num_classes, tau, beta) # 评估硬模型在验证集上的准确率
            val_gap = discretization_gap(val_soft_acc, val_hard_acc)

            model.eval()
            xb, _ = next(iter(val_loader)) # 获取一个 batch 的验证数据
            xb = xb.to(device, non_blocking=True)
            gate_stats, act_stats = collect_soft_layer_stats(model, xb) # 收集软模型每层的门控熵和激活统计信息

            lr = float(optimizer.param_groups[0]["lr"]) # 获取当前学习率
            epoch_time = time.time() - t0
            total_train_wall_seconds += epoch_time # 记录总训练时间

            with open(train_log_path, "a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                writer.writerow(
                    [
                        epoch,
                        f"{train_loss:.6f}",
                        f"{train_acc:.6f}",
                        f"{val_soft_acc:.6f}",
                        f"{val_hard_acc:.6f}",
                        f"{val_gap:.6f}",
                        f"{lr:.8f}",
                        f"{epoch_time:.4f}",
                    ]
                )

            with open(layer_stats_path, "a", newline="", encoding="utf-8") as fcsv:
                writer = csv.writer(fcsv)
                for li, ((h, mp), (am, ast)) in enumerate(zip(gate_stats, act_stats)):
                    writer.writerow(
                        [
                            epoch,
                            li,
                            f"{h:.6f}",
                            f"{mp:.6f}",
                            f"{am:.6f}",
                            f"{ast:.6f}",
                        ]
                    )

            msg_parts = [
                f"epoch {epoch}/{max_epochs}",
                f"train_loss={train_loss:.4f}",
                f"train_acc={train_acc:.4f}",
                f"val_soft={val_soft_acc:.4f}",
                f"val_hard={val_hard_acc:.4f}",
                f"val_gap={val_gap:.4f}",
                f"lr={lr:.6f}",
                f"epoch_time={epoch_time:.2f}s",
            ] # 打印训练信息

            # 跟踪 GPU 显存使用情况
            if device.type == "cuda":
                torch.cuda.synchronize() # 同步 CUDA 设备，确保所有 CUDA 操作都完成
                peak_a = torch.cuda.max_memory_allocated() / (1024**3)
                peak_r = torch.cuda.max_memory_reserved() / (1024**3)
                msg_parts.append(f"cuda_peak_alloc_gib={peak_a:.3f}")
                msg_parts.append(f"cuda_peak_reserved_gib={peak_r:.3f}")
                if cuda_mem_writer is not None:
                    cuda_mem_writer.writerow([epoch, f"{peak_a:.6f}", f"{peak_r:.6f}"])
                    cuda_mem_f.flush()

            if evaluate_test_during_training:
                test_soft_acc = evaluate_soft(model, test_loader, device, num_classes, tau, beta)
                test_hard_acc = evaluate_hard(hard, test_loader, device, num_classes, tau, beta)
                test_gap = discretization_gap(test_soft_acc, test_hard_acc)
                msg_parts.extend(
                    [
                        f"test_soft={test_soft_acc:.4f}",
                        f"test_hard={test_hard_acc:.4f}",
                        f"test_gap={test_gap:.4f}",
                    ]
                )

            print(" | ".join(msg_parts))

            # 是否记录软硬模型的一致性
            if log_soft_hard and (epoch % cons_interval == 0): 
                x_small = xb[: min(64, xb.size(0))]
                rates = soft_hard_layerwise_agreement(model, hard, x_small) # 计算软硬模型每层的一致性
                print("  soft_hard_layer_agree:", [round(r, 4) for r in rates])

            if val_soft_acc > best_val_soft:
                best_val_soft = val_soft_acc
                best_val_hard = val_hard_acc
                best_val_gap = val_gap
                best_epoch = epoch
                ckpt = {
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "epoch": epoch,
                    "best_val_soft_acc": best_val_soft,
                    "best_val_acc": best_val_soft,
                    "best_val_hard_acc": best_val_hard,
                    "best_val_gap": best_val_gap,
                    "config": cfg,
                    "config_path": str(cfg_path),
                    "connectivity_seed": conn_seed,
                }
                torch.save(ckpt, best_path)
                print(f"  saved best checkpoint val_soft_acc={best_val_soft:.4f} -> {best_path}")
    # ------------------------------------------------------------------------------------------------
    except torch.cuda.OutOfMemoryError as e:
        _print_training_oom_help(cfg_path, tcfg, mcfg)
        raise SystemExit(1) from e
    except RuntimeError as e:
        if _is_cuda_oom(e):
            _print_training_oom_help(cfg_path, tcfg, mcfg)
            raise SystemExit(1) from e
        raise
    finally:
        if cuda_mem_f is not None:
            cuda_mem_f.close()

    # Final one-shot test evaluation from best validation checkpoint.
    best_ckpt = torch.load(best_path, map_location="cpu", weights_only=False)
    best_model = DiffLogicNet(
        input_dim=int(mcfg["input_dim"]),
        num_layers=int(mcfg["num_layers"]),
        neurons_per_layer=int(mcfg["neurons_per_layer"]),
        connectivity_seed=conn_seed,
        gate_init=str(mcfg.get("gate_init", "normal")),
    ).to(device)
    best_model.load_state_dict(best_ckpt["model_state_dict"], strict=True)
    best_hard = HardLogicNet.from_soft_model(best_model).to(device)

    test_soft_acc = evaluate_soft(best_model, test_loader, device, num_classes, tau, beta)
    test_hard_acc = evaluate_hard(best_hard, test_loader, device, num_classes, tau, beta)
    test_gap = discretization_gap(test_soft_acc, test_hard_acc)

    final_test = {
        "best_epoch": int(best_ckpt.get("epoch", best_epoch)),
        "best_val_soft_acc": float(
            best_ckpt.get("best_val_soft_acc", best_ckpt.get("best_val_acc", best_val_soft))
        ),
        "best_val_hard_acc": float(best_ckpt.get("best_val_hard_acc", best_val_hard)),
        "best_val_gap": float(best_ckpt.get("best_val_gap", best_val_gap)),
        "test_soft_acc": float(test_soft_acc),
        "test_hard_acc": float(test_hard_acc),
        "test_gap": float(test_gap),
    }

    with open(final_test_path, "w", encoding="utf-8") as f:
        json.dump(final_test, f, indent=2)

    # Table 4–style summary: GPU hard inference timing + test metrics

    table4_model_name, param_str, ops_str = table4_model_param_ops(mcfg) # 获取模型名称、参数和操作字符串
    t_gpu_sec, t_gpu_note = benchmark_hard_inference_seconds_per_image(
        best_hard, test_loader, device, warmup_batches=5
    ) # 计算硬模型在测试集上的推理时间

    ds_key = str(dcfg.get("dataset", "mnist")).lower().replace("-", "").replace("_", "") 
    dataset_display = {"mnist": "MNIST", "cifar10": "CIFAR-10"}.get(
        ds_key, str(dcfg.get("dataset", "MNIST"))
    )

    csv_t4, md_t4 = write_table4_csv_md(
        run_dir=run_dir,
        final_test=final_test,
        train_wall_seconds=total_train_wall_seconds,
        t_gpu_sec_per_image=t_gpu_sec,
        t_gpu_note=t_gpu_note,
        model_name=table4_model_name,
        param_str=param_str,
        ops_str=ops_str,
        dataset_display=dataset_display,
    ) # 写入表格4的总结

    summary = {
        "run_dir": str(run_dir),
        "train_log_csv": str(train_log_path),
        "layer_stats_csv": str(layer_stats_path),
        "final_test_json": str(final_test_path),
        "best_checkpoint": str(best_path),
        "mnist_table4_summary_csv": str(csv_t4),
        "mnist_table4_summary_md": str(md_t4),
        "table4_gpu_timing_note": t_gpu_note,
        "table4_train_wall_seconds": total_train_wall_seconds,
    }
    if device.type == "cuda":
        summary["cuda_memory_csv"] = str(run_dir / "cuda_memory.csv")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print("Final test from best checkpoint:", final_test)
    print("Table 4 summary:", csv_t4, md_t4)
    print("Done.", summary)


if __name__ == "__main__":
    main()
