#!/usr/bin/env python3
"""Smoke test for configs/cifar10_small_paper.yaml: data batch + one forward/backward/Adam step.

Exits 0 on success and prints peak CUDA memory; exits 1 on failure. Does not modify the YAML.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.loaders import get_data_loaders, validate_model_data_config
from src.models.logic_net import DiffLogicNet
from src.utils.output_grouping import compute_class_scores


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test CIFAR-10 small paper config (4×12000).")
    ap.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "cifar10_small_paper.yaml"),
        help="YAML config (default: cifar10_small_paper.yaml)",
    )
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    mcfg = cfg["model"]
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    ocfg = cfg["output"]
    ccfg = cfg.get("checkpoint", {})

    device_str = tcfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; this check is intended for GPU.", file=sys.stderr)
        sys.exit(2)

    device = torch.device(device_str)
    validate_model_data_config(mcfg, dcfg)
    train_loader, _, _ = get_data_loaders(dcfg, batch_size=int(tcfg["batch_size"]))
    conn_seed = int(ccfg.get("seed", 0))

    print(f"Config: {cfg_path}")
    print(
        f"Architecture: layers={mcfg['num_layers']}, neurons_per_layer={mcfg['neurons_per_layer']}, "
        f"input_dim={mcfg['input_dim']}, batch_size={tcfg['batch_size']}, device={device}"
    )

    try:
        model = DiffLogicNet(
            input_dim=int(mcfg["input_dim"]),
            num_layers=int(mcfg["num_layers"]),
            neurons_per_layer=int(mcfg["neurons_per_layer"]),
            connectivity_seed=conn_seed,
            gate_init=str(mcfg.get("gate_init", "normal")),
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=float(tcfg["lr"]))
        criterion = nn.CrossEntropyLoss()

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

        x, y = next(iter(train_loader))
        x = x.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = compute_class_scores(out, int(mcfg["num_classes"]), float(ocfg["tau"]), float(ocfg["beta"]))
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_a = torch.cuda.max_memory_allocated() / (1024**3)
            peak_r = torch.cuda.max_memory_reserved() / (1024**3)
            print(f"OK: one batch train step. Peak allocated {peak_a:.3f} GiB, peak reserved {peak_r:.3f} GiB.")
            props = torch.cuda.get_device_properties(device)
            print(f"GPU: {props.name}, total memory {props.total_memory / (1024**3):.2f} GiB.")
        else:
            print("OK: one batch train step (CPU).")

    except torch.cuda.OutOfMemoryError as e:
        print(
            "\n======== OOM during CIFAR-10 small feasibility check ========\n"
            f"Config unchanged: {cfg_path}\n"
            "================================================================\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(f"\n======== OOM (RuntimeError): {e}\n", file=sys.stderr)
            raise SystemExit(1) from e
        raise


if __name__ == "__main__":
    main()
