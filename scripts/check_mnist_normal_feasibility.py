#!/usr/bin/env python3
"""One forward + backward + Adam step for configs/mnist_normal_paper.yaml (paper 6×64000).

Exits 0 on success and prints peak CUDA memory; exits 1 on OOM or other failure with a clear message.
Does not modify hyperparameters in the YAML.
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

from src.models.logic_net import DiffLogicNet
from src.utils.output_grouping import compute_class_scores


def load_config(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    ap = argparse.ArgumentParser(description="Smoke test paper MNIST normal (6×64000) one training step.")
    ap.add_argument(
        "--config",
        type=str,
        default=str(ROOT / "configs" / "mnist_normal_paper.yaml"),
        help="YAML config (default: mnist_normal_paper.yaml)",
    )
    args = ap.parse_args()
    cfg_path = Path(args.config).resolve()
    cfg = load_config(cfg_path)

    mcfg = cfg["model"]
    tcfg = cfg["training"]
    ocfg = cfg["output"]
    ccfg = cfg.get("checkpoint", {})

    device_str = tcfg.get("device", "cuda")
    if device_str == "cuda" and not torch.cuda.is_available():
        print("CUDA not available; this feasibility check is intended for GPU.", file=sys.stderr)
        sys.exit(2)

    device = torch.device(device_str)
    conn_seed = int(ccfg.get("seed", 0))
    batch_size = int(tcfg["batch_size"])
    num_classes = int(mcfg["num_classes"])
    tau = float(ocfg["tau"])
    beta = float(ocfg["beta"])

    print(f"Config: {cfg_path}")
    print(
        f"Architecture: layers={mcfg['num_layers']}, neurons_per_layer={mcfg['neurons_per_layer']}, "
        f"batch_size={batch_size}, device={device}"
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

        x = torch.randint(0, 2, (batch_size, int(mcfg["input_dim"])), dtype=torch.float32, device=device)
        y = torch.randint(0, num_classes, (batch_size,), device=device)

        model.train()
        optimizer.zero_grad(set_to_none=True)
        out = model(x)
        logits = compute_class_scores(out, num_classes, tau, beta)
        loss = criterion(logits, y)
        loss.backward()
        optimizer.step()

        if device.type == "cuda":
            torch.cuda.synchronize()
            peak_a = torch.cuda.max_memory_allocated() / (1024**3)
            peak_r = torch.cuda.max_memory_reserved() / (1024**3)
            print(f"OK: one forward+backward+step completed. Peak allocated {peak_a:.3f} GiB, peak reserved {peak_r:.3f} GiB.")
            props = torch.cuda.get_device_properties(device)
            print(f"GPU: {props.name}, total memory {props.total_memory / (1024**3):.2f} GiB.")
            if peak_r > 0.9 * (props.total_memory / (1024**3)):
                print(
                    "Warning: peak reserved memory is close to device capacity; full 200-epoch training may still OOM "
                    "(DataLoader, validation, hard-net construction, optimizer state).",
                    file=sys.stderr,
                )
        else:
            print("OK: one forward+backward+step completed (CPU).")

    except torch.cuda.OutOfMemoryError as e:
        print(
            "\n======== OOM during mnist_normal_paper feasibility step ========\n"
            f"The default paper settings were not changed ({cfg_path}).\n"
            "Full training with batch_size=100 is unlikely to fit on this GPU.\n"
            "================================================================\n",
            file=sys.stderr,
        )
        raise SystemExit(1) from e
    except RuntimeError as e:
        if "out of memory" in str(e).lower():
            print(
                "\n======== OOM (RuntimeError) during feasibility step ========\n"
                f"Message: {e}\n"
                "================================================================\n",
                file=sys.stderr,
            )
            raise SystemExit(1) from e
        raise


if __name__ == "__main__":
    main()
