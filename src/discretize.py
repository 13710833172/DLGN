"""把训练好的软模型 checkpoint 转成“可离散推理”的打包文件"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.logic_net import DiffLogicNet


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", type=str, required=True)
    ap.add_argument("--out", type=str, default="")
    args = ap.parse_args()

    ckpt_path = Path(args.checkpoint)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    mcfg = cfg["model"]
    ocfg = cfg["output"]
    ccfg = cfg.get("checkpoint", {})

    conn_seed = int(ckpt.get("connectivity_seed", ccfg.get("seed", 0)))
    model = DiffLogicNet(
        input_dim=int(mcfg["input_dim"]),
        num_layers=int(mcfg["num_layers"]),
        neurons_per_layer=int(mcfg["neurons_per_layer"]),
        connectivity_seed=conn_seed,
        gate_init=str(mcfg.get("gate_init", "normal")),
    )
    model.load_state_dict(ckpt["model_state_dict"], strict=True)

    gate_ids = [g.cpu() for g in model.discretized_gate_ids()]
    conns = [c.cpu() for c in model.connectivity_tensors()]

    out_path = Path(args.out) if args.out else ckpt_path.with_name(
        ckpt_path.stem + "_discrete.pt"
    )
    payload = {
        "connectivity_seed": conn_seed,
        "connectivities": conns,
        "gate_ids_per_layer": gate_ids,
        "meta": {
            "num_layers": int(mcfg["num_layers"]),
            "neurons_per_layer": int(mcfg["neurons_per_layer"]),
            "input_dim": int(mcfg["input_dim"]),
            "num_classes": int(mcfg["num_classes"]),
            "tau": float(ocfg["tau"]),
            "beta": float(ocfg["beta"]),
            "source_checkpoint": str(ckpt_path),
        },
    }
    torch.save(payload, out_path)
    print("Saved discretized bundle to", out_path)


if __name__ == "__main__":
    main()
