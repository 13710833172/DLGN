"""对 DataLoader 统一入口 + 配置分发 + 一致性校验"""

from __future__ import annotations

from src.data.cifar10 import (
    CIFAR10_PAPER_BINARY_INPUT_DIM,
    CIFAR10_PAPER_LARGE_INPUT_DIM,
    get_cifar10_loaders,
)
from src.data.mnist import get_mnist_loaders


def _norm_dataset(name: str) -> str:
    """Normalize e.g. ``cifar-10`` / ``CIFAR10`` → ``cifar10``."""
    return str(name).lower().replace("-", "").replace("_", "")


def cifar10_embedding_mode(dcfg: dict) -> str:
    """Return ``paper_small`` or ``paper_large`` (default ``paper_small`` if unset)."""
    raw = dcfg.get("cifar10_embedding", "paper_small")
    s = str(raw).lower().replace("-", "_")
    if s in ("paper_small", "small"):
        return "paper_small"
    if s in ("paper_large", "large"):
        return "paper_large"
    raise ValueError(
        "data.cifar10_embedding must be 'paper_small' or 'paper_large', "
        f"got {raw!r}"
    )


def validate_model_data_config(mcfg: dict, dcfg: dict) -> None:
    """检查模型和数据配置是否匹配"""
    nc = int(mcfg["num_classes"])
    npl = int(mcfg["neurons_per_layer"])
    if npl % nc != 0:
        raise ValueError(
            f"neurons_per_layer ({npl}) must be divisible by num_classes ({nc}) "
            "so the final layer can be grouped into classes."
        )

    ds = _norm_dataset(dcfg.get("dataset", "mnist"))
    idim = int(mcfg["input_dim"])

    if ds == "mnist":
        if idim != 784:
            raise ValueError(
                f"MNIST expects model.input_dim==784 (28×28 flattened binary), got {idim}."
            )
    elif ds == "cifar10":
        mode = cifar10_embedding_mode(dcfg)
        if mode == "paper_small":
            expected = CIFAR10_PAPER_BINARY_INPUT_DIM
            hint = "32×32×3×3 (small embedding)"
        else:
            expected = CIFAR10_PAPER_LARGE_INPUT_DIM
            hint = "32×32×3×31 (large embedding, thresholds i/32 for i=1..31)"
        if idim != expected:
            raise ValueError(
                f"CIFAR-10 with data.cifar10_embedding={mode!r} expects "
                f"model.input_dim=={expected} ({hint}), got {idim}."
            )
    else:
        raise ValueError(
            f"Unknown data.dataset {dcfg.get('dataset')!r}; expected 'mnist' or 'cifar10'."
        )


def get_data_loaders(
    dcfg: dict,
    batch_size: int,
    num_workers: int | None = None,
) -> tuple:
    """Return ``(train_loader, val_loader, test_loader)`` for ``dcfg['dataset']``."""
    ds = _norm_dataset(dcfg.get("dataset", "mnist"))
    nw = int(dcfg.get("num_workers", 2)) if num_workers is None else int(num_workers)
    data_dir = dcfg.get("data_dir", "data")
    val_split = int(dcfg.get("val_split", 5000))
    data_seed = int(dcfg.get("data_seed", 42))

    if ds == "mnist":
        return get_mnist_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            binarize_threshold=float(dcfg.get("binarize_threshold", 0.5)),
            val_split=val_split,
            data_seed=data_seed,
            num_workers=nw,
        )
    if ds == "cifar10":
        emb = cifar10_embedding_mode(dcfg)
        return get_cifar10_loaders(
            batch_size=batch_size,
            data_dir=data_dir,
            val_split=val_split,
            data_seed=data_seed,
            num_workers=nw,
            embedding=emb,
        )
    raise ValueError(f"Unknown data.dataset {dcfg.get('dataset')!r}")
