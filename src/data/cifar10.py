"""CIFAR-10 loaders with paper-style binary embeddings (no CNN augmentations).

Two embedding modes (select via ``data.cifar10_embedding`` in YAML):

**paper_small** (CIFAR-10 small / default if omitted):
  - ``ToTensor()`` → values in [0, 1].
  - Per channel, three strict thresholds: 0.25, 0.5, 0.75.
  - Flat dim: ``32x32x3x3 = 9216`` (``CIFAR10_PAPER_BINARY_INPUT_DIM``).

**paper_large** (CIFAR-10 large):
  - ``ToTensor()`` → values in [0, 1].
  - Color-channel resolution **32**: per channel, thresholds ``i/32`` for ``i = 1, …, 31``
    (strict inequality ``value > i/32``), i.e. 31 binary indicators per channel per pixel.
  - Layout: channels R,G,B (indices 0,1,2); within each channel, thresholds in increasing
    ``i``; each threshold plane flattened **height then width** row-major.
  - Flat dim: ``32x32x3x31 = 95232`` (``CIFAR10_PAPER_LARGE_INPUT_DIM``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms

# --- Small (3 thresholds per channel) ---

CIFAR10_PAPER_THRESHOLD_LEVELS: tuple[float, ...] = (0.25, 0.5, 0.75)

CIFAR10_PAPER_BINARY_INPUT_DIM: int = 32 * 32 * 3 * len(CIFAR10_PAPER_THRESHOLD_LEVELS)

# --- Large (resolution 32 → 31 thresholds per channel: i/32, i=1..31) ---

CIFAR10_PAPER_LARGE_COLOR_RESOLUTION: int = 32
# i = 1 .. 31  →  thresholds 1/32, 2/32, ..., 31/32
CIFAR10_PAPER_LARGE_THRESHOLDS: tuple[float, ...] = tuple(
    i / float(CIFAR10_PAPER_LARGE_COLOR_RESOLUTION) for i in range(1, CIFAR10_PAPER_LARGE_COLOR_RESOLUTION)
)

CIFAR10_PAPER_LARGE_INPUT_DIM: int = (
    32 * 32 * 3 * len(CIFAR10_PAPER_LARGE_THRESHOLDS)
) # 95232 = 32 * 32 * 3 * 31

# 实现 3-thresholds
def _embed_cifar10_paper_binary(x: torch.Tensor) -> torch.Tensor:
    """把一张 CIFAR-10 图像，从 [3,32,32] 的连续张量，变成一个 9216 维二值向量. 将RGB每个通道用3个阈值进行二值化"""
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    c, h, w = x.shape
    if c != 3:
        raise ValueError(f"CIFAR-10 expects 3 channels, got {c}")
    if h != 32 or w != 32:
        raise ValueError(f"CIFAR-10 expects 32x32 spatial size, got {h}x{w}")

    chunks: list[torch.Tensor] = []
    for ci in range(3): #  遍历 RGB 三个通道
        plane = x[ci] # 获取当前通道的图像平面
        for t in CIFAR10_PAPER_THRESHOLD_LEVELS:
            chunks.append((plane > t).float().reshape(-1)) # 将当前通道的图像平面与当前阈值进行比较，得到一个布尔值，然后将其转换为浮点数，并 reshape 成一个一维向量
    out = torch.cat(chunks, dim=0) # 将所有通道的图像平面拼接成一个一维向量
    # 所以对单个样本来说，Flatten 后得到的是一个 9216 维向量，这个向量里有9216个浮点数，每个浮点数都是0或1
    if out.numel() != CIFAR10_PAPER_BINARY_INPUT_DIM:
        raise RuntimeError(
            f"internal error: small embedding length {out.numel()} != {CIFAR10_PAPER_BINARY_INPUT_DIM}"
        )
    return out

#实现 31-thresholds
def _embed_cifar10_paper_large(x: torch.Tensor) -> torch.Tensor:
    """Large embedding: [C,H,W] in [0,1] → flat 95232-dim binary {0,1} vector.

    Per channel, 31 indicators at thresholds i/32 for i = 1..31 (strict ``>``).
    """
    if x.dim() != 3:
        raise ValueError(f"Expected CHW tensor, got shape {tuple(x.shape)}")
    c, h, w = x.shape
    if c != 3:
        raise ValueError(f"CIFAR-10 expects 3 channels, got {c}")
    if h != 32 or w != 32:
        raise ValueError(f"CIFAR-10 expects 32x32 spatial size, got {h}x{w}")

    chunks: list[torch.Tensor] = []
    for ci in range(3):
        plane = x[ci]
        for t in CIFAR10_PAPER_LARGE_THRESHOLDS:
            chunks.append((plane > t).float().reshape(-1))
    out = torch.cat(chunks, dim=0)
    if out.numel() != CIFAR10_PAPER_LARGE_INPUT_DIM:
        raise RuntimeError(
            f"internal error: large embedding length {out.numel()} != {CIFAR10_PAPER_LARGE_INPUT_DIM}"
        )
    return out


class _Cifar10PaperSmallTransform:
    """把前面的函数包装成一个 callable transform 对象，方便塞进 torchvision.transforms.Compose"""
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _embed_cifar10_paper_binary(x)


class _Cifar10PaperLargeTransform:
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return _embed_cifar10_paper_large(x)


Cifar10Embedding = Literal["paper_small", "paper_large"]


def get_cifar10_loaders(
    batch_size: int,
    data_dir: str | Path = "data",
    val_split: int = 5000,
    data_seed: int = 42,
    num_workers: int = 2,
    embedding: Cifar10Embedding = "paper_small",
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val (from train) / test loaders for CIFAR-10 with chosen paper embedding."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if embedding == "paper_small":
        inner = _Cifar10PaperSmallTransform()
    elif embedding == "paper_large":
        inner = _Cifar10PaperLargeTransform()
    else:
        raise ValueError(f"embedding must be 'paper_small' or 'paper_large', got {embedding!r}")

    #----------------------------------------
    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            inner, # 二值化操作
        ]
    )
    #----------------------------------------
    train_full = datasets.CIFAR10(
        root=str(data_dir), train=True, download=True, transform=tfm
    )
    test_set = datasets.CIFAR10(
        root=str(data_dir), train=False, download=True, transform=tfm
    )

    if val_split <= 0 or val_split >= len(train_full):
        raise ValueError("val_split must be in (0, len(train_full))")

    gen = torch.Generator().manual_seed(int(data_seed))
    train_set, val_set = random_split(
        train_full,
        [len(train_full) - val_split, val_split],
        generator=gen,
    )

    train_loader = DataLoader(
        train_set,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader, test_loader
