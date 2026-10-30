"""MNIST loaders with train/eval-specific transforms and reproducible split."""

from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms

#----------------------------------------
def _float_flatten():
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1)), # 将图像展平为一维向量
        ]
    )


def _binarize_flatten(threshold: float):
    return transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Lambda(lambda x: (x >= threshold).float()), # 将图像二值化
            transforms.Lambda(lambda x: x.view(-1)),
        ]
    )
#----------------------------------------

def get_mnist_loaders(
    batch_size: int,
    data_dir: str | Path = "data",
    binarize_threshold: float = 0.5,
    val_split: int = 5000,
    data_seed: int = 42,
    num_workers: int = 2,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Train / val (from train) / test loaders."""
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    #----------------------------------------
    train_tfm = _float_flatten()
    eval_tfm = _binarize_flatten(binarize_threshold)
    #----------------------------------------
    train_full_for_split = datasets.MNIST(
        root=str(data_dir), train=True, download=True, transform=train_tfm
    )
    train_full = datasets.MNIST(
        root=str(data_dir), train=True, download=False, transform=train_tfm
    )
    val_full = datasets.MNIST(
        root=str(data_dir), train=True, download=False, transform=eval_tfm
    )
    test_set = datasets.MNIST(
        root=str(data_dir), train=False, download=True, transform=eval_tfm
    )

    if val_split <= 0 or val_split >= len(train_full_for_split):
        raise ValueError("val_split must be in (0, len(train_full))")

    gen = torch.Generator().manual_seed(int(data_seed))
    train_split, val_split_dataset = random_split(
        train_full_for_split,
        [len(train_full_for_split) - val_split, val_split],
        generator=gen,
    )
    train_set = Subset(train_full, train_split.indices)
    val_set = Subset(val_full, val_split_dataset.indices)

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
