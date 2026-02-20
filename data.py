from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms


@dataclass
class MNISTLoaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader


def _filter_indices_by_digit(targets: torch.Tensor, digit: int) -> torch.Tensor:
    return (targets == digit).nonzero(as_tuple=False).squeeze(1)


def _filter_indices_not_digit(targets: torch.Tensor, digit: int) -> torch.Tensor:
    return (targets != digit).nonzero(as_tuple=False).squeeze(1)


def get_mnist_loaders(
    normal_digit: int = 0,
    batch_size: int = 128,
    num_workers: int = 0,
    pin_memory: bool = False,
    val_size: int = 5000,
    anomaly_fraction_in_val: float = 0.5,
    anomaly_fraction_in_test: float = 0.5,
    seed: int = 42,
) -> MNISTLoaders:
    """
    Train: only normal_digit
    Val/Test: mixture of normal and anomaly (all other digits)

    Returns loaders where each batch returns (x, y),
    and y is binary: 0=known normal, 1=anomaly.
    """
    assert 0 <= normal_digit <= 9
    assert 0.0 <= anomaly_fraction_in_val <= 1.0
    assert 0.0 <= anomaly_fraction_in_test <= 1.0

    g = torch.Generator().manual_seed(seed)

    tfm = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.5,), (0.5,)),  # map roughly to [-1, 1]
        ]
    )

    train_full = datasets.MNIST(root="data", train=True, download=True, transform=tfm)
    test_full = datasets.MNIST(root="data", train=False, download=True, transform=tfm)

    train_targets = train_full.targets.clone().detach()
    test_targets = test_full.targets.clone().detach()


    # --- Train indices (ONLY normal digit) ---
    train_idx = _filter_indices_by_digit(train_targets, normal_digit)

    # --- Build Val set from TRAIN split (mix normal + anomaly) ---
    # We'll sample val_size examples total.
    normal_idx_all = _filter_indices_by_digit(train_targets, normal_digit)
    anomaly_idx_all = _filter_indices_not_digit(train_targets, normal_digit)

    n_anom_val = int(val_size * anomaly_fraction_in_val)
    n_norm_val = val_size - n_anom_val

    normal_perm = normal_idx_all[torch.randperm(len(normal_idx_all), generator=g)]
    anomaly_perm = anomaly_idx_all[torch.randperm(len(anomaly_idx_all), generator=g)]

    val_norm_idx = normal_perm[:n_norm_val]
    val_anom_idx = anomaly_perm[:n_anom_val]
    val_idx = torch.cat([val_norm_idx, val_anom_idx])
    val_idx = val_idx[torch.randperm(len(val_idx), generator=g)]  # shuffle

    # --- Test set from TEST split (mix normal + anomaly) ---
    test_size = len(test_full)
    n_anom_test = int(test_size * anomaly_fraction_in_test)
    n_norm_test = test_size - n_anom_test

    test_norm_all = _filter_indices_by_digit(test_targets, normal_digit)
    test_anom_all = _filter_indices_not_digit(test_targets, normal_digit)

    test_norm_perm = test_norm_all[torch.randperm(len(test_norm_all), generator=g)]
    test_anom_perm = test_anom_all[torch.randperm(len(test_anom_all), generator=g)]

    # If there are not enough normal samples (can happen if anomaly_fraction very high), clamp safely
    n_norm_test = min(n_norm_test, len(test_norm_perm))
    n_anom_test = min(n_anom_test, len(test_anom_perm))

    test_norm_idx = test_norm_perm[:n_norm_test]
    test_anom_idx = test_anom_perm[:n_anom_test]
    test_idx = torch.cat([test_norm_idx, test_anom_idx])
    test_idx = test_idx[torch.randperm(len(test_idx), generator=g)]

    # --- Create subsets ---
    train_ds = Subset(train_full, train_idx.tolist())
    val_ds_raw = Subset(train_full, val_idx.tolist())
    test_ds_raw = Subset(test_full, test_idx.tolist())

    # --- Wrap to make binary labels (0 normal, 1 anomaly) ---
    def to_binary_label(original_label: int) -> int:
        return 0 if original_label == normal_digit else 1

    class BinaryLabelWrapper(torch.utils.data.Dataset):
        def __init__(self, subset):
            self.subset = subset

        def __len__(self):
            return len(self.subset)

        def __getitem__(self, i):
            x, y = self.subset[i]
            return x, torch.tensor(to_binary_label(int(y)), dtype=torch.long)

    val_ds = BinaryLabelWrapper(val_ds_raw)
    test_ds = BinaryLabelWrapper(test_ds_raw)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )
    test_loader = DataLoader(
        test_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory
    )

    return MNISTLoaders(train=train_loader, val=val_loader, test=test_loader)
