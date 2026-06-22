#!/usr/bin/env python3
"""Load a few RLDS batches and print their shapes and dtypes."""

from __future__ import annotations

import argparse
from collections.abc import Mapping
from typing import Any

import torch
from torch.utils.data import DataLoader

from rldsdataloader import OXE_NAMED_MIXTURES, RLDSDataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", required=True, help="Root containing dataset/version folders")
    parser.add_argument(
        "--data-mix",
        default="libero_10_no_noops",
        choices=sorted(OXE_NAMED_MIXTURES),
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=2)
    parser.add_argument("--window-size", type=int, default=1)
    parser.add_argument("--action-chunk-size", type=int, default=10)
    parser.add_argument("--height", type=int, default=224)
    parser.add_argument("--width", type=int, default=224)
    parser.add_argument("--shuffle-buffer-size", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-wrist", action="store_true", help="Do not load wrist images")
    parser.add_argument("--no-proprio", action="store_true", help="Do not load proprioception")
    return parser.parse_args()


def describe(value: Any, indent: int = 0) -> None:
    prefix = " " * indent
    if isinstance(value, Mapping):
        for key, child in value.items():
            print(f"{prefix}{key}:")
            describe(child, indent + 2)
    elif torch.is_tensor(value):
        print(f"{prefix}shape={tuple(value.shape)} dtype={value.dtype}")
    elif isinstance(value, (list, tuple)):
        preview = value[:2]
        print(f"{prefix}{type(value).__name__}(len={len(value)}, preview={preview!r})")
    else:
        print(f"{prefix}{type(value).__name__}: {value!r}")


def main() -> None:
    args = parse_args()
    dataset = RLDSDataset(
        data_root_dir=args.data_root,
        data_mix=args.data_mix,
        resize_resolution=(args.height, args.width),
        window_size=args.window_size,
        action_chunk_size=args.action_chunk_size,
        load_proprio=not args.no_proprio,
        load_wrist=not args.no_wrist,
        shuffle_buffer_size=args.shuffle_buffer_size,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=args.num_workers > 0,
    )

    print("Dataset statistics:")
    for name, stats in dataset.dataset_statistics.items():
        print(
            f"  {name}: {stats['num_trajectories']} trajectories, "
            f"{stats['num_transitions']} transitions"
        )

    for batch_index, batch in zip(range(args.num_batches), loader):
        print(f"\nBatch {batch_index}:")
        describe(batch, indent=2)


if __name__ == "__main__":
    main()
