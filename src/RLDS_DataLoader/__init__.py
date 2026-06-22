"""Lightweight, TensorFlow-free RLDS loading for PyTorch."""

from .oxe_configs import (
    OXE_DATASET_CONFIGS,
    OXE_NAMED_MIXTURES,
    DatasetConfig,
    resolve_mixture,
)
from .rlds_dataset import RLDSDataset, default_batch_transform
from .tfrecord import iter_examples, iter_tfrecord, parse_example

__all__ = [
    "DatasetConfig",
    "OXE_DATASET_CONFIGS",
    "OXE_NAMED_MIXTURES",
    "RLDSDataset",
    "default_batch_transform",
    "iter_examples",
    "iter_tfrecord",
    "parse_example",
    "resolve_mixture",
]

__version__ = "0.1.0"
