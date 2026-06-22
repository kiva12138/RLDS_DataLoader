# RLDS_DataLoader

**A lightweight RLDS dataloader for PyTorch: no TensorFlow, no `dlimp`, and no TensorFlow installation chain.**

`RLDS_DataLoader` streams TFDS-style RLDS TFRecord shards directly into a
PyTorch `IterableDataset`. TFRecord framing and `tf.train.Example` protobuf
messages are parsed in pure Python, while NumPy and Pillow handle numerical
data and JPEG decoding.

> 中文简介：这是一个面向 VLA 训练的轻量级 RLDS 数据加载器。它不依赖
> TensorFlow，也不需要安装 `dlimp`，可直接接入 PyTorch `DataLoader`。

## Why this project?

Many RLDS training pipelines pull in TensorFlow and `dlimp` only to read
TFRecord shards, even when the model is trained entirely with PyTorch. This
project keeps the data path small and explicit:

- zero TensorFlow dependency;
- zero `dlimp` dependency;
- direct streaming from on-disk `*.tfrecord-*-of-*` shards;
- PyTorch `IterableDataset`, multi-worker and DDP rank sharding;
- cached action/proprio statistics and several normalization schemes;
- history windows, future action chunks and padding masks;
- weighted mixing across datasets;
- JPEG decoding and resizing with Pillow.

The current release includes schema support for the **modified LIBERO RLDS**
datasets. RLDS is a data convention rather than one fixed serialized schema;
other datasets require a small dataset config and standardization function.

## Installation

Python 3.10 or newer is required.

**If you don't want to install, copy `src/RLDS_DataLoader/` into your project and import it directly.**

```bash
git clone <YOUR_REPOSITORY_URL>
cd RLDS_DataLoader
pip install -e .
```

Only three runtime packages are required:

```text
numpy
Pillow
torch
```

TensorFlow and `dlimp` are deliberately absent from the dependency list.

## Expected data layout

The bundled LIBERO configs expect TFDS-style directories:

```text
DATA_ROOT/
├── libero_spatial_no_noops/
│   └── 1.0.0/
│       ├── ...-train.tfrecord-00000-of-00016
│       └── ...-val.tfrecord-00000-of-00001
├── libero_object_no_noops/
│   └── 1.0.0/
├── libero_goal_no_noops/
│   └── 1.0.0/
└── libero_10_no_noops/
    └── 1.0.0/
        ├── ...-train.tfrecord-00000-of-00016
        └── ...-val.tfrecord-00000-of-00001
```

The first construction of a dataset scans its selected shards and writes a
`dataset_statistics_<hash>.json` cache next to the data. Later runs reuse the
cache.

## Quick start

```python
from torch.utils.data import DataLoader

from RLDS_DataLoader import RLDSDataset

dataset = RLDSDataset(
    data_root_dir="/path/to/modified_libero_rlds",
    data_mix="libero_10_no_noops",
    resize_resolution=(224, 224),
    window_size=1,
    action_chunk_size=10,  # current action + 9 future actions
    load_proprio=True,
    load_wrist=True,
    normalization_type="bounds_q99",
    shuffle_buffer_size=4096,
    seed=42,
)

loader = DataLoader(
    dataset,
    batch_size=16,
    num_workers=4,
    persistent_workers=True,
)

batch = next(iter(loader))
print(batch["observation"]["image_primary"].shape)
# torch.Size([16, 1, 224, 224, 3]) -- uint8 NHWC

print(batch["action_chunk"].shape)
# torch.Size([16, 10, 7])
```

The dataset is an infinite training stream. Control epoch/step length in your
training loop rather than waiting for the iterator to terminate.

## Demo

The included inspection script loads a few batches and prints every field,
shape and dtype:

```bash
python examples/inspect_dataset.py \
  --data-root /path/to/modified_libero_rlds \
  --data-mix libero_10_no_noops \
  --batch-size 4 \
  --num-workers 2 \
  --num-batches 2
```

Useful options:

```bash
python examples/inspect_dataset.py --help
```

## Sample structure

Before PyTorch's default collation, each sample has this structure:

```python
{
    "observation": {
        "image_primary": np.ndarray,  # [W, H, W, 3], uint8
        "image_wrist": np.ndarray,    # [W, H, W, 3], uint8; optional
        "proprio": np.ndarray,        # [W, D], float32; optional
        "pad_mask": np.ndarray,       # [W], bool
    },
    "action_chunk": np.ndarray,       # [W + C - 1, A], float32
    "current_action": np.ndarray,     # [A], float32
    "history_actions": np.ndarray,    # [W - 1, A], float32
    "future_actions": np.ndarray,     # [C - 1, A], float32
    "language_instruction": str,
    "dataset_name": str,
}
```

Here `W` is `window_size`, `C` is `action_chunk_size`, and `A` is the action
dimension. `action_chunk_size` counts the current action, so a value of 10
means one current and nine future actions.

## Main options

| Argument | Default | Meaning |
| --- | ---: | --- |
| `data_mix` | required | Dataset name or named mixture |
| `resize_resolution` | `(224, 224)` | Output image height and width |
| `window_size` | `1` | Observation/history window |
| `action_chunk_size` | `1` | Current plus future action count |
| `normalization_type` | `bounds_q99` | `normal`, `bounds`, or `bounds_q99` |
| `shuffle_buffer_size` | `4096` | Streaming sample shuffle buffer |
| `balance_weights` | `True` | Scale mixture weights by transition count |
| `train` | `True` | Select training or validation shards |
| `rank`, `world_size` | `0`, `1` | Non-overlapping DDP rank views |
| `per_dataset_data_dirs` | `None` | Optional root override per dataset |

## Dataset mixtures

Bundled names are:

```text
libero_spatial_no_noops
libero_object_no_noops
libero_goal_no_noops
libero_10_no_noops
libero_all
```

For datasets stored under different roots:

```python
dataset = RLDSDataset(
    data_root_dir="/default/root",
    data_mix="libero_all",
    per_dataset_data_dirs={
        "libero_10_no_noops": "/another/disk/libero",
    },
)
```

## Distributed training

Pass each process's rank and world size to prevent normal DDP processes from
reading the same shard view:

```python
dataset = RLDSDataset(
    data_root_dir=data_root,
    data_mix="libero_all",
    rank=distributed_rank,
    world_size=distributed_world_size,
)
```

Within each rank, PyTorch `DataLoader` workers receive another non-overlapping
shard view. If there are fewer shards than ranks, shard reuse is unavoidable
and the loader falls back to modulo assignment so no rank stalls.

## Adding another RLDS schema

Add a standardization function and `DatasetConfig` in
`src/RLDS_DataLoader/oxe_configs.py`. The function receives one parsed
`tf.train.Example` as a flattened feature dictionary and must return:

```python
{
    "observation": {
        "image_primary": list[bytes],
        "proprio": np.ndarray,  # optional
    },
    "action": np.ndarray,
    "task": {"language_instruction": str},
}
```

Then register the config in `OXE_DATASET_CONFIGS` and optionally add a named
mixture to `OXE_NAMED_MIXTURES`.

## Design boundaries

- The protobuf reader supports the feature types used by TFDS RLDS episodes:
  `BytesList`, `FloatList`, and `Int64List`.
- CRC fields are skipped for speed rather than validated.
- JPEG/PNG bytes are decoded through Pillow; encoded video features are not
  supported.
- Schema discovery is intentionally explicit. Unknown RLDS datasets are not
  automatically standardized.

## Development

```bash
pip install -e '.[dev]'
pytest
```

Before publishing, add the open-source license you want to use (for example
Apache-2.0 or MIT) as `LICENSE` and declare it in `pyproject.toml`.
