"""Lightweight RLDS pipeline.

A pure-Python / PyTorch replacement for the ``rlds_datasets`` package: no
``tensorflow`` or ``dlimp`` dependency.  Supports

* on-disk reading of TFDS-format RLDS shards (``*.tfrecord-*-of-*``),
* per-dataset standardization,
* action/proprio normalization with cached statistics,
* history window + future action-chunk slicing (with pad mask),
* weighted mixing across datasets, and
* shuffle-buffer + worker-sharded ``IterableDataset`` consumption.

Sample dict produced by the default batch transform::

    {
        "observation": {
            "image_primary": np.uint8[W, H, W_, 3],   # history window
            "image_wrist":   np.uint8[W, H, W_, 3],   # if dataset has wrist cam
            "proprio":       np.float32[W, D],        # if load_proprio
            "pad_mask":      np.bool_[W],
        },
        "action_chunk":     np.float32[W + F, A],     # full chunk
        "current_action":   np.float32[A],
        "history_actions":  np.float32[W - 1, A],
        "future_actions":   np.float32[F, A],
        "language_instruction": str,
        "dataset_name":     str,
    }
"""
from __future__ import annotations

import glob
import hashlib
import inspect
import io
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from torch.utils.data import IterableDataset, get_worker_info

from .oxe_configs import (
    OXE_DATASET_CONFIGS,
    DatasetConfig,
    resolve_mixture,
)
from .tfrecord import iter_examples


# --- statistics ------------------------------------------------------------

def _stats_cache_path(data_dir: Path, cfg: DatasetConfig) -> Path:
    src = inspect.getsource(cfg.standardize_fn)
    h = hashlib.sha1(src.encode("utf-8")).hexdigest()[:8]
    return data_dir / f"dataset_statistics_{h}.json"


def _aggregate_stats(arrs: np.ndarray) -> Dict[str, list]:
    return {
        "mean": arrs.mean(0).tolist(),
        "std":  arrs.std(0).tolist(),
        "min":  arrs.min(0).tolist(),
        "max":  arrs.max(0).tolist(),
        "q01":  np.quantile(arrs, 0.01, axis=0).tolist(),
        "q99":  np.quantile(arrs, 0.99, axis=0).tolist(),
    }


def compute_or_load_statistics(
    cfg: DatasetConfig, data_dir: Path, shards: Sequence[Path],
) -> Dict[str, Any]:
    cache = _stats_cache_path(data_dir, cfg)
    if cache.exists():
        with cache.open() as f:
            return json.load(f)

    actions, proprios, n_trans, n_traj = [], [], 0, 0
    for shard in shards:
        for ex in iter_examples(str(shard)):
            traj = cfg.standardize_fn(ex)
            actions.append(traj["action"])
            proprios.append(traj["observation"]["proprio"])
            n_trans += traj["action"].shape[0]
            n_traj += 1

    actions = np.concatenate(actions, axis=0)
    proprios = np.concatenate(proprios, axis=0)
    stats = {
        "action":           _aggregate_stats(actions),
        "proprio":          _aggregate_stats(proprios),
        "num_transitions":  int(n_trans),
        "num_trajectories": int(n_traj),
    }
    if cfg.action_normalization_mask:
        stats["action"]["mask"] = list(cfg.action_normalization_mask)

    try:
        with cache.open("w") as f:
            json.dump(stats, f)
    except OSError:
        fallback = Path.home() / ".cache" / "RLDS_DataLoader" / cache.name
        fallback.parent.mkdir(parents=True, exist_ok=True)
        with fallback.open("w") as f:
            json.dump(stats, f)
    return stats


# --- normalization ---------------------------------------------------------

def _normalize_array(x: np.ndarray, info: Dict[str, Any], scheme: str) -> np.ndarray:
    mask = np.asarray(info.get("mask", [True] * x.shape[-1]), dtype=bool)
    if scheme == "normal":
        mean = np.asarray(info["mean"], dtype=np.float32)
        std = np.asarray(info["std"], dtype=np.float32) + 1e-8
        normed = (x - mean) / std
    elif scheme in ("bounds", "bounds_q99"):
        low = np.asarray(info["min" if scheme == "bounds" else "q01"], dtype=np.float32)
        high = np.asarray(info["max" if scheme == "bounds" else "q99"], dtype=np.float32)
        normed = np.clip(2 * (x - low) / (high - low + 1e-8) - 1, -1, 1)
        # dims with min == max collapse to 0 (e.g. unused action dims)
        zero_dims = np.asarray(info["min"], dtype=np.float32) == np.asarray(info["max"], dtype=np.float32)
        normed = np.where(zero_dims, 0.0, normed)
    else:
        raise ValueError(f"unknown normalization scheme {scheme!r}")
    return np.where(mask, normed, x).astype(np.float32, copy=False)


def normalize_traj(traj: Dict[str, Any], stats: Dict[str, Any], scheme: str) -> Dict[str, Any]:
    traj["action"] = _normalize_array(traj["action"], stats["action"], scheme)
    if "proprio" in traj["observation"]:
        traj["observation"]["proprio"] = _normalize_array(
            traj["observation"]["proprio"], stats["proprio"], scheme,
        )
    return traj


# --- chunking --------------------------------------------------------------

def iter_chunks(
    traj: Dict[str, Any],
    window_size: int,
    future_action_window_size: int,
    absolute_action_mask: np.ndarray,
    skip_unlabeled: bool,
):
    """Yield one chunked frame per timestep ``t`` in the trajectory.

    For step ``t`` the observation window covers ``[t-W+1, t]`` and the
    action chunk covers ``[t-W+1, t+F]``.  Indices before the start of the
    trajectory are clamped to 0 (pad_mask marks them); indices past the
    last step are clamped to ``T-1`` and zeroed (or repeated for absolute
    action dims).
    """
    T = traj["action"].shape[0]
    W, F = window_size, future_action_window_size
    lang = traj["task"]["language_instruction"]
    if skip_unlabeled and not lang:
        return

    obs_offset = np.arange(-W + 1, 1)
    act_offset = np.arange(-W + 1, F + 1)

    for t in range(T):
        obs_idx = obs_offset + t
        act_idx = act_offset + t
        pad_mask = obs_idx >= 0
        obs_idx_c = np.maximum(obs_idx, 0)
        act_idx_c = np.clip(act_idx, 0, T - 1)

        obs_chunk: Dict[str, Any] = {}
        for k, v in traj["observation"].items():
            if isinstance(v, np.ndarray):
                obs_chunk[k] = v[obs_idx_c]
            else:                              # list of bytes (encoded image)
                obs_chunk[k] = [v[i] for i in obs_idx_c]
        obs_chunk["pad_mask"] = pad_mask

        chunk = traj["action"][act_idx_c]                              # [W+F, A]
        neutral = np.where(absolute_action_mask, chunk, 0.0)
        past_goal = (act_idx > T - 1)[:, None]
        chunk = np.where(past_goal, neutral, chunk).astype(np.float32)

        yield {
            "observation": obs_chunk,
            "action": chunk,
            "task": {"language_instruction": lang},
            "dataset_name": traj["dataset_name"],
        }


# --- frame transforms ------------------------------------------------------

def _decode_jpeg(buf: bytes, resize: Tuple[int, int]) -> np.ndarray:
    if not buf:
        return np.zeros((*resize, 3), dtype=np.uint8)
    img = Image.open(io.BytesIO(buf)).convert("RGB")
    if img.size != (resize[1], resize[0]):       # PIL is (W, H)
        img = img.resize((resize[1], resize[0]), Image.BILINEAR)
    return np.asarray(img, dtype=np.uint8)


def decode_obs_images(obs: Dict[str, Any], resize_size: Tuple[int, int]) -> None:
    """Decode every ``image_*`` byte list in ``obs`` to ``(N, H, W, 3) uint8`` in place."""
    for k in list(obs.keys()):
        if not k.startswith("image_"):
            continue
        obs[k] = np.stack([_decode_jpeg(b, resize_size) for b in obs[k]], axis=0)


# --- per-dataset stream ----------------------------------------------------

@dataclass
class _DatasetSpec:
    cfg: DatasetConfig
    data_dir: Path
    shards: List[Path]
    stats: Dict[str, Any]
    weight: float
    absolute_action_mask: np.ndarray
    load_proprio: bool
    load_wrist: bool
    resize_size: Tuple[int, int]
    window_size: int
    future_action_window_size: int
    skip_unlabeled: bool
    normalization_type: str


def _drop_unused_obs_keys(traj: Dict[str, Any], spec: _DatasetSpec) -> Dict[str, Any]:
    obs = traj["observation"]
    if not spec.load_proprio:
        obs.pop("proprio", None)
    if not spec.load_wrist:
        for k in list(obs.keys()):
            if k in spec.cfg.wrist_keys:
                obs.pop(k)
    return traj


def _per_dataset_stream(spec: _DatasetSpec, rng: np.random.Generator):
    """Infinite generator of fully-prepared frames for one dataset."""
    while True:
        order = list(range(len(spec.shards)))
        rng.shuffle(order)
        for i in order:
            for ex in iter_examples(str(spec.shards[i])):
                traj = spec.cfg.standardize_fn(ex)
                traj["dataset_name"] = spec.cfg.name
                traj = _drop_unused_obs_keys(traj, spec)
                traj = normalize_traj(traj, spec.stats, spec.normalization_type)
                # Decode JPEGs once per trajectory; iter_chunks then slices the
                # resulting ndarray per window instead of re-decoding W bytes per step.
                decode_obs_images(traj["observation"], spec.resize_size)
                yield from iter_chunks(
                    traj,
                    spec.window_size,
                    spec.future_action_window_size,
                    spec.absolute_action_mask,
                    spec.skip_unlabeled,
                )


# --- batch transform -------------------------------------------------------

def default_batch_transform(frame: Dict[str, Any]) -> Dict[str, Any]:
    """Repackage a chunked frame into the canonical training sample."""
    W = frame["observation"]["pad_mask"].shape[0]
    actions = frame["action"]                            # [W+F, A]
    history = actions[: W - 1] if W > 1 else np.zeros((0, actions.shape[-1]), dtype=actions.dtype)
    current = actions[W - 1]
    future = actions[W:]
    return {
        "observation": frame["observation"],
        "action_chunk": actions,
        "current_action": current,
        "history_actions": history,
        "future_actions":  future,
        "language_instruction": frame["task"]["language_instruction"],
        "dataset_name": frame["dataset_name"],
    }


# --- top-level dataset -----------------------------------------------------

def _list_shards(data_root: Path, cfg: DatasetConfig, train: bool) -> List[Path]:
    base = data_root / cfg.name / cfg.version
    train_shards = sorted(Path(p) for p in glob.glob(str(base / "*-train.tfrecord-*-of-*")))
    val_shards = sorted(Path(p) for p in glob.glob(str(base / "*-val.tfrecord-*-of-*")))
    if not train_shards and not val_shards:
        raise FileNotFoundError(f"No TFRecord shards under {base}")
    if val_shards:
        return train_shards if train else val_shards
    # No explicit val split — use the last 5% of shards as validation.
    n_val = max(1, len(train_shards) // 20)
    return train_shards[:-n_val] if train else train_shards[-n_val:]


class RLDSDataset(IterableDataset):
    """Streaming mixture of RLDS datasets with history + action chunking."""

    def __init__(
        self,
        data_root_dir: str | os.PathLike,
        data_mix: str,
        *,
        resize_resolution: Tuple[int, int] = (224, 224),
        window_size: int = 1,
        action_chunk_size: int = 1,
        load_proprio: bool = True,
        load_wrist: bool = True,
        skip_unlabeled: bool = True,
        normalization_type: str = "bounds_q99",
        balance_weights: bool = True,
        shuffle_buffer_size: int = 4096,
        train: bool = True,
        batch_transform: Callable[[Dict[str, Any]], Dict[str, Any]] = default_batch_transform,
        per_dataset_data_dirs: Optional[Dict[str, str | os.PathLike]] = None,
        seed: int = 0,
        rank: int = 0,
        world_size: int = 1,
    ) -> None:
        super().__init__()
        if action_chunk_size < 1:
            raise ValueError("action_chunk_size must be >= 1 (current step counts)")
        if window_size < 1:
            raise ValueError("window_size must be >= 1")
        if world_size < 1:
            raise ValueError("world_size must be >= 1")
        if not 0 <= rank < world_size:
            raise ValueError(f"rank={rank} must be in [0, {world_size})")

        self.data_root_dir = Path(data_root_dir)
        self.per_dataset_data_dirs = {k: Path(v) for k, v in (per_dataset_data_dirs or {}).items()}
        self.window_size = window_size
        self.future_action_window_size = action_chunk_size - 1
        self.resize_resolution = tuple(resize_resolution)
        self.load_proprio = load_proprio
        self.load_wrist = load_wrist
        self.skip_unlabeled = skip_unlabeled
        self.normalization_type = normalization_type
        self.shuffle_buffer_size = max(1, shuffle_buffer_size)
        self.batch_transform = batch_transform
        self.seed = seed
        self.train = train
        self.rank = rank
        self.world_size = world_size

        mixture = resolve_mixture(data_mix)
        specs: List[_DatasetSpec] = []
        sizes: List[int] = []
        self.dataset_statistics: Dict[str, Dict[str, Any]] = {}
        for name, weight in mixture:
            cfg = OXE_DATASET_CONFIGS[name]
            root = self.per_dataset_data_dirs.get(name, self.data_root_dir)
            shards = _list_shards(root, cfg, train)
            data_dir = root / cfg.name / cfg.version
            stats = compute_or_load_statistics(cfg, data_dir, shards)
            self.dataset_statistics[name] = stats
            specs.append(_DatasetSpec(
                cfg=cfg,
                data_dir=data_dir,
                shards=shards,
                stats=stats,
                weight=float(weight),
                absolute_action_mask=np.asarray(cfg.absolute_action_mask, dtype=bool),
                load_proprio=load_proprio,
                load_wrist=load_wrist,
                resize_size=self.resize_resolution,
                window_size=window_size,
                future_action_window_size=self.future_action_window_size,
                skip_unlabeled=skip_unlabeled,
                normalization_type=normalization_type,
            ))
            sizes.append(stats["num_transitions"])

        weights = np.asarray([s.weight for s in specs], dtype=np.float64)
        if balance_weights:
            weights = weights * np.asarray(sizes, dtype=np.float64)
        self.sample_weights = (weights / weights.sum()).astype(np.float64)
        self._specs = specs
        self._sizes = sizes
        primary = np.array([i for i, s in enumerate(specs) if s.weight == 1.0])
        if len(primary) == 0:
            primary = np.arange(len(specs))
        self.dataset_length = int((np.asarray(sizes) / self.sample_weights)[primary].max())

    # ----- iteration -------------------------------------------------------

    def _make_rng(self) -> np.random.Generator:
        info = get_worker_info()
        wid = info.id if info is not None else 0
        num_workers = info.num_workers if info is not None else 1
        global_worker_id = self.rank * num_workers + wid
        return np.random.default_rng(self.seed + global_worker_id * 1_000_003)

    def _worker_shard_view(self, shards: List[Path]) -> List[Path]:
        """Return this DDP rank/DataLoader worker's non-overlapping shards.

        Rank sharding is applied first so every process retains its own view;
        worker sharding is then applied inside that rank. If there are fewer
        shards than ranks, modulo fallback keeps every rank alive at the cost
        of unavoidable shard reuse.
        """
        if not shards:
            return []
        rank_shards = list(shards[self.rank :: self.world_size])
        if not rank_shards:
            rank_shards = [shards[self.rank % len(shards)]]

        info = get_worker_info()
        if info is None or info.num_workers <= 1:
            return rank_shards
        return rank_shards[info.id :: info.num_workers]

    def __iter__(self):
        rng = self._make_rng()
        streams, kept_weights = [], []
        for spec, w in zip(self._specs, self.sample_weights):
            shards = self._worker_shard_view(spec.shards)
            if not shards:
                continue
            local_spec = _DatasetSpec(**{**spec.__dict__, "shards": shards})
            streams.append(_per_dataset_stream(local_spec, rng))
            kept_weights.append(w)

        if not streams:
            return
        weights = np.asarray(kept_weights, dtype=np.float64)
        weights = weights / weights.sum()

        # Pre-fill shuffle buffer, then sample-and-replace.
        buf: List[Dict[str, Any]] = []
        while len(buf) < self.shuffle_buffer_size:
            i = int(rng.choice(len(streams), p=weights))
            try:
                buf.append(next(streams[i]))
            except StopIteration:
                break
        if not buf:
            return

        while True:
            i = int(rng.choice(len(streams), p=weights))
            try:
                sample = next(streams[i])
            except StopIteration:
                continue
            j = int(rng.integers(0, len(buf)))
            out, buf[j] = buf[j], sample
            yield self.batch_transform(out)

    def __len__(self) -> int:
        return self.dataset_length
