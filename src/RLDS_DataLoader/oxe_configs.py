"""Per-dataset configs, named mixtures, and standardize functions.

A *dataset config* describes the raw RLDS schema we expect on disk and
which keys map onto our canonical ``observation`` / ``action`` layout.  The
*standardize* function takes the parsed episode (a dict of feature -> list)
and reshapes / renames fields so the downstream pipeline can stay schema-
agnostic.

Only the LIBERO modified-RLDS family is included here; add new entries by
following the same pattern.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np


# --- standardize functions -------------------------------------------------

def _stack_per_step(values: Sequence[float], traj_len: int, dim: int) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    if arr.size != traj_len * dim:
        raise ValueError(f"expected {traj_len * dim} values, got {arr.size}")
    return arr.reshape(traj_len, dim)


def libero_standardize(episode: Dict[str, Tuple[str, list]]) -> Dict[str, Any]:
    """Restructure a LIBERO RLDS episode into the canonical layout.

    The modified-RLDS LIBERO recordings store:
      action            FloatList   T*7        (6-DoF eef delta + gripper)
      observation/state FloatList   T*8        (6-DoF pose + 2-DoF gripper)
      observation/image BytesList   T          (JPEG-encoded RGB)
      observation/wrist_image BytesList T
      language_instruction BytesList T         (repeated; we keep one)
    """
    action_flat = episode["steps/action"][1]
    state_flat = episode["steps/observation/state"][1]
    img_primary = episode["steps/observation/image"][1] # Already up-down reversed in the dataset 
    img_wrist = episode["steps/observation/wrist_image"][1] # Already up-down reversed in the dataset 
    lang = episode["steps/language_instruction"][1]

    traj_len = len(img_primary)
    action = _stack_per_step(action_flat, traj_len, 7)
    state = _stack_per_step(state_flat, traj_len, 8)

    # Gripper: original is -1 (open) .. 1 (close) → clip to [0,1], invert → +1 open / 0 close.
    gripper = np.clip(action[:, -1:], 0.0, 1.0)
    gripper = 1.0 - gripper
    action = np.concatenate([action[:, :6], gripper], axis=1)

    # Proprio = EEF_pose (6) + a single zero pad + gripper_open/close (1)  → 8-D.
    proprio = np.concatenate(
        [state[:, :6], np.zeros((traj_len, 1), dtype=np.float32), state[:, -1:]],
        axis=1,
    )

    return {
        "observation": {
            "image_primary": list(img_primary),
            "image_wrist": list(img_wrist),
            "proprio": proprio,
        },
        "action": action,
        "task": {"language_instruction": lang[0].decode("utf-8")},
    }


# --- dataset config --------------------------------------------------------

@dataclass
class DatasetConfig:
    """Static metadata describing one RLDS dataset on disk."""

    name: str
    standardize_fn: Callable[[Dict[str, Tuple[str, list]]], Dict[str, Any]]
    action_dim: int
    proprio_dim: int
    image_keys: Tuple[str, ...] = ("image_primary",)
    wrist_keys: Tuple[str, ...] = ()
    # gripper dim is absolute (don't zero out past the goal) and not normalized
    absolute_action_mask: Tuple[bool, ...] = ()
    action_normalization_mask: Tuple[bool, ...] = ()
    version: str = "1.0.0"


_LIBERO_BASE = dict(
    standardize_fn=libero_standardize,
    action_dim=7,
    proprio_dim=8,
    image_keys=("image_primary",),
    wrist_keys=("image_wrist",),
    absolute_action_mask=(False,) * 6 + (True,),
    action_normalization_mask=(True,) * 6 + (False,),
)

OXE_DATASET_CONFIGS: Dict[str, DatasetConfig] = {
    "libero_spatial_no_noops": DatasetConfig(name="libero_spatial_no_noops", **_LIBERO_BASE),
    "libero_object_no_noops":  DatasetConfig(name="libero_object_no_noops",  **_LIBERO_BASE),
    "libero_goal_no_noops":    DatasetConfig(name="libero_goal_no_noops",    **_LIBERO_BASE),
    "libero_10_no_noops":      DatasetConfig(name="libero_10_no_noops",      **_LIBERO_BASE),
}


# --- named mixtures --------------------------------------------------------

OXE_NAMED_MIXTURES: Dict[str, List[Tuple[str, float]]] = {
    "libero_spatial_no_noops": [("libero_spatial_no_noops", 1.0)],
    "libero_object_no_noops":  [("libero_object_no_noops",  1.0)],
    "libero_goal_no_noops":    [("libero_goal_no_noops",    1.0)],
    "libero_10_no_noops":      [("libero_10_no_noops",      1.0)],
    "libero_all": [
        ("libero_spatial_no_noops", 1.0),
        ("libero_object_no_noops",  1.0),
        ("libero_goal_no_noops",    1.0),
        ("libero_10_no_noops",      1.0),
    ],
}


def resolve_mixture(name: str) -> List[Tuple[str, float]]:
    if name in OXE_NAMED_MIXTURES:
        return list(OXE_NAMED_MIXTURES[name])
    if name in OXE_DATASET_CONFIGS:
        return [(name, 1.0)]
    raise KeyError(f"unknown dataset / mixture: {name!r}")
