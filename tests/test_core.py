from __future__ import annotations

import numpy as np

from RLDS_DataLoader.rlds_dataset import default_batch_transform, iter_chunks


def test_history_and_future_padding() -> None:
    trajectory = {
        "observation": {
            "image_primary": np.zeros((2, 4, 4, 3), dtype=np.uint8),
            "proprio": np.arange(4, dtype=np.float32).reshape(2, 2),
        },
        "action": np.asarray([[1.0, 0.25], [2.0, 0.75]], dtype=np.float32),
        "task": {"language_instruction": "move"},
        "dataset_name": "test",
    }

    frames = list(
        iter_chunks(
            trajectory,
            window_size=2,
            future_action_window_size=2,
            absolute_action_mask=np.asarray([False, True]),
            skip_unlabeled=True,
        )
    )

    assert len(frames) == 2
    np.testing.assert_array_equal(frames[0]["observation"]["pad_mask"], [False, True])
    # Beyond the trajectory, relative dimensions become zero while absolute
    # dimensions retain the terminal value.
    np.testing.assert_allclose(frames[-1]["action"][-1], [0.0, 0.75])

    sample = default_batch_transform(frames[0])
    assert sample["history_actions"].shape == (1, 2)
    assert sample["current_action"].shape == (2,)
    assert sample["future_actions"].shape == (2, 2)


def test_unlabeled_trajectory_is_skipped() -> None:
    trajectory = {
        "observation": {"proprio": np.zeros((1, 2), dtype=np.float32)},
        "action": np.zeros((1, 2), dtype=np.float32),
        "task": {"language_instruction": ""},
        "dataset_name": "test",
    }
    frames = list(
        iter_chunks(
            trajectory,
            window_size=1,
            future_action_window_size=0,
            absolute_action_mask=np.asarray([False, False]),
            skip_unlabeled=True,
        )
    )
    assert frames == []
