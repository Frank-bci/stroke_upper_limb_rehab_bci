from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.model_selection import GroupShuffleSplit, train_test_split


@dataclass(frozen=True)
class SplitResult:
    train_idx: np.ndarray
    test_idx: np.ndarray
    strategy: str
    test_subject_ids: list[str]


@dataclass(frozen=True)
class WindowedEpochs:
    X: np.ndarray
    y: np.ndarray
    source_epoch_idx: np.ndarray
    window_center_seconds: np.ndarray


def make_train_test_split(y: np.ndarray, subject_ids: np.ndarray, config: dict) -> SplitResult:
    """Create a reproducible train/test split for offline and pseudo-online evaluation."""

    y = np.asarray(y)
    groups = np.asarray(subject_ids)
    dataset_name = str(config.get("dataset", {}).get("name", "")).lower()
    model_cfg = config.get("model", {})
    strategy = str(model_cfg.get("split_strategy") or _default_strategy(dataset_name)).lower()
    test_size = float(model_cfg.get("test_size", 0.25))
    random_state = int(config.get("dataset", {}).get("random_state", 7))
    indices = np.arange(len(y))

    if strategy in {"subject", "group", "group_shuffle"}:
        return _group_split(indices, y, groups, test_size, random_state)

    if strategy != "random":
        raise ValueError(f"Unsupported split_strategy: {strategy}")

    train_idx, test_idx = train_test_split(
        indices,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )
    return SplitResult(
        train_idx=np.asarray(train_idx, dtype=int),
        test_idx=np.asarray(test_idx, dtype=int),
        strategy="random",
        test_subject_ids=sorted(map(str, np.unique(groups[test_idx]))),
    )


def _default_strategy(dataset_name: str) -> str:
    return "random" if dataset_name == "synthetic" else "subject"


def _group_split(
    indices: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    test_size: float,
    random_state: int,
) -> SplitResult:
    unique_groups = np.unique(groups)
    if len(unique_groups) < 2:
        raise ValueError("Subject-aware split requires at least two subject_ids.")

    splitter = GroupShuffleSplit(n_splits=16, test_size=test_size, random_state=random_state)
    last_split: tuple[np.ndarray, np.ndarray] | None = None
    for train_idx, test_idx in splitter.split(indices, y, groups):
        last_split = (train_idx, test_idx)
        if len(np.unique(y[train_idx])) >= 2 and len(np.unique(y[test_idx])) >= 2:
            return SplitResult(
                train_idx=np.asarray(train_idx, dtype=int),
                test_idx=np.asarray(test_idx, dtype=int),
                strategy="subject",
                test_subject_ids=sorted(map(str, np.unique(groups[test_idx]))),
            )

    if last_split is None:
        raise RuntimeError("Unable to create a subject-aware split.")

    train_idx, test_idx = last_split
    raise ValueError(
        "Subject-aware split produced a single-class train or test set. "
        f"Check label balance for held-out subjects: {sorted(map(str, np.unique(groups[test_idx])))}"
    )


def make_online_windows(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    config: dict,
    epoch_indices: np.ndarray | None = None,
) -> WindowedEpochs:
    """Convert epochs into online-style sliding windows for model training/evaluation."""

    online_cfg = config["online"]
    window_samples = int(round(float(online_cfg["window_seconds"]) * sfreq))
    step_samples = int(round(float(online_cfg["step_seconds"]) * sfreq))
    task_start_seconds = float(online_cfg["task_start_seconds"])
    task_end_seconds = float(online_cfg["task_end_seconds"])

    if window_samples <= 0 or step_samples <= 0:
        raise ValueError("window_seconds and step_seconds must produce positive sample counts.")
    if X.shape[-1] < window_samples:
        raise ValueError("Epochs are shorter than the configured online window.")

    if epoch_indices is None:
        epoch_indices = np.arange(len(y))
    epoch_indices = np.asarray(epoch_indices, dtype=int)

    windows: list[np.ndarray] = []
    labels: list[int] = []
    source_epoch_idx: list[int] = []
    centers: list[float] = []

    for local_idx, (epoch, label) in enumerate(zip(X, y)):
        for start in range(0, epoch.shape[-1] - window_samples + 1, step_samples):
            stop = start + window_samples
            center = (start + window_samples / 2) / sfreq
            if not (task_start_seconds <= center <= task_end_seconds):
                continue
            windows.append(epoch[:, start:stop])
            labels.append(int(label))
            source_epoch_idx.append(int(epoch_indices[local_idx]))
            centers.append(float(center))

    if not windows:
        raise RuntimeError("No online training windows were produced. Check online window/task timing config.")

    return WindowedEpochs(
        X=np.asarray(windows),
        y=np.asarray(labels, dtype=int),
        source_epoch_idx=np.asarray(source_epoch_idx, dtype=int),
        window_center_seconds=np.asarray(centers, dtype=float),
    )
