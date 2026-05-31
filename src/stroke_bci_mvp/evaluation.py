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
