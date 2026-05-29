from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class EpochDataset:
    """Container for epoched EEG data in microvolts."""

    X: np.ndarray
    y: np.ndarray
    sfreq: float
    ch_names: list[str]
    subject_ids: np.ndarray
    label_names: dict[int, str]


@dataclass(frozen=True)
class QualityResult:
    valid: bool
    score: float
    bad_channels: list[str]
    reject_reasons: list[str]

