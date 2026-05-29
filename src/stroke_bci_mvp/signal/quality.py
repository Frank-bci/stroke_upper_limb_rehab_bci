from __future__ import annotations

import numpy as np
from scipy import signal

from stroke_bci_mvp.types import QualityResult


def assess_epoch_quality(
    epoch: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    config: dict,
) -> QualityResult:
    max_abs_uv = float(config.get("max_abs_uv", 150))
    min_channel_std_uv = float(config.get("min_channel_std_uv", 0.05))
    max_bad_channel_ratio = float(config.get("max_bad_channel_ratio", 0.35))
    max_line_noise_ratio = float(config.get("max_line_noise_ratio", 0.35))
    min_score = float(config.get("min_score", 70))

    reasons: list[str] = []
    bad_channels: set[str] = set()
    score = 100.0

    peak = np.max(np.abs(epoch), axis=1)
    flat = np.std(epoch, axis=1)
    for idx, ch in enumerate(ch_names):
        if peak[idx] > max_abs_uv:
            bad_channels.add(ch)
        if flat[idx] < min_channel_std_uv:
            bad_channels.add(ch)

    if bad_channels:
        reasons.append("bad_channel_amplitude_or_flatline")
        score -= min(35.0, 10.0 * len(bad_channels))

    bad_ratio = len(bad_channels) / max(1, len(ch_names))
    if bad_ratio > max_bad_channel_ratio:
        reasons.append("too_many_bad_channels")
        score -= 35.0

    freqs, psd = signal.welch(epoch, fs=sfreq, nperseg=min(epoch.shape[-1], int(sfreq)))
    total_power = np.trapz(psd, freqs, axis=-1) + 1e-12
    line_mask = (freqs >= 48) & (freqs <= 52)
    if np.any(line_mask):
        line_power = np.trapz(psd[:, line_mask], freqs[line_mask], axis=-1)
        line_ratio = float(np.median(line_power / total_power))
        if line_ratio > max_line_noise_ratio:
            reasons.append("line_noise")
            score -= 20.0

    score = float(np.clip(score, 0, 100))
    valid = score >= min_score and "too_many_bad_channels" not in reasons
    return QualityResult(valid=valid, score=score, bad_channels=sorted(bad_channels), reject_reasons=reasons)


def filter_valid_epochs(
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    config: dict,
) -> tuple[np.ndarray, np.ndarray, list[QualityResult]]:
    results = [assess_epoch_quality(epoch, sfreq, ch_names, config) for epoch in X]
    keep = np.asarray([result.valid for result in results], dtype=bool)
    return X[keep], y[keep], results

