from __future__ import annotations

import numpy as np
from scipy import signal

from stroke_bci_mvp.types import EpochDataset


def _band_limited_noise(
    rng: np.random.Generator,
    n_times: int,
    sfreq: float,
    low: float,
    high: float,
    scale: float,
) -> np.ndarray:
    white = rng.normal(0, 1, n_times)
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, white) * scale


def make_synthetic_dataset(config: dict) -> EpochDataset:
    """Create EEG-like epochs with a motor-intention ERD pattern.

    Units are microvolts. Label 0 is rest; label 1 is affected-hand intention.
    Intent epochs suppress mu/beta rhythm over C3/Cz/C4 during the task window.
    """

    rng = np.random.default_rng(config.get("random_state", 7))
    n_subjects = int(config.get("n_subjects", 6))
    trials_per_subject = int(config.get("trials_per_subject", 60))
    sfreq = float(config.get("sfreq", 160))
    epoch_seconds = float(config.get("epoch_seconds", 4.0))
    ch_names = list(config.get("channels", ["C3", "Cz", "C4"]))
    n_channels = len(ch_names)
    n_times = int(round(epoch_seconds * sfreq))
    t = np.arange(n_times) / sfreq

    motor_idx = [i for i, ch in enumerate(ch_names) if ch in {"C3", "Cz", "C4"}]
    if not motor_idx:
        motor_idx = list(range(min(3, n_channels)))

    epochs: list[np.ndarray] = []
    labels: list[int] = []
    subject_ids: list[str] = []

    for subject in range(n_subjects):
        subject_gain = rng.uniform(0.85, 1.25)
        erd_strength = rng.uniform(0.35, 0.65)
        for trial in range(trials_per_subject):
            label = trial % 2
            epoch = np.zeros((n_channels, n_times), dtype=float)

            for ch in range(n_channels):
                alpha = _band_limited_noise(rng, n_times, sfreq, 8, 13, scale=8.0)
                beta = _band_limited_noise(rng, n_times, sfreq, 13, 30, scale=5.0)
                broadband = rng.normal(0, 2.0, n_times)
                slow_drift = 4.0 * np.sin(2 * np.pi * rng.uniform(0.1, 0.4) * t)
                epoch[ch] = subject_gain * (alpha + beta) + broadband + slow_drift

            if label == 1:
                task_mask = (t >= 0.8) & (t <= 3.2)
                taper = np.ones(n_times)
                taper[task_mask] = 1.0 - erd_strength
                for ch in motor_idx:
                    mu_beta = _band_limited_noise(rng, n_times, sfreq, 8, 30, scale=8.0)
                    epoch[ch] = epoch[ch] * taper + mu_beta * (taper - 1.0)

            # Inject occasional artifacts so the quality gate has real work to do.
            if rng.random() < 0.08:
                bad_ch = rng.integers(0, n_channels)
                start = rng.integers(0, max(1, n_times - int(0.25 * sfreq)))
                stop = min(n_times, start + int(0.25 * sfreq))
                epoch[bad_ch, start:stop] += rng.normal(180, 25, stop - start)

            epochs.append(epoch)
            labels.append(label)
            subject_ids.append(f"S{subject + 1:02d}")

    return EpochDataset(
        X=np.asarray(epochs),
        y=np.asarray(labels, dtype=int),
        sfreq=sfreq,
        ch_names=ch_names,
        subject_ids=np.asarray(subject_ids),
        label_names={0: "rest", 1: "motor_intention"},
    )

