from __future__ import annotations

import numpy as np
from scipy import signal


def notch_epochs(X: np.ndarray, sfreq: float, notch_hz: float | None) -> np.ndarray:
    if not notch_hz:
        return X
    b, a = signal.iirnotch(w0=notch_hz, Q=30, fs=sfreq)
    return signal.filtfilt(b, a, X, axis=-1)


def bandpass_epochs(X: np.ndarray, sfreq: float, low: float, high: float) -> np.ndarray:
    sos = signal.butter(4, [low, high], btype="bandpass", fs=sfreq, output="sos")
    return signal.sosfiltfilt(sos, X, axis=-1)

