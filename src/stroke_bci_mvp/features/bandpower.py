from __future__ import annotations

import numpy as np
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


class BandpowerTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, sfreq: float, bands: list[tuple[float, float]]):
        self.sfreq = sfreq
        self.bands = bands

    def fit(self, X: np.ndarray, y: np.ndarray | None = None):
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        freqs, psd = signal.welch(X, fs=self.sfreq, nperseg=min(X.shape[-1], int(self.sfreq)), axis=-1)
        features = []
        for low, high in self.bands:
            mask = (freqs >= low) & (freqs <= high)
            power = np.trapezoid(psd[..., mask], freqs[mask], axis=-1)
            features.append(np.log(power + 1e-12))
        return np.concatenate(features, axis=1)
