from __future__ import annotations

import numpy as np
from mne.decoding import CSP
from scipy import signal
from sklearn.base import BaseEstimator, TransformerMixin


class FBCSPTransformer(BaseEstimator, TransformerMixin):
    """Filter-bank CSP transformer for epochs shaped as n_epochs x n_channels x n_times."""

    def __init__(
        self,
        sfreq: float,
        bands: list[tuple[float, float]],
        n_components: int = 4,
        reg: str | float | None = "ledoit_wolf",
    ):
        self.sfreq = sfreq
        self.bands = bands
        self.n_components = n_components
        self.reg = reg

    def fit(self, X: np.ndarray, y: np.ndarray):
        self._csps = []
        for band in self.bands:
            Xb = self._filter(X, band)
            csp = CSP(n_components=self.n_components, reg=self.reg, log=True, norm_trace=False)
            csp.fit(Xb, y)
            self._csps.append(csp)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        features = []
        for band, csp in zip(self.bands, self._csps):
            Xb = self._filter(X, band)
            features.append(csp.transform(Xb))
        return np.concatenate(features, axis=1)

    def _filter(self, X: np.ndarray, band: tuple[float, float]) -> np.ndarray:
        low, high = band
        sos = signal.butter(4, [low, high], btype="bandpass", fs=self.sfreq, output="sos")
        return signal.sosfiltfilt(sos, X, axis=-1)

