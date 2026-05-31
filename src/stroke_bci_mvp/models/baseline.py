from __future__ import annotations

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from scipy import signal

from stroke_bci_mvp.features import BandpowerTransformer, FBCSPTransformer


def build_model(config: dict, sfreq: float):
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "fbcsp_lda")
    bands = [tuple(map(float, band)) for band in model_cfg.get("bands", [(8, 12), (12, 16), (16, 24), (24, 30)])]

    if model_type == "fbcsp_lda":
        return _maybe_calibrate(
            Pipeline(
                steps=[
                    (
                        "fbcsp",
                        FBCSPTransformer(
                            sfreq=sfreq,
                            bands=bands,
                            n_components=int(model_cfg.get("csp_components", 4)),
                        ),
                    ),
                    ("scaler", StandardScaler()),
                    ("clf", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")),
                ]
            ),
            model_cfg,
        )

    if model_type == "fbcsp_logreg":
        return _maybe_calibrate(
            Pipeline(
                steps=[
                    (
                        "fbcsp",
                        FBCSPTransformer(
                            sfreq=sfreq,
                            bands=bands,
                            n_components=int(model_cfg.get("csp_components", 4)),
                        ),
                    ),
                    ("scaler", StandardScaler()),
                    (
                        "clf",
                        LogisticRegression(
                            max_iter=int(model_cfg.get("max_iter", 1000)),
                            class_weight=model_cfg.get("class_weight", "balanced"),
                        ),
                    ),
                ]
            ),
            model_cfg,
        )

    if model_type in {"riemannian_logreg", "riemannian_lda"}:
        return _maybe_calibrate(_build_riemannian_model(config, sfreq), model_cfg)

    if model_type == "bandpower_logreg":
        return _maybe_calibrate(
            Pipeline(
                steps=[
                    ("bandpower", BandpowerTransformer(sfreq=sfreq, bands=bands)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
            model_cfg,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def _build_riemannian_model(config: dict, sfreq: float) -> Pipeline:
    from pyriemann.estimation import Covariances
    from pyriemann.tangentspace import TangentSpace

    model_cfg = config["model"]
    model_type = model_cfg.get("type", "riemannian_logreg")
    covariance_estimator = str(model_cfg.get("covariance_estimator", "oas"))
    metric = str(model_cfg.get("riemannian_metric", "riemann"))
    bandpass_hz = model_cfg.get("bandpass_hz", config.get("preprocessing", {}).get("bandpass_hz", [8, 30]))

    if model_type == "riemannian_lda":
        classifier = LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
    else:
        classifier = LogisticRegression(
            max_iter=int(model_cfg.get("max_iter", 1000)),
            class_weight=model_cfg.get("class_weight", "balanced"),
        )

    return Pipeline(
        steps=_normalization_steps(model_cfg)
        + [
            ("bandpass", BandpassTransformer(sfreq=sfreq, band=tuple(map(float, bandpass_hz)))),
            ("covariances", Covariances(estimator=covariance_estimator)),
            ("tangent_space", TangentSpace(metric=metric)),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def _normalization_steps(model_cfg: dict) -> list[tuple[str, BaseEstimator]]:
    normalization = str(model_cfg.get("normalization", "none")).lower()
    if normalization in {"none", ""}:
        return []
    if normalization in {"epoch_zscore", "epoch_standardize"}:
        return [("epoch_standardize", EpochStandardizer())]
    raise ValueError(f"Unsupported model normalization: {normalization}")


def _maybe_calibrate(model, model_cfg: dict):
    calibration_cfg = model_cfg.get("probability_calibration", {})
    if not calibration_cfg or not bool(calibration_cfg.get("enabled", False)):
        return model

    return CalibratedClassifierCV(
        estimator=model,
        method=str(calibration_cfg.get("method", "sigmoid")),
        cv=int(calibration_cfg.get("cv", 3)),
    )


class BandpassTransformer(BaseEstimator, TransformerMixin):
    """Bandpass epochs while keeping the n_epochs x n_channels x n_times shape."""

    def __init__(self, sfreq: float, band: tuple[float, float]):
        self.sfreq = sfreq
        self.band = band

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        low, high = self.band
        sos = signal.butter(4, [low, high], btype="bandpass", fs=self.sfreq, output="sos")
        return signal.sosfiltfilt(sos, X, axis=-1)


class EpochStandardizer(BaseEstimator, TransformerMixin):
    """Remove per-epoch/channel DC drift and scale differences."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        mean = X.mean(axis=-1, keepdims=True)
        std = X.std(axis=-1, keepdims=True)
        return (X - mean) / np.maximum(std, self.eps)
