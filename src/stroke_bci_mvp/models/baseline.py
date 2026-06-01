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


def build_model(config: dict, sfreq: float, ch_names: list[str] | None = None):
    model_cfg = config["model"]
    model_type = model_cfg.get("type", "fbcsp_lda")
    bands = [tuple(map(float, band)) for band in model_cfg.get("bands", [(8, 12), (12, 16), (16, 24), (24, 30)])]

    if model_type == "fbcsp_lda":
        return _maybe_calibrate(
            Pipeline(
                steps=_channel_selection_steps(model_cfg, ch_names)
                + _normalization_steps(model_cfg)
                + [
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
                steps=_channel_selection_steps(model_cfg, ch_names)
                + _normalization_steps(model_cfg)
                + [
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
        return _maybe_calibrate(_build_riemannian_model(config, sfreq, ch_names), model_cfg)

    if model_type == "bandpower_logreg":
        return _maybe_calibrate(
            Pipeline(
                steps=_channel_selection_steps(model_cfg, ch_names)
                + _normalization_steps(model_cfg)
                + [
                    ("bandpower", BandpowerTransformer(sfreq=sfreq, bands=bands)),
                    ("scaler", StandardScaler()),
                    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
                ]
            ),
            model_cfg,
        )

    raise ValueError(f"Unsupported model type: {model_type}")


def _build_riemannian_model(config: dict, sfreq: float, ch_names: list[str] | None = None) -> Pipeline:
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
        steps=_channel_selection_steps(model_cfg, ch_names)
        + _normalization_steps(model_cfg)
        + [
            ("bandpass", BandpassTransformer(sfreq=sfreq, band=tuple(map(float, bandpass_hz)))),
            ("covariances", Covariances(estimator=covariance_estimator)),
            ("tangent_space", TangentSpace(metric=metric)),
            ("scaler", StandardScaler()),
            ("clf", classifier),
        ]
    )


def _channel_selection_steps(model_cfg: dict, ch_names: list[str] | None) -> list[tuple[str, BaseEstimator]]:
    selection_cfg = model_cfg.get("channel_selection", {})
    if not selection_cfg or not bool(selection_cfg.get("enabled", False)):
        return []

    if ch_names is None:
        raise ValueError("model.channel_selection requires dataset channel names.")

    method = str(selection_cfg.get("method", "fixed")).lower()
    include_channels = selection_cfg.get("include_channels")
    exclude_channels = set(map(str, selection_cfg.get("exclude_channels", [])))
    if include_channels:
        include_set = set(map(str, include_channels))
        selected = [idx for idx, name in enumerate(ch_names) if name in include_set and name not in exclude_channels]
    else:
        selected = [idx for idx, name in enumerate(ch_names) if name not in exclude_channels]

    min_channels = int(selection_cfg.get("min_channels", 2))
    if len(selected) < min_channels:
        raise ValueError(
            f"Channel selection kept {len(selected)} channels, fewer than min_channels={min_channels}."
        )

    if method in {"fixed", "manual"}:
        return [("channel_selector", ChannelSelector(indices=selected))]

    if method in {"supervised_top_k", "data_driven_top_k", "top_k"}:
        top_k = int(selection_cfg.get("top_k", len(selected)))
        if top_k < min_channels:
            raise ValueError(f"top_k={top_k} is fewer than min_channels={min_channels}.")
        if top_k > len(selected):
            raise ValueError(f"top_k={top_k} exceeds available candidate channels={len(selected)}.")
        return [("channel_selector", SupervisedTopKChannelSelector(candidate_indices=selected, top_k=top_k))]

    raise ValueError(f"Unsupported channel_selection method: {method}")


def _normalization_steps(model_cfg: dict) -> list[tuple[str, BaseEstimator]]:
    normalization = str(model_cfg.get("normalization", "none")).lower()
    if normalization in {"none", ""}:
        return []
    if normalization in {"epoch_zscore", "epoch_standardize"}:
        return [("epoch_standardize", EpochStandardizer())]
    if normalization in {"train_channel_zscore", "train_channel_standardize", "channel_standardize"}:
        return [("train_channel_standardize", TrainChannelStandardizer())]
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


class ChannelSelector(BaseEstimator, TransformerMixin):
    """Keep a stable channel subset before downstream EEG feature extraction."""

    def __init__(self, indices: list[int]):
        self.indices = indices

    def fit(self, X, y=None):
        if not self.indices:
            raise ValueError("ChannelSelector requires at least one channel index.")
        max_idx = max(self.indices)
        if X.shape[1] <= max_idx:
            raise ValueError(f"Channel index {max_idx} is out of bounds for {X.shape[1]} channels.")
        return self

    def transform(self, X):
        return X[:, self.indices, :]


class SupervisedTopKChannelSelector(BaseEstimator, TransformerMixin):
    """Select channels by training-set class separation in log-variance space."""

    def __init__(self, candidate_indices: list[int], top_k: int, eps: float = 1e-12):
        self.candidate_indices = candidate_indices
        self.top_k = top_k
        self.eps = eps

    def fit(self, X, y):
        if y is None:
            raise ValueError("SupervisedTopKChannelSelector requires labels.")
        if not self.candidate_indices:
            raise ValueError("SupervisedTopKChannelSelector requires candidate channel indices.")
        if self.top_k <= 0:
            raise ValueError("top_k must be positive.")
        max_idx = max(self.candidate_indices)
        if X.shape[1] <= max_idx:
            raise ValueError(f"Channel index {max_idx} is out of bounds for {X.shape[1]} channels.")

        y = np.asarray(y)
        classes = np.unique(y)
        if len(classes) != 2:
            raise ValueError("SupervisedTopKChannelSelector currently supports binary labels only.")

        X_candidates = X[:, self.candidate_indices, :]
        log_variance = np.log(np.var(X_candidates, axis=-1) + self.eps)
        class0 = log_variance[y == classes[0]]
        class1 = log_variance[y == classes[1]]
        pooled_std = np.sqrt((np.var(class0, axis=0) + np.var(class1, axis=0)) / 2.0)
        scores = np.abs(np.mean(class1, axis=0) - np.mean(class0, axis=0)) / np.maximum(pooled_std, self.eps)

        order = np.argsort(scores)[::-1][: self.top_k]
        self.selected_indices_ = [int(self.candidate_indices[idx]) for idx in order]
        self.scores_ = scores
        return self

    def transform(self, X):
        return X[:, self.selected_indices_, :]


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


class TrainChannelStandardizer(BaseEstimator, TransformerMixin):
    """Standardize channels using statistics learned from the training split."""

    def __init__(self, eps: float = 1e-6):
        self.eps = eps

    def fit(self, X, y=None):
        self.mean_ = X.mean(axis=(0, 2), keepdims=True)
        self.std_ = X.std(axis=(0, 2), keepdims=True)
        return self

    def transform(self, X):
        return (X - self.mean_) / np.maximum(self.std_, self.eps)
