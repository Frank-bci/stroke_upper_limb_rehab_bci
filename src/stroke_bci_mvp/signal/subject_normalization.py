from __future__ import annotations

import numpy as np


def apply_subject_normalization(X: np.ndarray, subject_ids: np.ndarray, config: dict) -> np.ndarray:
    """Apply optional unsupervised per-subject/session signal normalization."""

    norm_cfg = config.get("preprocessing", {}).get("subject_normalization", {})
    if not norm_cfg or not bool(norm_cfg.get("enabled", False)):
        return X

    method = str(norm_cfg.get("method", "per_subject_channel_standardize")).lower()
    if method not in {"per_subject_channel_standardize", "subject_channel_standardize"}:
        raise ValueError(f"Unsupported subject_normalization method: {method}")

    eps = float(norm_cfg.get("eps", 1e-6))
    X_norm = np.asarray(X, dtype=float).copy()
    subject_ids = np.asarray(subject_ids).astype(str)
    for subject_id in np.unique(subject_ids):
        mask = subject_ids == subject_id
        subject_X = X_norm[mask]
        mean = subject_X.mean(axis=(0, 2), keepdims=True)
        std = subject_X.std(axis=(0, 2), keepdims=True)
        X_norm[mask] = (subject_X - mean) / np.maximum(std, eps)
    return X_norm
