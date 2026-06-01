from __future__ import annotations

import numpy as np

from stroke_bci_mvp.datasets.synthetic import make_synthetic_dataset
from stroke_bci_mvp.datasets.physionet import _label_for_annotation, _label_names
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.online.calibration import calibrate_trigger_threshold
from stroke_bci_mvp.online.trigger_decision import TriggerDecision, TriggerParams
from stroke_bci_mvp.signal.quality import assess_epoch_quality, filter_valid_epochs


def _small_config() -> dict:
    return {
        "random_state": 7,
        "n_subjects": 3,
        "trials_per_subject": 4,
        "sfreq": 80,
        "epoch_seconds": 2.0,
        "channels": ["C3", "Cz", "C4"],
    }


def test_synthetic_dataset_shape_and_labels() -> None:
    dataset = make_synthetic_dataset(_small_config())

    assert dataset.X.shape == (12, 3, 160)
    assert set(dataset.y.tolist()) == {0, 1}
    assert len(np.unique(dataset.subject_ids)) == 3


def test_physionet_task_label_mapping_supports_left_vs_right_mi() -> None:
    assert _label_for_annotation("T0", "rest_vs_mi") == 0
    assert _label_for_annotation("T1", "rest_vs_mi") == 1
    assert _label_for_annotation("T2", "rest_vs_mi") == 1
    assert _label_for_annotation("T0", "left_vs_right_mi") is None
    assert _label_for_annotation("T1", "left_vs_right_mi") == 0
    assert _label_for_annotation("T2", "left_vs_right_mi") == 1
    assert _label_names("left_vs_right_mi") == {0: "left_hand_mi", 1: "right_hand_mi"}


def test_quality_gate_keeps_clean_synthetic_epochs() -> None:
    dataset = make_synthetic_dataset(_small_config())
    quality_cfg = {
        "min_score": 70,
        "max_abs_uv": 250,
        "min_channel_std_uv": 0.05,
        "max_bad_channel_ratio": 0.35,
        "max_line_noise_ratio": 0.35,
    }

    result = assess_epoch_quality(dataset.X[0], dataset.sfreq, dataset.ch_names, quality_cfg)
    X_valid, y_valid, results = filter_valid_epochs(
        dataset.X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        quality_cfg,
    )

    assert result.score >= 70
    assert len(results) == len(dataset.y)
    assert len(X_valid) == len(y_valid)


def test_subject_split_has_no_subject_overlap() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "dataset": {"name": "physionet", "random_state": 7},
        "model": {"test_size": 0.34, "split_strategy": "subject"},
    }

    split = make_train_test_split(dataset.y, dataset.subject_ids, config)
    train_subjects = set(dataset.subject_ids[split.train_idx].tolist())
    test_subjects = set(dataset.subject_ids[split.test_idx].tolist())

    assert split.strategy == "subject"
    assert train_subjects.isdisjoint(test_subjects)
    assert test_subjects == set(split.test_subject_ids)


def test_online_windows_follow_task_timing() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "online": {
            "window_seconds": 0.5,
            "step_seconds": 0.25,
            "task_start_seconds": 0.5,
            "task_end_seconds": 1.5,
        }
    }

    windows = make_online_windows(dataset.X[:2], dataset.y[:2], dataset.sfreq, config, epoch_indices=np.array([4, 5]))

    assert windows.X.shape[1:] == (3, 40)
    assert set(windows.source_epoch_idx.tolist()) == {4, 5}
    assert windows.window_center_seconds.min() >= 0.5
    assert windows.window_center_seconds.max() <= 1.5


def test_riemannian_model_fits_online_windows() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "model": {
            "type": "riemannian_logreg",
            "covariance_estimator": "oas",
            "riemannian_metric": "riemann",
            "class_weight": "balanced",
            "training_mode": "online_windows",
        },
        "online": {
            "window_seconds": 0.5,
            "step_seconds": 0.25,
            "task_start_seconds": 0.5,
            "task_end_seconds": 1.5,
        },
    }
    windows = make_online_windows(dataset.X, dataset.y, dataset.sfreq, config)

    model = build_model(config, dataset.sfreq)
    model.fit(windows.X, windows.y)
    y_score = model.predict_proba(windows.X[:4])

    assert y_score.shape == (4, 2)


def test_riemannian_model_supports_epoch_standardization() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "model": {
            "type": "riemannian_logreg",
            "covariance_estimator": "oas",
            "riemannian_metric": "riemann",
            "class_weight": "balanced",
            "normalization": "epoch_standardize",
        },
        "online": {
            "window_seconds": 0.5,
            "step_seconds": 0.25,
            "task_start_seconds": 0.5,
            "task_end_seconds": 1.5,
        },
    }
    windows = make_online_windows(dataset.X, dataset.y, dataset.sfreq, config)

    model = build_model(config, dataset.sfreq)
    model.fit(windows.X, windows.y)

    assert model.predict(windows.X[:4]).shape == (4,)


def test_riemannian_model_supports_train_channel_standardization() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "model": {
            "type": "riemannian_logreg",
            "covariance_estimator": "oas",
            "riemannian_metric": "riemann",
            "class_weight": "balanced",
            "normalization": "train_channel_standardize",
        },
        "online": {
            "window_seconds": 0.5,
            "step_seconds": 0.25,
            "task_start_seconds": 0.5,
            "task_end_seconds": 1.5,
        },
    }
    windows = make_online_windows(dataset.X, dataset.y, dataset.sfreq, config)

    model = build_model(config, dataset.sfreq)
    model.fit(windows.X, windows.y)

    assert model.predict_proba(windows.X[:4]).shape == (4, 2)


def test_fbcsp_model_supports_train_channel_standardization() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "model": {
            "type": "fbcsp_lda",
            "normalization": "train_channel_standardize",
            "bands": [(8, 12), (12, 20)],
            "csp_components": 1,
        }
    }

    model = build_model(config, dataset.sfreq)
    model.fit(dataset.X, dataset.y)

    assert model.predict(dataset.X[:4]).shape == (4,)


def test_model_supports_channel_selection_by_name() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "model": {
            "type": "riemannian_logreg",
            "covariance_estimator": "oas",
            "riemannian_metric": "riemann",
            "normalization": "train_channel_standardize",
            "channel_selection": {
                "enabled": True,
                "exclude_channels": ["Cz"],
                "min_channels": 2,
            },
        }
    }

    model = build_model(config, dataset.sfreq, ch_names=dataset.ch_names)
    model.fit(dataset.X, dataset.y)

    assert model.predict_proba(dataset.X[:4]).shape == (4, 2)


def test_model_supports_supervised_top_k_channel_selection() -> None:
    dataset = make_synthetic_dataset(_small_config())
    config = {
        "model": {
            "type": "riemannian_logreg",
            "covariance_estimator": "oas",
            "riemannian_metric": "riemann",
            "normalization": "train_channel_standardize",
            "channel_selection": {
                "enabled": True,
                "method": "supervised_top_k",
                "top_k": 2,
                "min_channels": 2,
            },
        }
    }

    model = build_model(config, dataset.sfreq, ch_names=dataset.ch_names)
    model.fit(dataset.X, dataset.y)

    selector = model.named_steps["channel_selector"]
    assert len(selector.selected_indices_) == 2
    assert model.predict_proba(dataset.X[:4]).shape == (4, 2)


def test_trigger_decision_requires_consecutive_windows() -> None:
    decision = TriggerDecision(
        TriggerParams(
            threshold=0.8,
            consecutive_windows=2,
            refractory_seconds=3.0,
            min_quality_score=70,
            task_start_seconds=0.0,
            task_end_seconds=2.0,
        )
    )

    first = decision.update(time_seconds=0.5, probability=0.9, quality_score=100)
    second = decision.update(time_seconds=0.75, probability=0.9, quality_score=100)

    assert first == (False, "waiting_for_confirmation")
    assert second == (True, "trigger_bu100_assist_open_hand")


def test_threshold_calibration_selects_feasible_candidate() -> None:
    class ConstantModel:
        def predict_proba(self, X):
            return np.tile([[0.2, 0.8]], (len(X), 1))

    dataset = make_synthetic_dataset(_small_config())
    config = {
        "online": {
            "window_seconds": 0.5,
            "step_seconds": 0.25,
            "trigger_threshold": 0.5,
            "consecutive_windows": 1,
            "refractory_seconds": 3.0,
            "task_start_seconds": 0.5,
            "task_end_seconds": 1.5,
        },
        "quality": {
            "min_score": 70,
            "max_abs_uv": 250,
            "min_channel_std_uv": 0.05,
            "max_bad_channel_ratio": 0.35,
            "max_line_noise_ratio": 0.35,
        },
    }

    result = calibrate_trigger_threshold(
        ConstantModel(),
        dataset.X[:4],
        dataset.y[:4],
        dataset.sfreq,
        dataset.ch_names,
        config,
        thresholds=[0.5, 0.85],
        max_false_trigger_rate=0.0,
    )

    assert result["selected_threshold"] == 0.85
    assert result["selected_metrics"]["false_trigger_rate"] == 0.0
