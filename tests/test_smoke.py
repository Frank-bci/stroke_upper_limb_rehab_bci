from __future__ import annotations

import numpy as np

from stroke_bci_mvp.datasets.synthetic import make_synthetic_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
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
