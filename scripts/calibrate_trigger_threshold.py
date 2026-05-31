from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.model_selection import GroupKFold, StratifiedKFold

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.online import calibrate_trigger_threshold, simulate_session
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def calibrate(config_path: str, thresholds: list[float] | None = None, max_false_trigger_rate: float | None = None) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
    calibration_cfg = config.get("calibration", {})
    thresholds = thresholds or [float(value) for value in calibration_cfg.get("thresholds", _default_thresholds())]
    max_false_trigger_rate = float(
        calibration_cfg.get("max_false_trigger_rate", 0.1)
        if max_false_trigger_rate is None
        else max_false_trigger_rate
    )

    bundle = joblib.load(outputs["model_path"])
    dataset = load_dataset(config)
    X = notch_epochs(dataset.X, dataset.sfreq, config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, quality_results = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        config["quality"],
    )
    valid_subject_ids = dataset.subject_ids[[result.valid for result in quality_results]]
    split = make_train_test_split(y_valid, valid_subject_ids, config)

    cv_result = _cross_validated_calibration(
        X=X_valid[split.train_idx],
        y=y_valid[split.train_idx],
        subject_ids=valid_subject_ids[split.train_idx],
        sfreq=dataset.sfreq,
        ch_names=dataset.ch_names,
        config=config,
        thresholds=thresholds,
        max_false_trigger_rate=max_false_trigger_rate,
    )

    train_split_result = calibrate_trigger_threshold(
        model=bundle["model"],
        X=X_valid[split.train_idx],
        y=y_valid[split.train_idx],
        sfreq=dataset.sfreq,
        ch_names=dataset.ch_names,
        config=config,
        thresholds=thresholds,
        max_false_trigger_rate=max_false_trigger_rate,
    )
    result = cv_result or train_split_result
    result["single_model_train_split_calibration"] = train_split_result
    heldout_config = json.loads(json.dumps(config))
    heldout_config["online"]["trigger_threshold"] = result["selected_threshold"]
    heldout_report = simulate_session(
        model=bundle["model"],
        X=X_valid[split.test_idx],
        y=y_valid[split.test_idx],
        sfreq=dataset.sfreq,
        ch_names=dataset.ch_names,
        config=heldout_config,
    )
    result["heldout_test_metrics_at_selected_threshold"] = {
        "n_trials": heldout_report["n_trials"],
        "true_intention_trials": heldout_report["true_intention_trials"],
        "rest_trials": heldout_report["rest_trials"],
        "trigger_rate": heldout_report["trigger_rate"],
        "false_trigger_rate": heldout_report["false_trigger_rate"],
        "mean_trigger_delay_seconds": heldout_report["mean_trigger_delay_seconds"],
        "triggered_intention_trials": heldout_report["triggered_intention_trials"],
        "triggered_rest_trials": heldout_report["triggered_rest_trials"],
    }
    result["metadata"] = {
        "dataset": config["dataset"]["name"],
        "config_path": str(Path(config_path)),
        "model_path": outputs["model_path"],
        "calibration_split": split.strategy,
        "calibration_epochs": int(len(split.train_idx)),
        "heldout_test_subject_ids": split.test_subject_ids,
        "training_mode": bundle.get("training_mode", config["model"].get("training_mode", "epoch")),
    }

    output_path = Path(outputs.get("threshold_calibration_json", Path(outputs["dir"]) / "threshold_calibration.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _cross_validated_calibration(
    X,
    y,
    subject_ids,
    sfreq: float,
    ch_names: list[str],
    config: dict,
    thresholds: list[float],
    max_false_trigger_rate: float,
) -> dict | None:
    calibration_cfg = config.get("calibration", {})
    cv_folds = int(calibration_cfg.get("cv_folds", 5))
    split_strategy = str(config.get("model", {}).get("split_strategy", "random")).lower()
    y = np.asarray(y)
    subject_ids = np.asarray(subject_ids)
    indices = np.arange(len(y))

    if split_strategy in {"subject", "group", "group_shuffle"}:
        unique_subjects = np.unique(subject_ids)
        if len(unique_subjects) < 2:
            return None
        splitter = GroupKFold(n_splits=min(cv_folds, len(unique_subjects)))
        splits = splitter.split(indices, y, subject_ids)
        calibration_strategy = "subject_cv"
    else:
        min_class_count = min(np.bincount(y))
        if min_class_count < 2:
            return None
        splitter = StratifiedKFold(
            n_splits=min(cv_folds, int(min_class_count)),
            shuffle=True,
            random_state=int(config.get("dataset", {}).get("random_state", 7)),
        )
        splits = splitter.split(indices, y)
        calibration_strategy = "stratified_cv"

    aggregate = {
        float(threshold): {
            "threshold": float(threshold),
            "true_intention_trials": 0,
            "rest_trials": 0,
            "triggered_intention_trials": 0,
            "triggered_rest_trials": 0,
            "delay_sum": 0.0,
            "delay_count": 0,
        }
        for threshold in thresholds
    }
    fold_reports = []

    for fold_idx, (train_idx, calib_idx) in enumerate(splits, start=1):
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[calib_idx])) < 2:
            fold_reports.append({"fold": fold_idx, "skipped": True, "reason": "single_class_fold"})
            continue

        X_train_model, y_train_model = _model_training_samples(X[train_idx], y[train_idx], sfreq, config)
        model = build_model(config, sfreq)
        model.fit(X_train_model, y_train_model)

        fold_rows = []
        for threshold in thresholds:
            run_config = json.loads(json.dumps(config))
            run_config["online"]["trigger_threshold"] = float(threshold)
            report = simulate_session(model, X[calib_idx], y[calib_idx], sfreq, ch_names, run_config)
            row = aggregate[float(threshold)]
            row["true_intention_trials"] += int(report["true_intention_trials"])
            row["rest_trials"] += int(report["rest_trials"])
            row["triggered_intention_trials"] += int(report["triggered_intention_trials"])
            row["triggered_rest_trials"] += int(report["triggered_rest_trials"])
            if report["mean_trigger_delay_seconds"] is not None:
                row["delay_sum"] += float(report["mean_trigger_delay_seconds"]) * int(report["triggered_intention_trials"])
                row["delay_count"] += int(report["triggered_intention_trials"])
            fold_rows.append(
                {
                    "threshold": float(threshold),
                    "trigger_rate": float(report["trigger_rate"]),
                    "false_trigger_rate": float(report["false_trigger_rate"]),
                }
            )

        fold_reports.append(
            {
                "fold": fold_idx,
                "skipped": False,
                "calibration_subject_ids": sorted(map(str, np.unique(subject_ids[calib_idx]))),
                "train_epochs": int(len(train_idx)),
                "calibration_epochs": int(len(calib_idx)),
                "candidates": fold_rows,
            }
        )

    candidates = []
    for row in aggregate.values():
        delay_count = int(row.pop("delay_count"))
        delay_sum = float(row.pop("delay_sum"))
        row["trigger_rate"] = row["triggered_intention_trials"] / max(1, row["true_intention_trials"])
        row["false_trigger_rate"] = row["triggered_rest_trials"] / max(1, row["rest_trials"])
        row["mean_trigger_delay_seconds"] = None if delay_count == 0 else delay_sum / delay_count
        candidates.append(row)
    candidates = sorted(candidates, key=lambda row: row["threshold"])
    selected, reason = _select_candidate(candidates, max_false_trigger_rate)
    return {
        "selected_threshold": selected["threshold"],
        "selection_reason": reason,
        "calibration_strategy": calibration_strategy,
        "max_false_trigger_rate": float(max_false_trigger_rate),
        "selected_metrics": selected,
        "candidates": candidates,
        "folds": fold_reports,
    }


def _model_training_samples(X, y, sfreq: float, config: dict):
    training_mode = str(config.get("model", {}).get("training_mode", "epoch")).lower()
    if training_mode == "online_windows":
        windows = make_online_windows(X, y, sfreq, config)
        return windows.X, windows.y
    if training_mode == "epoch":
        return X, y
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def _select_candidate(candidates: list[dict], max_false_trigger_rate: float) -> tuple[dict, str]:
    feasible = [row for row in candidates if row["false_trigger_rate"] <= max_false_trigger_rate]
    if feasible:
        return max(
            feasible,
            key=lambda row: (
                row["trigger_rate"],
                _delay_score(row["mean_trigger_delay_seconds"]),
                row["threshold"],
            ),
        ), "max_cv_trigger_rate_under_false_trigger_constraint"
    return max(
        candidates,
        key=lambda row: (
            -row["false_trigger_rate"],
            row["trigger_rate"],
            _delay_score(row["mean_trigger_delay_seconds"]),
            row["threshold"],
        ),
    ), "no_cv_threshold_met_false_trigger_constraint"


def _delay_score(delay: float | None) -> float:
    if delay is None:
        return float("-inf")
    return -float(delay)


def _default_thresholds() -> list[float]:
    return [round(value, 2) for value in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate BU100 trigger threshold on the training split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float)
    parser.add_argument("--max-false-trigger-rate", type=float)
    args = parser.parse_args()

    result = calibrate(args.config, args.thresholds, args.max_false_trigger_rate)
    compact = {
        key: value
        for key, value in result.items()
        if key not in {"candidates", "folds", "single_model_train_split_calibration"}
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
