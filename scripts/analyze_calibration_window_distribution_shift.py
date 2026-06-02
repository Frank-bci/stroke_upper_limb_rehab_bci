from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.signal import apply_subject_normalization, filter_valid_epochs, notch_epochs

from evaluate_calibration_window_adaptation import _split_subject_calibration


def analyze_distribution_shift(
    config_path: str,
    repeated_json: str,
    min_evaluation_false_trigger_rate: float = 0.10,
    output_path: str | None = None,
) -> dict:
    repeated = json.loads(Path(repeated_json).read_text(encoding="utf-8"))
    base_config = load_config(config_path)

    dataset = load_dataset(base_config)
    X = notch_epochs(dataset.X, dataset.sfreq, base_config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, quality_results = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        base_config["quality"],
    )
    valid_subject_ids = dataset.subject_ids[np.asarray([result.valid for result in quality_results], dtype=bool)]
    X_valid = apply_subject_normalization(X_valid, valid_subject_ids, base_config)

    runs = []
    for run in repeated["runs"]:
        seed = int(run["seed"])
        config = copy.deepcopy(base_config)
        config["dataset"]["random_state"] = seed
        split = make_train_test_split(y_valid, valid_subject_ids, config)
        train_data = _training_samples(X_valid[split.train_idx], y_valid[split.train_idx], dataset.sfreq, config)
        model = build_model(config, dataset.sfreq, ch_names=dataset.ch_names)
        model.fit(train_data["X"], train_data["y"])

        subject_rows = []
        for subject in run["subjects"]:
            evaluation_false = float(subject.get("evaluation_metrics", {}).get("false_trigger_rate", 0.0))
            if evaluation_false < min_evaluation_false_trigger_rate:
                continue
            subject_id = str(subject["subject_id"])
            subject_mask = valid_subject_ids[split.test_idx].astype(str) == subject_id
            if not np.any(subject_mask):
                continue
            subject_epoch_idx = split.test_idx[subject_mask]
            calibration_idx, evaluation_idx = _split_subject_calibration(
                subject_epoch_idx,
                y_valid,
                calibration_trials_per_class=int(repeated["metadata"].get("calibration_trials_per_class", 5)),
            )
            selected_threshold = float(subject.get("selected_threshold", repeated["metadata"]["thresholds"][0]))
            subject_rows.append(
                _subject_shift_row(
                    subject=subject,
                    selected_threshold=selected_threshold,
                    model=model,
                    X_valid=X_valid,
                    y_valid=y_valid,
                    calibration_idx=calibration_idx,
                    evaluation_idx=evaluation_idx,
                    sfreq=dataset.sfreq,
                    config=config,
                )
            )
        runs.append(
            {
                "seed": seed,
                "test_subject_ids": run["test_subject_ids"],
                "high_false_subjects": subject_rows,
            }
        )

    result = {
        "metadata": {
            "dataset": base_config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "repeated_json": str(Path(repeated_json)),
            "min_evaluation_false_trigger_rate": float(min_evaluation_false_trigger_rate),
        },
        "summary": _summary(runs),
        "runs": runs,
    }

    if output_path is None:
        output_path = str(Path(repeated_json).with_name("calibration_window_distribution_shift.json"))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _subject_shift_row(
    subject: dict[str, Any],
    selected_threshold: float,
    model,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    calibration_idx: np.ndarray,
    evaluation_idx: np.ndarray,
    sfreq: float,
    config: dict,
) -> dict:
    calibration = _window_probability_summary(model, X_valid[calibration_idx], y_valid[calibration_idx], sfreq, config, selected_threshold)
    evaluation = _window_probability_summary(model, X_valid[evaluation_idx], y_valid[evaluation_idx], sfreq, config, selected_threshold)
    return {
        "subject_id": subject["subject_id"],
        "selected_threshold": selected_threshold,
        "calibration_epochs": int(len(calibration_idx)),
        "evaluation_epochs": int(len(evaluation_idx)),
        "evaluation_trigger_rate": float(subject.get("evaluation_metrics", {}).get("trigger_rate", 0.0)),
        "evaluation_false_trigger_rate": float(subject.get("evaluation_metrics", {}).get("false_trigger_rate", 0.0)),
        "calibration_selected_metrics": subject.get("calibration_selected_metrics", {}),
        "calibration_offline": subject.get("calibration_offline", {}),
        "probability_shift": {
            "calibration": calibration,
            "evaluation": evaluation,
            "rest_p95_delta": _delta(evaluation["rest"]["p95"], calibration["rest"]["p95"]),
            "rest_mean_delta": _delta(evaluation["rest"]["mean"], calibration["rest"]["mean"]),
            "intention_mean_delta": _delta(evaluation["motor_intention"]["mean"], calibration["motor_intention"]["mean"]),
        },
        "interpretation": _interpret_shift(calibration, evaluation, selected_threshold),
    }


def _window_probability_summary(model, X: np.ndarray, y: np.ndarray, sfreq: float, config: dict, threshold: float) -> dict:
    windows = make_online_windows(X, y, sfreq, config)
    probabilities = model.predict_proba(windows.X)[:, 1]
    rest = probabilities[windows.y == 0]
    intention = probabilities[windows.y == 1]
    return {
        "rest": _summary_values(rest),
        "motor_intention": _summary_values(intention),
        "rest_windows_over_threshold_rate": float(np.mean(rest >= threshold)) if len(rest) else 0.0,
        "intention_windows_over_threshold_rate": float(np.mean(intention >= threshold)) if len(intention) else 0.0,
    }


def _training_samples(X, y, sfreq: float, config: dict) -> dict:
    training_mode = str(config.get("model", {}).get("training_mode", "epoch")).lower()
    if training_mode == "online_windows":
        windows = make_online_windows(X, y, sfreq, config)
        return {"X": windows.X, "y": windows.y}
    if training_mode == "epoch":
        return {"X": X, "y": y}
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def _summary_values(values: np.ndarray) -> dict:
    if len(values) == 0:
        return {"count": 0, "mean": None, "std": None, "p50": None, "p90": None, "p95": None, "max": None}
    return {
        "count": int(len(values)),
        "mean": float(np.mean(values)),
        "std": float(np.std(values, ddof=0)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "max": float(np.max(values)),
    }


def _delta(after: float | None, before: float | None) -> float | None:
    if after is None or before is None:
        return None
    return float(after - before)


def _interpret_shift(calibration: dict, evaluation: dict, threshold: float) -> list[str]:
    messages = []
    cal_rest_over = calibration["rest_windows_over_threshold_rate"]
    eval_rest_over = evaluation["rest_windows_over_threshold_rate"]
    if cal_rest_over == 0 and eval_rest_over > 0:
        messages.append("Calibration rest windows did not expose later rest risk.")
    if evaluation["rest"]["p95"] is not None and evaluation["rest"]["p95"] >= threshold:
        messages.append("Evaluation rest p95 reaches or exceeds selected threshold.")
    if evaluation["motor_intention"]["mean"] is not None and evaluation["rest"]["mean"] is not None:
        if evaluation["rest"]["mean"] >= evaluation["motor_intention"]["mean"]:
            messages.append("Evaluation rest mean overlaps or exceeds motor-intention mean.")
    return messages


def _summary(runs: list[dict]) -> dict:
    high_false_counts = [len(run["high_false_subjects"]) for run in runs]
    rest_p95_deltas = [
        row["probability_shift"]["rest_p95_delta"]
        for run in runs
        for row in run["high_false_subjects"]
        if row["probability_shift"]["rest_p95_delta"] is not None
    ]
    rest_over_deltas = [
        row["probability_shift"]["evaluation"]["rest_windows_over_threshold_rate"]
        - row["probability_shift"]["calibration"]["rest_windows_over_threshold_rate"]
        for run in runs
        for row in run["high_false_subjects"]
    ]
    return {
        "high_false_subject_count": _mean_std(high_false_counts),
        "rest_p95_delta": _mean_std(rest_p95_deltas),
        "rest_windows_over_threshold_rate_delta": _mean_std(rest_over_deltas),
    }


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0}
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze calibration/evaluation probability shift for high-false-trigger subjects.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_full_riemannian_train_channel_standardized.yaml")
    parser.add_argument("--repeated-json", required=True)
    parser.add_argument("--min-evaluation-false-trigger-rate", type=float, default=0.10)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = analyze_distribution_shift(
        config_path=args.config,
        repeated_json=args.repeated_json,
        min_evaluation_false_trigger_rate=args.min_evaluation_false_trigger_rate,
        output_path=args.output,
    )
    print(json.dumps({"metadata": result["metadata"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
