from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.online import simulate_session
from stroke_bci_mvp.signal import apply_subject_normalization, filter_valid_epochs, notch_epochs


def evaluate_calibration_window_adaptation(
    config_path: str,
    thresholds: list[float],
    calibration_trials_per_class: int = 5,
    output_path: str | None = None,
    max_false_trigger_rate: float | None = None,
    min_trigger_rate: float = 0.05,
    min_auc: float = 0.60,
) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
    max_false_trigger_rate = (
        float(config.get("calibration", {}).get("max_false_trigger_rate", 0.1))
        if max_false_trigger_rate is None
        else float(max_false_trigger_rate)
    )

    dataset = load_dataset(config)
    X = notch_epochs(dataset.X, dataset.sfreq, config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, quality_results = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        config["quality"],
    )
    valid_subject_ids = dataset.subject_ids[np.asarray([result.valid for result in quality_results], dtype=bool)]
    X_valid = apply_subject_normalization(X_valid, valid_subject_ids, config)

    split = make_train_test_split(y_valid, valid_subject_ids, config)
    train_data = _training_samples(X_valid[split.train_idx], y_valid[split.train_idx], dataset.sfreq, config)
    model = build_model(config, dataset.sfreq, ch_names=dataset.ch_names)
    model.fit(train_data["X"], train_data["y"])

    subject_rows = []
    aggregate = _empty_aggregate()
    for subject_id in sorted(map(str, np.unique(valid_subject_ids[split.test_idx]))):
        subject_mask = valid_subject_ids[split.test_idx].astype(str) == subject_id
        subject_epoch_idx = split.test_idx[subject_mask]
        calibration_idx, evaluation_idx = _split_subject_calibration(
            subject_epoch_idx,
            y_valid,
            calibration_trials_per_class=calibration_trials_per_class,
        )
        if len(calibration_idx) == 0 or len(evaluation_idx) == 0 or len(np.unique(y_valid[evaluation_idx])) < 2:
            subject_rows.append(
                {
                    "subject_id": subject_id,
                    "status": "skipped_insufficient_calibration_or_evaluation",
                    "calibration_epochs": int(len(calibration_idx)),
                    "evaluation_epochs": int(len(evaluation_idx)),
                }
            )
            continue

        candidates = _threshold_candidates(
            model=model,
            X=X_valid[calibration_idx],
            y=y_valid[calibration_idx],
            sfreq=dataset.sfreq,
            ch_names=dataset.ch_names,
            config=config,
            thresholds=thresholds,
        )
        selected = _select_candidate(candidates, max_false_trigger_rate)
        eval_report = _evaluate_selected_threshold(
            model=model,
            X=X_valid[evaluation_idx],
            y=y_valid[evaluation_idx],
            sfreq=dataset.sfreq,
            ch_names=dataset.ch_names,
            config=config,
            threshold=float(selected["threshold"]),
        )
        offline = _offline_metrics(model, X_valid[evaluation_idx], y_valid[evaluation_idx], dataset.sfreq, config)
        status = _subject_status(offline, eval_report, max_false_trigger_rate, min_trigger_rate, min_auc)
        _add_to_aggregate(aggregate, eval_report)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "status": status,
                "calibration_epochs": int(len(calibration_idx)),
                "evaluation_epochs": int(len(evaluation_idx)),
                "selected_threshold": float(selected["threshold"]),
                "calibration_selected_metrics": selected,
                "evaluation_metrics": eval_report,
                "offline": offline,
                "calibration_candidates": candidates,
            }
        )

    result = {
        "metadata": {
            "dataset": config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "split_strategy": split.strategy,
            "test_subject_ids": split.test_subject_ids,
            "thresholds": thresholds,
            "calibration_trials_per_class": int(calibration_trials_per_class),
            "max_false_trigger_rate": max_false_trigger_rate,
            "min_trigger_rate": min_trigger_rate,
            "min_auc": min_auc,
            "n_epochs_total": int(len(dataset.y)),
            "n_epochs_valid": int(len(y_valid)),
            "quality_reject_rate": float(1.0 - len(y_valid) / max(1, len(dataset.y))),
        },
        "aggregate_evaluation_metrics": _finalize_aggregate(aggregate),
        "status_counts": _status_counts(subject_rows),
        "subjects": subject_rows,
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "calibration_window_thresholds.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _split_subject_calibration(
    subject_epoch_idx: np.ndarray,
    y: np.ndarray,
    calibration_trials_per_class: int,
) -> tuple[np.ndarray, np.ndarray]:
    calibration: list[int] = []
    for label in sorted(np.unique(y[subject_epoch_idx]).tolist()):
        label_idx = subject_epoch_idx[y[subject_epoch_idx] == label]
        calibration.extend(label_idx[:calibration_trials_per_class].tolist())
    calibration_set = set(calibration)
    evaluation = [idx for idx in subject_epoch_idx.tolist() if idx not in calibration_set]
    return np.asarray(sorted(calibration), dtype=int), np.asarray(evaluation, dtype=int)


def _training_samples(X, y, sfreq: float, config: dict) -> dict:
    training_mode = str(config.get("model", {}).get("training_mode", "epoch")).lower()
    if training_mode == "online_windows":
        windows = make_online_windows(X, y, sfreq, config)
        return {"X": windows.X, "y": windows.y}
    if training_mode == "epoch":
        return {"X": X, "y": y}
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def _offline_metrics(model, X: np.ndarray, y: np.ndarray, sfreq: float, config: dict) -> dict:
    samples = _training_samples(X, y, sfreq, config)
    y_pred = model.predict(samples["X"])
    y_score = model.predict_proba(samples["X"])[:, 1]
    return {
        "n_samples": int(len(samples["y"])),
        "balanced_accuracy": float(balanced_accuracy_score(samples["y"], y_pred)),
        "auc": float(roc_auc_score(samples["y"], y_score)),
        "f1": float(f1_score(samples["y"], y_pred)),
    }


def _threshold_candidates(model, X, y, sfreq: float, ch_names: list[str], config: dict, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        run_config = copy.deepcopy(config)
        run_config["online"]["trigger_threshold"] = float(threshold)
        report = simulate_session(model, X, y, sfreq, ch_names, run_config)
        rows.append(_compact_report(report) | {"threshold": float(threshold)})
    return rows


def _evaluate_selected_threshold(model, X, y, sfreq: float, ch_names: list[str], config: dict, threshold: float) -> dict:
    run_config = copy.deepcopy(config)
    run_config["online"]["trigger_threshold"] = float(threshold)
    return _compact_report(simulate_session(model, X, y, sfreq, ch_names, run_config))


def _compact_report(report: dict) -> dict:
    return {
        "n_trials": int(report["n_trials"]),
        "true_intention_trials": int(report["true_intention_trials"]),
        "rest_trials": int(report["rest_trials"]),
        "triggered_intention_trials": int(report["triggered_intention_trials"]),
        "triggered_rest_trials": int(report["triggered_rest_trials"]),
        "trigger_rate": float(report["trigger_rate"]),
        "false_trigger_rate": float(report["false_trigger_rate"]),
        "mean_trigger_delay_seconds": report["mean_trigger_delay_seconds"],
    }


def _select_candidate(candidates: list[dict], max_false_trigger_rate: float) -> dict:
    feasible = [row for row in candidates if row["false_trigger_rate"] <= max_false_trigger_rate]
    if feasible:
        return max(feasible, key=lambda row: (row["trigger_rate"], _delay_score(row["mean_trigger_delay_seconds"]), row["threshold"]))
    return min(candidates, key=lambda row: (row["false_trigger_rate"], -row["trigger_rate"], -row["threshold"]))


def _subject_status(
    offline: dict,
    evaluation: dict,
    max_false_trigger_rate: float,
    min_trigger_rate: float,
    min_auc: float,
) -> str:
    if float(offline.get("auc", 0.0)) < min_auc:
        return "not_ready_low_auc"
    if evaluation["false_trigger_rate"] > max_false_trigger_rate:
        return "not_ready_high_false_trigger"
    if evaluation["trigger_rate"] < min_trigger_rate:
        return "monitor_only_low_trigger"
    return "ready_for_trigger"


def _empty_aggregate() -> dict:
    return {
        "true_intention_trials": 0,
        "rest_trials": 0,
        "triggered_intention_trials": 0,
        "triggered_rest_trials": 0,
        "delay_sum": 0.0,
        "delay_count": 0,
    }


def _add_to_aggregate(aggregate: dict, selected: dict) -> None:
    aggregate["true_intention_trials"] += int(selected["true_intention_trials"])
    aggregate["rest_trials"] += int(selected["rest_trials"])
    aggregate["triggered_intention_trials"] += int(selected["triggered_intention_trials"])
    aggregate["triggered_rest_trials"] += int(selected["triggered_rest_trials"])
    if selected["mean_trigger_delay_seconds"] is not None:
        aggregate["delay_sum"] += float(selected["mean_trigger_delay_seconds"]) * int(selected["triggered_intention_trials"])
        aggregate["delay_count"] += int(selected["triggered_intention_trials"])


def _finalize_aggregate(aggregate: dict) -> dict:
    delay_count = int(aggregate.pop("delay_count"))
    delay_sum = float(aggregate.pop("delay_sum"))
    aggregate["trigger_rate"] = aggregate["triggered_intention_trials"] / max(1, aggregate["true_intention_trials"])
    aggregate["false_trigger_rate"] = aggregate["triggered_rest_trials"] / max(1, aggregate["rest_trials"])
    aggregate["mean_trigger_delay_seconds"] = None if delay_count == 0 else delay_sum / delay_count
    return aggregate


def _status_counts(subject_rows: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in subject_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def _delay_score(delay: float | None) -> float:
    if delay is None:
        return float("-inf")
    return -float(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate target-subject calibration-window threshold adaptation.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_full_riemannian_train_channel_standardized.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75])
    parser.add_argument("--calibration-trials-per-class", type=int, default=5)
    parser.add_argument("--max-false-trigger-rate", type=float)
    parser.add_argument("--min-trigger-rate", type=float, default=0.05)
    parser.add_argument("--min-auc", type=float, default=0.60)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_calibration_window_adaptation(
        config_path=args.config,
        thresholds=args.thresholds,
        calibration_trials_per_class=args.calibration_trials_per_class,
        output_path=args.output,
        max_false_trigger_rate=args.max_false_trigger_rate,
        min_trigger_rate=args.min_trigger_rate,
        min_auc=args.min_auc,
    )
    compact = {
        "metadata": result["metadata"],
        "aggregate_evaluation_metrics": result["aggregate_evaluation_metrics"],
        "status_counts": result["status_counts"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
