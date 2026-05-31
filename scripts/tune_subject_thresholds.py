from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.online import simulate_session
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def tune_subject_thresholds(
    config_path: str,
    thresholds: list[float],
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
    bundle = joblib.load(outputs["model_path"])
    model = bundle["model"]

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
    split = make_train_test_split(y_valid, valid_subject_ids, config)

    subject_rows = []
    aggregate = _empty_aggregate()
    for subject_id in sorted(map(str, np.unique(valid_subject_ids[split.test_idx]))):
        subject_mask = valid_subject_ids[split.test_idx].astype(str) == subject_id
        epoch_idx = split.test_idx[subject_mask]
        X_subject = X_valid[epoch_idx]
        y_subject = y_valid[epoch_idx]
        offline = _offline_metrics(model, X_subject, y_subject, dataset.sfreq, config)
        candidates = _subject_candidates(model, X_subject, y_subject, dataset.sfreq, dataset.ch_names, config, thresholds)
        selected = _select_candidate(candidates, max_false_trigger_rate)
        status = _subject_status(
            offline=offline,
            selected=selected,
            max_false_trigger_rate=max_false_trigger_rate,
            min_trigger_rate=min_trigger_rate,
            min_auc=min_auc,
        )
        _add_to_aggregate(aggregate, selected)
        subject_rows.append(
            {
                "subject_id": subject_id,
                "status": status,
                "selected_threshold": selected["threshold"],
                "selected_metrics": selected,
                "offline": offline,
                "candidates": candidates,
            }
        )

    aggregate_metrics = _finalize_aggregate(aggregate)
    result = {
        "metadata": {
            "dataset": config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "model_path": outputs["model_path"],
            "split_strategy": split.strategy,
            "test_subject_ids": split.test_subject_ids,
            "thresholds": thresholds,
            "max_false_trigger_rate": max_false_trigger_rate,
            "min_trigger_rate": min_trigger_rate,
            "min_auc": min_auc,
        },
        "aggregate_personalized_metrics": aggregate_metrics,
        "status_counts": _status_counts(subject_rows),
        "subjects": subject_rows,
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "subject_thresholds.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _subject_candidates(model, X, y, sfreq: float, ch_names: list[str], config: dict, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        run_config = json.loads(json.dumps(config))
        run_config["online"]["trigger_threshold"] = float(threshold)
        report = simulate_session(model, X, y, sfreq, ch_names, run_config)
        rows.append(
            {
                "threshold": float(threshold),
                "n_trials": int(report["n_trials"]),
                "true_intention_trials": int(report["true_intention_trials"]),
                "rest_trials": int(report["rest_trials"]),
                "triggered_intention_trials": int(report["triggered_intention_trials"]),
                "triggered_rest_trials": int(report["triggered_rest_trials"]),
                "trigger_rate": float(report["trigger_rate"]),
                "false_trigger_rate": float(report["false_trigger_rate"]),
                "mean_trigger_delay_seconds": report["mean_trigger_delay_seconds"],
            }
        )
    return rows


def _select_candidate(candidates: list[dict], max_false_trigger_rate: float) -> dict:
    feasible = [row for row in candidates if row["false_trigger_rate"] <= max_false_trigger_rate]
    if feasible:
        return max(feasible, key=lambda row: (row["trigger_rate"], _delay_score(row["mean_trigger_delay_seconds"]), row["threshold"]))
    return min(candidates, key=lambda row: (row["false_trigger_rate"], -row["trigger_rate"], -row["threshold"]))


def _subject_status(
    offline: dict,
    selected: dict,
    max_false_trigger_rate: float,
    min_trigger_rate: float,
    min_auc: float,
) -> str:
    if offline.get("skipped"):
        return "not_ready_single_class"
    if float(offline.get("auc", 0.0)) < min_auc:
        return "not_ready_low_auc"
    if selected["false_trigger_rate"] > max_false_trigger_rate:
        return "not_ready_high_false_trigger"
    if selected["trigger_rate"] < min_trigger_rate:
        return "monitor_only_low_trigger"
    return "ready_for_trigger"


def _offline_metrics(model, X: np.ndarray, y: np.ndarray, sfreq: float, config: dict) -> dict:
    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "single_class_subject"}

    training_mode = str(config.get("model", {}).get("training_mode", "epoch")).lower()
    if training_mode == "online_windows":
        windows = make_online_windows(X, y, sfreq, config)
        X_eval, y_eval = windows.X, windows.y
    elif training_mode == "epoch":
        X_eval, y_eval = X, y
    else:
        raise ValueError(f"Unsupported training_mode: {training_mode}")

    y_pred = model.predict(X_eval)
    y_score = model.predict_proba(X_eval)[:, 1]
    return {
        "skipped": False,
        "n_samples": int(len(y_eval)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
        "auc": float(roc_auc_score(y_eval, y_score)),
        "f1": float(f1_score(y_eval, y_pred)),
    }


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
    parser = argparse.ArgumentParser(description="Tune per-subject trigger thresholds on held-out test subjects.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_full_riemannian.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    parser.add_argument("--max-false-trigger-rate", type=float)
    parser.add_argument("--min-trigger-rate", type=float, default=0.05)
    parser.add_argument("--min-auc", type=float, default=0.60)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = tune_subject_thresholds(
        config_path=args.config,
        thresholds=args.thresholds,
        output_path=args.output,
        max_false_trigger_rate=args.max_false_trigger_rate,
        min_trigger_rate=args.min_trigger_rate,
        min_auc=args.min_auc,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
