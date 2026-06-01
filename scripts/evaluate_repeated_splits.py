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


def evaluate_repeated_splits(
    config_path: str,
    seeds: list[int],
    thresholds: list[float],
    output_path: str | None = None,
    min_auc: float = 0.60,
    min_trigger_rate: float = 0.05,
    max_false_trigger_rate: float | None = None,
) -> dict:
    base_config = load_config(config_path)
    outputs = base_config["outputs"]
    max_false_trigger_rate = (
        float(base_config.get("calibration", {}).get("max_false_trigger_rate", 0.1))
        if max_false_trigger_rate is None
        else float(max_false_trigger_rate)
    )

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
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["dataset"]["random_state"] = int(seed)
        split = make_train_test_split(y_valid, valid_subject_ids, config)
        train_data = _training_samples(X_valid[split.train_idx], y_valid[split.train_idx], dataset.sfreq, config)
        test_data = _training_samples(X_valid[split.test_idx], y_valid[split.test_idx], dataset.sfreq, config)

        model = build_model(config, dataset.sfreq, ch_names=dataset.ch_names)
        model.fit(train_data["X"], train_data["y"])

        offline = _offline_metrics(model, test_data["X"], test_data["y"])
        global_report = _compact_session(
            simulate_session(
                model=model,
                X=X_valid[split.test_idx],
                y=y_valid[split.test_idx],
                sfreq=dataset.sfreq,
                ch_names=dataset.ch_names,
                config=config,
            )
        )
        personalized = _personalized_thresholds(
            model=model,
            X=X_valid[split.test_idx],
            y=y_valid[split.test_idx],
            subject_ids=valid_subject_ids[split.test_idx],
            sfreq=dataset.sfreq,
            ch_names=dataset.ch_names,
            config=config,
            thresholds=thresholds,
            max_false_trigger_rate=max_false_trigger_rate,
            min_auc=min_auc,
            min_trigger_rate=min_trigger_rate,
        )
        runs.append(
            {
                "seed": int(seed),
                "test_subject_ids": split.test_subject_ids,
                "train_epochs": int(len(split.train_idx)),
                "test_epochs": int(len(split.test_idx)),
                "train_samples": int(len(train_data["y"])),
                "test_samples": int(len(test_data["y"])),
                "offline": offline,
                "global_threshold": {
                    "threshold": float(config["online"]["trigger_threshold"]),
                    "metrics": global_report,
                },
                "personalized_thresholds": personalized,
            }
        )

    result = {
        "metadata": {
            "dataset": base_config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "seeds": seeds,
            "thresholds": thresholds,
            "max_false_trigger_rate": max_false_trigger_rate,
            "min_auc": min_auc,
            "min_trigger_rate": min_trigger_rate,
            "n_epochs_total": int(len(dataset.y)),
            "n_epochs_valid": int(len(y_valid)),
            "quality_reject_rate": float(1.0 - len(y_valid) / max(1, len(dataset.y))),
        },
        "summary": _summary(runs),
        "runs": runs,
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "repeated_subject_splits.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _training_samples(X, y, sfreq: float, config: dict) -> dict:
    training_mode = str(config.get("model", {}).get("training_mode", "epoch")).lower()
    if training_mode == "online_windows":
        windows = make_online_windows(X, y, sfreq, config)
        return {"X": windows.X, "y": windows.y}
    if training_mode == "epoch":
        return {"X": X, "y": y}
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def _offline_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
    }


def _personalized_thresholds(
    model,
    X,
    y,
    subject_ids,
    sfreq: float,
    ch_names: list[str],
    config: dict,
    thresholds: list[float],
    max_false_trigger_rate: float,
    min_auc: float,
    min_trigger_rate: float,
) -> dict:
    subjects = []
    aggregate_all = _empty_aggregate()
    aggregate_ready = _empty_aggregate()

    for subject_id in sorted(map(str, np.unique(subject_ids))):
        mask = subject_ids.astype(str) == subject_id
        X_subject = X[mask]
        y_subject = y[mask]
        offline = _subject_offline(model, X_subject, y_subject, sfreq, config)
        candidates = _subject_candidates(model, X_subject, y_subject, sfreq, ch_names, config, thresholds)
        selected = _select_candidate(candidates, max_false_trigger_rate)
        status = _subject_status(offline, selected, max_false_trigger_rate, min_trigger_rate, min_auc)
        _add_to_aggregate(aggregate_all, selected)
        if status == "ready_for_trigger":
            _add_to_aggregate(aggregate_ready, selected)
        else:
            _add_monitor_only_to_aggregate(aggregate_ready, selected)
        subjects.append(
            {
                "subject_id": subject_id,
                "status": status,
                "selected_threshold": selected["threshold"],
                "selected_metrics": selected,
                "offline": offline,
            }
        )

    return {
        "all_subjects": _finalize_aggregate(aggregate_all),
        "ready_subjects_only": _finalize_aggregate(aggregate_ready),
        "status_counts": _status_counts(subjects),
        "subjects": subjects,
    }


def _subject_offline(model, X, y, sfreq: float, config: dict) -> dict:
    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "single_class_subject"}
    samples = _training_samples(X, y, sfreq, config)
    return _offline_metrics(model, samples["X"], samples["y"]) | {"skipped": False}


def _subject_candidates(model, X, y, sfreq: float, ch_names: list[str], config: dict, thresholds: list[float]) -> list[dict]:
    rows = []
    for threshold in thresholds:
        run_config = copy.deepcopy(config)
        run_config["online"]["trigger_threshold"] = float(threshold)
        report = simulate_session(model, X, y, sfreq, ch_names, run_config)
        rows.append(
            {
                "threshold": float(threshold),
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


def _subject_status(offline: dict, selected: dict, max_false_trigger_rate: float, min_trigger_rate: float, min_auc: float) -> str:
    if offline.get("skipped"):
        return "not_ready_single_class"
    if float(offline.get("auc", 0.0)) < min_auc:
        return "not_ready_low_auc"
    if selected["false_trigger_rate"] > max_false_trigger_rate:
        return "not_ready_high_false_trigger"
    if selected["trigger_rate"] < min_trigger_rate:
        return "monitor_only_low_trigger"
    return "ready_for_trigger"


def _compact_session(report: dict) -> dict:
    keys = [
        "n_trials",
        "true_intention_trials",
        "rest_trials",
        "triggered_intention_trials",
        "triggered_rest_trials",
        "trigger_rate",
        "false_trigger_rate",
        "mean_trigger_delay_seconds",
    ]
    return {key: report.get(key) for key in keys}


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


def _add_monitor_only_to_aggregate(aggregate: dict, selected: dict) -> None:
    aggregate["true_intention_trials"] += int(selected["true_intention_trials"])
    aggregate["rest_trials"] += int(selected["rest_trials"])


def _finalize_aggregate(aggregate: dict) -> dict:
    delay_count = int(aggregate.pop("delay_count"))
    delay_sum = float(aggregate.pop("delay_sum"))
    aggregate["trigger_rate"] = aggregate["triggered_intention_trials"] / max(1, aggregate["true_intention_trials"])
    aggregate["false_trigger_rate"] = aggregate["triggered_rest_trials"] / max(1, aggregate["rest_trials"])
    aggregate["mean_trigger_delay_seconds"] = None if delay_count == 0 else delay_sum / delay_count
    return aggregate


def _status_counts(subjects: list[dict]) -> dict:
    counts: dict[str, int] = {}
    for row in subjects:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    return counts


def _summary(runs: list[dict]) -> dict:
    paths = {
        "offline_balanced_accuracy": ["offline", "balanced_accuracy"],
        "offline_auc": ["offline", "auc"],
        "offline_f1": ["offline", "f1"],
        "global_trigger_rate": ["global_threshold", "metrics", "trigger_rate"],
        "global_false_trigger_rate": ["global_threshold", "metrics", "false_trigger_rate"],
        "personalized_all_trigger_rate": ["personalized_thresholds", "all_subjects", "trigger_rate"],
        "personalized_all_false_trigger_rate": ["personalized_thresholds", "all_subjects", "false_trigger_rate"],
        "personalized_ready_trigger_rate": ["personalized_thresholds", "ready_subjects_only", "trigger_rate"],
        "personalized_ready_false_trigger_rate": ["personalized_thresholds", "ready_subjects_only", "false_trigger_rate"],
    }
    return {key: _mean_std([_get_path(run, path) for run in runs]) for key, path in paths.items()}


def _get_path(row: dict, path: list[str]) -> float:
    value = row
    for key in path:
        value = value[key]
    return float(value)


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def _delay_score(delay: float | None) -> float:
    if delay is None:
        return float("-inf")
    return -float(delay)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repeated subject-aware train/test splits.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_full_riemannian.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.55, 0.6, 0.65, 0.7])
    parser.add_argument("--max-false-trigger-rate", type=float)
    parser.add_argument("--min-trigger-rate", type=float, default=0.05)
    parser.add_argument("--min-auc", type=float, default=0.60)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_repeated_splits(
        config_path=args.config,
        seeds=args.seeds,
        thresholds=args.thresholds,
        output_path=args.output,
        min_auc=args.min_auc,
        min_trigger_rate=args.min_trigger_rate,
        max_false_trigger_rate=args.max_false_trigger_rate,
    )
    compact = {"metadata": result["metadata"], "summary": result["summary"]}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
