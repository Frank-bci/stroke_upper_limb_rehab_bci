from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_train_test_split
from stroke_bci_mvp.online import simulate_session
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def analyze_subject_errors(
    config_path: str,
    subjects: list[str] | None = None,
    output_path: str | None = None,
    subject_thresholds_path: str | None = None,
    top_windows: int = 10,
) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
    bundle = joblib.load(outputs["model_path"])
    model = bundle["model"]
    subject_thresholds = _load_subject_thresholds(subject_thresholds_path, outputs)
    subjects = subjects or _default_subjects(subject_thresholds)

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

    rows = []
    for subject_id in subjects:
        subject_mask = valid_subject_ids[split.test_idx].astype(str) == str(subject_id)
        if not np.any(subject_mask):
            rows.append({"subject_id": subject_id, "skipped": True, "reason": "subject_not_in_test_split"})
            continue
        epoch_idx = split.test_idx[subject_mask]
        X_subject = X_valid[epoch_idx]
        y_subject = y_valid[epoch_idx]
        threshold = _subject_threshold(subject_id, subject_thresholds, config)
        run_config = json.loads(json.dumps(config))
        run_config["online"]["trigger_threshold"] = threshold
        report = simulate_session(model, X_subject, y_subject, dataset.sfreq, dataset.ch_names, run_config)
        rows.append(
            _analyze_subject_report(
                subject_id=str(subject_id),
                threshold=threshold,
                report=report,
                source_epoch_indices=epoch_idx,
                top_windows=top_windows,
            )
        )

    result = {
        "metadata": {
            "dataset": config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "model_path": outputs["model_path"],
            "split_strategy": split.strategy,
            "test_subject_ids": split.test_subject_ids,
            "subject_thresholds_path": _resolved_subject_thresholds_path(subject_thresholds_path, outputs),
            "top_windows": int(top_windows),
        },
        "subjects": rows,
        "interpretation": _interpret(rows),
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "subject_error_analysis.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _analyze_subject_report(
    subject_id: str,
    threshold: float,
    report: dict,
    source_epoch_indices: np.ndarray,
    top_windows: int,
) -> dict:
    false_trigger_trials = []
    true_trigger_trials = []
    high_probability_rest_windows = []
    rest_probabilities = []
    intention_probabilities = []
    rest_quality_scores = []
    intention_quality_scores = []

    for trial in report["trials"]:
        label = int(trial["label"])
        source_epoch_idx = int(source_epoch_indices[int(trial["trial_index"])])
        timeline = trial["timeline"]
        probabilities = [float(row["intention_probability"]) for row in timeline]
        quality_scores = [float(row["quality_score"]) for row in timeline]
        max_row = max(timeline, key=lambda row: float(row["intention_probability"]))
        trigger_rows = [row for row in timeline if row["triggered"]]
        trial_summary = {
            "trial_index": int(trial["trial_index"]),
            "source_epoch_index": source_epoch_idx,
            "trigger_time_seconds": trial["trigger_time_seconds"],
            "max_probability": float(max_row["intention_probability"]),
            "max_probability_time_seconds": float(max_row["time_seconds"]),
            "mean_probability": float(np.mean(probabilities)),
            "min_quality_score": float(np.min(quality_scores)),
            "mean_quality_score": float(np.mean(quality_scores)),
            "trigger_windows": trigger_rows,
        }
        if label == 0:
            rest_probabilities.extend(probabilities)
            rest_quality_scores.extend(quality_scores)
            if trial["triggered"]:
                false_trigger_trials.append(trial_summary)
            for row in timeline:
                high_probability_rest_windows.append(
                    {
                        "trial_index": int(trial["trial_index"]),
                        "source_epoch_index": source_epoch_idx,
                        "time_seconds": float(row["time_seconds"]),
                        "intention_probability": float(row["intention_probability"]),
                        "quality_score": float(row["quality_score"]),
                        "triggered": bool(row["triggered"]),
                        "reason": row["reason"],
                    }
                )
        else:
            intention_probabilities.extend(probabilities)
            intention_quality_scores.extend(quality_scores)
            if trial["triggered"]:
                true_trigger_trials.append(trial_summary)

    high_probability_rest_windows = sorted(
        high_probability_rest_windows,
        key=lambda row: row["intention_probability"],
        reverse=True,
    )[:top_windows]

    return {
        "subject_id": subject_id,
        "skipped": False,
        "threshold": float(threshold),
        "session": {
            "n_trials": int(report["n_trials"]),
            "true_intention_trials": int(report["true_intention_trials"]),
            "rest_trials": int(report["rest_trials"]),
            "triggered_intention_trials": int(report["triggered_intention_trials"]),
            "triggered_rest_trials": int(report["triggered_rest_trials"]),
            "trigger_rate": float(report["trigger_rate"]),
            "false_trigger_rate": float(report["false_trigger_rate"]),
            "mean_trigger_delay_seconds": report["mean_trigger_delay_seconds"],
        },
        "probability_summary": {
            "rest": _summary(rest_probabilities),
            "motor_intention": _summary(intention_probabilities),
            "rest_minus_intention_mean": _mean_delta(rest_probabilities, intention_probabilities),
        },
        "quality_summary": {
            "rest": _summary(rest_quality_scores),
            "motor_intention": _summary(intention_quality_scores),
        },
        "false_trigger_trials": false_trigger_trials,
        "true_trigger_trials": true_trigger_trials,
        "top_high_probability_rest_windows": high_probability_rest_windows,
        "decision_reasons": report["decision_reasons"],
    }


def _load_subject_thresholds(path: str | None, outputs: dict) -> dict:
    resolved = _resolved_subject_thresholds_path(path, outputs)
    if not resolved:
        return {}
    threshold_path = Path(resolved)
    if not threshold_path.exists():
        return {}
    data = json.loads(threshold_path.read_text(encoding="utf-8"))
    return {row["subject_id"]: row for row in data.get("subjects", [])}


def _resolved_subject_thresholds_path(path: str | None, outputs: dict) -> str | None:
    if path:
        return path
    default = Path(outputs["dir"]) / "subject_thresholds.json"
    return str(default) if default.exists() else None


def _default_subjects(subject_thresholds: dict) -> list[str]:
    if not subject_thresholds:
        return ["sub-21", "sub-43"]
    rows = list(subject_thresholds.values())
    risky = [
        row["subject_id"]
        for row in rows
        if row.get("status") != "ready_for_trigger"
        or float(row.get("selected_metrics", {}).get("false_trigger_rate", 0.0)) >= 0.075
        or float(row.get("selected_threshold", 0.0)) >= 0.65
    ]
    return sorted(risky) or ["sub-21", "sub-43"]


def _subject_threshold(subject_id: str, subject_thresholds: dict, config: dict) -> float:
    if subject_id in subject_thresholds:
        return float(subject_thresholds[subject_id]["selected_threshold"])
    return float(config["online"]["trigger_threshold"])


def _summary(values: list[float]) -> dict:
    if not values:
        return {"count": 0, "mean": None, "std": None, "min": None, "p95": None, "max": None}
    arr = np.asarray(values, dtype=float)
    return {
        "count": int(len(arr)),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "p95": float(np.quantile(arr, 0.95)),
        "max": float(np.max(arr)),
    }


def _mean_delta(rest: list[float], intention: list[float]) -> float | None:
    if not rest or not intention:
        return None
    return float(np.mean(rest) - np.mean(intention))


def _interpret(rows: list[dict]) -> list[str]:
    messages = []
    for row in rows:
        if row.get("skipped"):
            messages.append(f"{row['subject_id']}: skipped ({row['reason']}).")
            continue
        rest_mean = row["probability_summary"]["rest"]["mean"]
        intention_mean = row["probability_summary"]["motor_intention"]["mean"]
        false_rate = row["session"]["false_trigger_rate"]
        trigger_rate = row["session"]["trigger_rate"]
        if false_rate > 0.1:
            messages.append(f"{row['subject_id']}: false trigger remains high at threshold {row['threshold']}.")
        elif trigger_rate < 0.05:
            messages.append(f"{row['subject_id']}: monitor-only candidate because true trigger rate is low.")
        elif rest_mean is not None and intention_mean is not None and rest_mean >= intention_mean:
            messages.append(f"{row['subject_id']}: rest probabilities overlap or exceed intention probabilities; check label timing or session drift.")
        else:
            messages.append(f"{row['subject_id']}: personalized threshold controls false triggers, but inspect top rest windows for drift.")
    return messages


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze high-risk subject false triggers and window probabilities.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_full_riemannian.yaml")
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--subject-thresholds")
    parser.add_argument("--top-windows", type=int, default=10)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = analyze_subject_errors(
        config_path=args.config,
        subjects=args.subjects,
        output_path=args.output,
        subject_thresholds_path=args.subject_thresholds,
        top_windows=args.top_windows,
    )
    compact = {
        "metadata": result["metadata"],
        "interpretation": result["interpretation"],
        "subjects": [
            {
                "subject_id": row["subject_id"],
                "threshold": row.get("threshold"),
                "session": row.get("session"),
                "probability_summary": row.get("probability_summary"),
                "false_trigger_count": len(row.get("false_trigger_trials", [])),
            }
            for row in result["subjects"]
        ],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
