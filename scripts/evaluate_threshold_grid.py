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
from stroke_bci_mvp.signal import apply_subject_normalization, filter_valid_epochs, notch_epochs


def evaluate_threshold_grid(config_path: str, thresholds: list[float], output_path: str | None = None) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
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
    valid_subject_ids = dataset.subject_ids[np.asarray([result.valid for result in quality_results], dtype=bool)]
    X_valid = apply_subject_normalization(X_valid, valid_subject_ids, config)
    split = make_train_test_split(y_valid, valid_subject_ids, config)
    X_test = X_valid[split.test_idx]
    y_test = y_valid[split.test_idx]

    rows = []
    for threshold in thresholds:
        run_config = json.loads(json.dumps(config))
        run_config["online"]["trigger_threshold"] = float(threshold)
        report = simulate_session(bundle["model"], X_test, y_test, dataset.sfreq, dataset.ch_names, run_config)
        rows.append(
            {
                "threshold": float(threshold),
                "n_trials": int(report["n_trials"]),
                "trigger_rate": float(report["trigger_rate"]),
                "false_trigger_rate": float(report["false_trigger_rate"]),
                "mean_trigger_delay_seconds": report["mean_trigger_delay_seconds"],
                "triggered_intention_trials": int(report["triggered_intention_trials"]),
                "triggered_rest_trials": int(report["triggered_rest_trials"]),
            }
        )

    selected = _select_threshold(rows, max_false_trigger_rate=float(config.get("calibration", {}).get("max_false_trigger_rate", 0.1)))
    result = {
        "metadata": {
            "dataset": config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "model_path": outputs["model_path"],
            "split_strategy": split.strategy,
            "test_subject_ids": split.test_subject_ids,
        },
        "selected_threshold": selected["threshold"],
        "selection_reason": "max_heldout_trigger_rate_under_false_trigger_constraint",
        "candidates": rows,
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "threshold_grid_heldout.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _select_threshold(rows: list[dict], max_false_trigger_rate: float) -> dict:
    feasible = [row for row in rows if row["false_trigger_rate"] <= max_false_trigger_rate]
    if feasible:
        return max(feasible, key=lambda row: (row["trigger_rate"], row["threshold"]))
    return min(rows, key=lambda row: (row["false_trigger_rate"], -row["trigger_rate"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out trigger metrics across thresholds.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_full_riemannian.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.55, 0.6, 0.65, 0.7, 0.75, 0.8])
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_threshold_grid(args.config, args.thresholds, args.output)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
