from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from stroke_bci_mvp.config import load_config

from evaluate_calibration_window_adaptation import evaluate_calibration_window_adaptation


def evaluate_repeated_calibration_window_adaptation(
    config_path: str,
    seeds: list[int],
    thresholds: list[float],
    calibration_trials_per_class: int = 5,
    output_path: str | None = None,
    max_false_trigger_rate: float | None = None,
    false_trigger_margin: float = 0.0,
    threshold_safety_steps: int = 0,
    min_trigger_rate: float = 0.05,
    min_auc: float = 0.60,
) -> dict:
    base_config = load_config(config_path)
    outputs = base_config["outputs"]

    runs = []
    for seed in seeds:
        result = evaluate_calibration_window_adaptation(
            config_path=config_path,
            thresholds=thresholds,
            calibration_trials_per_class=calibration_trials_per_class,
            save_output=False,
            random_state=int(seed),
            max_false_trigger_rate=max_false_trigger_rate,
            false_trigger_margin=false_trigger_margin,
            threshold_safety_steps=threshold_safety_steps,
            min_trigger_rate=min_trigger_rate,
            min_auc=min_auc,
        )
        runs.append(
            {
                "seed": int(seed),
                "test_subject_ids": result["metadata"]["test_subject_ids"],
                "aggregate_evaluation_metrics": result["aggregate_evaluation_metrics"],
                "aggregate_ready_evaluation_metrics": result["aggregate_ready_evaluation_metrics"],
                "status_counts": result["status_counts"],
                "deployment_status_counts": result["deployment_status_counts"],
                "subjects": result["subjects"],
            }
        )

    result = {
        "metadata": {
            "dataset": base_config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "seeds": seeds,
            "thresholds": thresholds,
            "calibration_trials_per_class": int(calibration_trials_per_class),
            "max_false_trigger_rate": (
                float(base_config.get("calibration", {}).get("max_false_trigger_rate", 0.1))
                if max_false_trigger_rate is None
                else float(max_false_trigger_rate)
            ),
            "false_trigger_margin": float(false_trigger_margin),
            "threshold_safety_steps": int(threshold_safety_steps),
            "min_trigger_rate": float(min_trigger_rate),
            "min_auc": float(min_auc),
        },
        "summary": _summary(runs),
        "runs": runs,
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "calibration_window_repeated_splits.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _summary(runs: list[dict]) -> dict:
    paths = {
        "trigger_rate": ["aggregate_evaluation_metrics", "trigger_rate"],
        "false_trigger_rate": ["aggregate_evaluation_metrics", "false_trigger_rate"],
        "mean_trigger_delay_seconds": ["aggregate_evaluation_metrics", "mean_trigger_delay_seconds"],
        "ready_trigger_rate": ["aggregate_ready_evaluation_metrics", "trigger_rate"],
        "ready_false_trigger_rate": ["aggregate_ready_evaluation_metrics", "false_trigger_rate"],
        "ready_mean_trigger_delay_seconds": ["aggregate_ready_evaluation_metrics", "mean_trigger_delay_seconds"],
        "ready_for_trigger": ["status_counts", "ready_for_trigger"],
        "monitor_only_low_trigger": ["status_counts", "monitor_only_low_trigger"],
        "not_ready_high_false_trigger": ["status_counts", "not_ready_high_false_trigger"],
        "not_ready_low_auc": ["status_counts", "not_ready_low_auc"],
        "deployment_ready_for_trigger": ["deployment_status_counts", "ready_for_trigger"],
        "deployment_monitor_only_low_trigger": ["deployment_status_counts", "monitor_only_low_trigger"],
        "deployment_not_ready_high_false_trigger": ["deployment_status_counts", "not_ready_high_false_trigger"],
        "deployment_not_ready_low_auc": ["deployment_status_counts", "not_ready_low_auc"],
    }
    return {key: _mean_std([_get_path(run, path) for run in runs]) for key, path in paths.items()}


def _get_path(row: dict, path: list[str]) -> float:
    value: Any = row
    for key in path:
        if not isinstance(value, dict):
            return 0.0
        value = value.get(key, 0)
    if value is None:
        return 0.0
    return float(value)


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repeated calibration-window adaptation splits.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_full_riemannian_train_channel_standardized.yaml")
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.5, 0.525, 0.55, 0.575, 0.6, 0.625, 0.65, 0.675, 0.7, 0.725, 0.75])
    parser.add_argument("--calibration-trials-per-class", type=int, default=5)
    parser.add_argument("--max-false-trigger-rate", type=float)
    parser.add_argument("--false-trigger-margin", type=float, default=0.0)
    parser.add_argument("--threshold-safety-steps", type=int, default=0)
    parser.add_argument("--min-trigger-rate", type=float, default=0.05)
    parser.add_argument("--min-auc", type=float, default=0.60)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_repeated_calibration_window_adaptation(
        config_path=args.config,
        seeds=args.seeds,
        thresholds=args.thresholds,
        calibration_trials_per_class=args.calibration_trials_per_class,
        output_path=args.output,
        max_false_trigger_rate=args.max_false_trigger_rate,
        false_trigger_margin=args.false_trigger_margin,
        threshold_safety_steps=args.threshold_safety_steps,
        min_trigger_rate=args.min_trigger_rate,
        min_auc=args.min_auc,
    )
    print(json.dumps({"metadata": result["metadata"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
