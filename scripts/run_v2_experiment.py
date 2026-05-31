from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_trigger_threshold import calibrate
from simulate_online import simulate
from train_baseline import train


def run_v2_experiment(config_paths: list[str], summary_path: str) -> list[dict]:
    results = []
    for config_path in config_paths:
        offline = train(config_path)
        session = simulate(config_path)
        threshold = calibrate(config_path)
        results.append(
            {
                "config_path": config_path,
                "dataset": offline["dataset"],
                "model_type": _model_type(config_path),
                "offline": _compact_offline(offline),
                "session": _compact_session(session),
                "threshold": _compact_threshold(threshold),
            }
        )

    output_path = Path(summary_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return results


def _model_type(config_path: str) -> str:
    from stroke_bci_mvp.config import load_config

    return str(load_config(config_path).get("model", {}).get("type", "unknown"))


def _compact_offline(metrics: dict) -> dict:
    keys = [
        "n_epochs_valid",
        "train_epochs",
        "test_epochs",
        "train_samples",
        "test_samples",
        "split_strategy",
        "test_subject_ids",
        "balanced_accuracy",
        "auc",
        "f1",
    ]
    return {key: metrics.get(key) for key in keys}


def _compact_session(report: dict) -> dict:
    keys = [
        "n_trials",
        "true_intention_trials",
        "rest_trials",
        "trigger_rate",
        "false_trigger_rate",
        "mean_trigger_delay_seconds",
        "triggered_intention_trials",
        "triggered_rest_trials",
    ]
    return {key: report.get(key) for key in keys}


def _compact_threshold(result: dict) -> dict:
    return {
        "selected_threshold": result.get("selected_threshold"),
        "selection_reason": result.get("selection_reason"),
        "calibration_strategy": result.get("calibration_strategy"),
        "max_false_trigger_rate": result.get("max_false_trigger_rate"),
        "selected_metrics": result.get("selected_metrics"),
        "heldout_test_metrics_at_selected_threshold": result.get("heldout_test_metrics_at_selected_threshold"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the v2.0 real-data Riemannian experiment bundle.")
    parser.add_argument(
        "--configs",
        nargs="+",
        default=[
            "configs/v2/figshare_stroke_riemannian.yaml",
            "configs/v2/physionet_riemannian.yaml",
        ],
    )
    parser.add_argument("--summary-path", default="outputs/v2_summary.json")
    args = parser.parse_args()

    results = run_v2_experiment(args.configs, args.summary_path)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
