from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def analyze_gates(
    repeated_json: str,
    min_thresholds: list[float],
    min_calibration_auc: float = 0.60,
    output_path: str | None = None,
) -> dict:
    data = json.loads(Path(repeated_json).read_text(encoding="utf-8"))
    rows = []
    for min_threshold in min_thresholds:
        run_metrics = []
        for run in data["runs"]:
            aggregate = _empty_aggregate()
            ready_subjects = 0
            for subject in run["subjects"]:
                evaluation = subject.get("evaluation_metrics", {})
                if not evaluation:
                    continue
                _add_denominators(aggregate, evaluation)
                if _passes_gate(subject, min_threshold=min_threshold, min_calibration_auc=min_calibration_auc):
                    _add_triggers(aggregate, evaluation)
                    ready_subjects += 1
            run_metrics.append(_finalize_aggregate(aggregate) | {"ready_subjects": float(ready_subjects)})
        rows.append(
            {
                "min_selected_threshold": float(min_threshold),
                "min_calibration_auc": float(min_calibration_auc),
                "summary": _summary(run_metrics),
                "runs": run_metrics,
            }
        )

    result = {
        "metadata": {
            "source": str(Path(repeated_json)),
            "seeds": data.get("metadata", {}).get("seeds", []),
            "min_thresholds": min_thresholds,
            "min_calibration_auc": float(min_calibration_auc),
        },
        "gate_sweep": rows,
    }
    if output_path is None:
        output_path = str(Path(repeated_json).with_name("calibration_window_gate_sensitivity.json"))
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _passes_gate(subject: dict[str, Any], min_threshold: float, min_calibration_auc: float) -> bool:
    return (
        float(subject.get("selected_threshold", 0.0)) >= float(min_threshold)
        and float(subject.get("calibration_offline", {}).get("auc", 0.0)) >= float(min_calibration_auc)
    )


def _empty_aggregate() -> dict:
    return {
        "true_intention_trials": 0,
        "rest_trials": 0,
        "triggered_intention_trials": 0,
        "triggered_rest_trials": 0,
        "delay_sum": 0.0,
        "delay_count": 0,
    }


def _add_denominators(aggregate: dict, evaluation: dict) -> None:
    aggregate["true_intention_trials"] += int(evaluation["true_intention_trials"])
    aggregate["rest_trials"] += int(evaluation["rest_trials"])


def _add_triggers(aggregate: dict, evaluation: dict) -> None:
    aggregate["triggered_intention_trials"] += int(evaluation["triggered_intention_trials"])
    aggregate["triggered_rest_trials"] += int(evaluation["triggered_rest_trials"])
    if evaluation["mean_trigger_delay_seconds"] is not None:
        aggregate["delay_sum"] += float(evaluation["mean_trigger_delay_seconds"]) * int(evaluation["triggered_intention_trials"])
        aggregate["delay_count"] += int(evaluation["triggered_intention_trials"])


def _finalize_aggregate(aggregate: dict) -> dict:
    delay_count = int(aggregate.pop("delay_count"))
    delay_sum = float(aggregate.pop("delay_sum"))
    aggregate["trigger_rate"] = aggregate["triggered_intention_trials"] / max(1, aggregate["true_intention_trials"])
    aggregate["false_trigger_rate"] = aggregate["triggered_rest_trials"] / max(1, aggregate["rest_trials"])
    aggregate["mean_trigger_delay_seconds"] = None if delay_count == 0 else delay_sum / delay_count
    return aggregate


def _summary(runs: list[dict]) -> dict:
    keys = ["trigger_rate", "false_trigger_rate", "mean_trigger_delay_seconds", "ready_subjects"]
    return {key: _mean_std([float(run.get(key) or 0.0) for run in runs]) for key in keys}


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze deployment gates on repeated calibration-window results.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--min-thresholds", nargs="+", type=float, default=[0.0, 0.525, 0.55, 0.575, 0.6])
    parser.add_argument("--min-calibration-auc", type=float, default=0.60)
    parser.add_argument("--output")
    args = parser.parse_args()

    result = analyze_gates(
        repeated_json=args.input,
        min_thresholds=args.min_thresholds,
        min_calibration_auc=args.min_calibration_auc,
        output_path=args.output,
    )
    compact = {
        "metadata": result["metadata"],
        "gate_sweep": [
            {
                "min_selected_threshold": row["min_selected_threshold"],
                "summary": row["summary"],
            }
            for row in result["gate_sweep"]
        ],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
