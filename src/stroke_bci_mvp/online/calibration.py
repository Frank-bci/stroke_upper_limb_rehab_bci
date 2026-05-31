from __future__ import annotations

import copy
from typing import Any

import numpy as np

from stroke_bci_mvp.online.stream_simulator import simulate_session


def calibrate_trigger_threshold(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    config: dict,
    thresholds: list[float],
    max_false_trigger_rate: float,
) -> dict[str, Any]:
    """Select a trigger threshold using calibration epochs.

    The selected threshold maximizes effective trigger rate subject to a maximum
    false-trigger-rate constraint. If no candidate satisfies the constraint, the
    least false-triggering candidate is selected as a conservative fallback.
    """

    rows = []
    for threshold in thresholds:
        run_config = copy.deepcopy(config)
        run_config["online"]["trigger_threshold"] = float(threshold)
        report = simulate_session(model, X, y, sfreq, ch_names, run_config)
        rows.append(
            {
                "threshold": float(threshold),
                "trigger_rate": float(report["trigger_rate"]),
                "false_trigger_rate": float(report["false_trigger_rate"]),
                "mean_trigger_delay_seconds": report["mean_trigger_delay_seconds"],
                "triggered_intention_trials": int(report["triggered_intention_trials"]),
                "triggered_rest_trials": int(report["triggered_rest_trials"]),
            }
        )

    feasible = [row for row in rows if row["false_trigger_rate"] <= max_false_trigger_rate]
    if feasible:
        selected = max(
            feasible,
            key=lambda row: (
                row["trigger_rate"],
                _delay_score(row["mean_trigger_delay_seconds"]),
                row["threshold"],
            ),
        )
        reason = "max_trigger_rate_under_false_trigger_constraint"
    else:
        selected = max(
            rows,
            key=lambda row: (
                -row["false_trigger_rate"],
                row["trigger_rate"],
                _delay_score(row["mean_trigger_delay_seconds"]),
                row["threshold"],
            ),
        )
        reason = "no_threshold_met_false_trigger_constraint"

    return {
        "selected_threshold": selected["threshold"],
        "selection_reason": reason,
        "max_false_trigger_rate": float(max_false_trigger_rate),
        "selected_metrics": selected,
        "candidates": rows,
    }


def _delay_score(delay: float | None) -> float:
    if delay is None:
        return float("-inf")
    return -float(delay)
