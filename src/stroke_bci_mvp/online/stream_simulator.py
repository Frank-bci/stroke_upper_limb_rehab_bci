from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from stroke_bci_mvp.online.trigger_decision import TriggerDecision, TriggerParams
from stroke_bci_mvp.signal.quality import assess_epoch_quality


def _positive_proba(model, X_window: np.ndarray) -> float:
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(X_window)[0, 1])
    decision = float(model.decision_function(X_window)[0])
    return 1.0 / (1.0 + np.exp(-decision))


def simulate_session(
    model,
    X: np.ndarray,
    y: np.ndarray,
    sfreq: float,
    ch_names: list[str],
    config: dict,
) -> dict[str, Any]:
    online_cfg = config["online"]
    quality_cfg = config["quality"]

    window_samples = int(round(float(online_cfg["window_seconds"]) * sfreq))
    step_samples = int(round(float(online_cfg["step_seconds"]) * sfreq))
    params = TriggerParams(
        threshold=float(online_cfg["trigger_threshold"]),
        consecutive_windows=int(online_cfg["consecutive_windows"]),
        refractory_seconds=float(online_cfg["refractory_seconds"]),
        min_quality_score=float(quality_cfg["min_score"]),
        task_start_seconds=float(online_cfg["task_start_seconds"]),
        task_end_seconds=float(online_cfg["task_end_seconds"]),
    )

    trial_summaries = []
    reason_counter: Counter[str] = Counter()
    trigger_delays = []
    true_intention_trials = int(np.sum(y == 1))
    rest_trials = int(np.sum(y == 0))
    triggered_intention = 0
    triggered_rest = 0

    for trial_idx, (epoch, label) in enumerate(zip(X, y)):
        decision = TriggerDecision(params)
        trial_triggered = False
        trigger_time = None
        timeline = []

        for start in range(0, epoch.shape[-1] - window_samples + 1, step_samples):
            stop = start + window_samples
            window = epoch[:, start:stop][np.newaxis, :, :]
            window_center = (start + window_samples / 2) / sfreq
            quality = assess_epoch_quality(window[0], sfreq, ch_names, quality_cfg)
            proba = _positive_proba(model, window)
            triggered, reason = decision.update(window_center, proba, quality.score)
            reason_counter[reason] += 1
            timeline.append(
                {
                    "time_seconds": round(window_center, 3),
                    "intention_probability": round(proba, 4),
                    "quality_score": round(quality.score, 2),
                    "triggered": triggered,
                    "reason": reason,
                }
            )
            if triggered and not trial_triggered:
                trial_triggered = True
                trigger_time = window_center

        if trial_triggered and label == 1:
            triggered_intention += 1
            trigger_delays.append(max(0.0, float(trigger_time) - params.task_start_seconds))
        if trial_triggered and label == 0:
            triggered_rest += 1

        trial_summaries.append(
            {
                "trial_index": trial_idx,
                "label": int(label),
                "triggered": trial_triggered,
                "trigger_time_seconds": None if trigger_time is None else round(float(trigger_time), 3),
                "timeline": timeline,
            }
        )

    return {
        "n_trials": int(len(y)),
        "true_intention_trials": true_intention_trials,
        "rest_trials": rest_trials,
        "triggered_intention_trials": triggered_intention,
        "triggered_rest_trials": triggered_rest,
        "trigger_rate": triggered_intention / max(1, true_intention_trials),
        "false_trigger_rate": triggered_rest / max(1, rest_trials),
        "mean_trigger_delay_seconds": None if not trigger_delays else float(np.mean(trigger_delays)),
        "decision_reasons": dict(reason_counter),
        "trials": trial_summaries,
    }

