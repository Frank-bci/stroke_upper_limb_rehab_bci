from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TriggerParams:
    threshold: float
    consecutive_windows: int
    refractory_seconds: float
    min_quality_score: float
    task_start_seconds: float
    task_end_seconds: float


class TriggerDecision:
    def __init__(self, params: TriggerParams):
        self.params = params
        self._streak = 0
        self._last_trigger_time = -1e9

    def update(self, time_seconds: float, probability: float, quality_score: float) -> tuple[bool, str]:
        if not (self.params.task_start_seconds <= time_seconds <= self.params.task_end_seconds):
            self._streak = 0
            return False, "outside_task_period"

        if time_seconds - self._last_trigger_time < self.params.refractory_seconds:
            self._streak = 0
            return False, "refractory_period"

        if quality_score < self.params.min_quality_score:
            self._streak = 0
            return False, "low_signal_quality"

        if probability < self.params.threshold:
            self._streak = 0
            return False, "low_intention_probability"

        self._streak += 1
        if self._streak < self.params.consecutive_windows:
            return False, "waiting_for_confirmation"

        self._last_trigger_time = time_seconds
        self._streak = 0
        return True, "trigger_bu100_assist_open_hand"

