from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_train_test_split
from stroke_bci_mvp.online import calibrate_trigger_threshold, simulate_session
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def calibrate(config_path: str, thresholds: list[float] | None = None, max_false_trigger_rate: float | None = None) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
    calibration_cfg = config.get("calibration", {})
    thresholds = thresholds or [float(value) for value in calibration_cfg.get("thresholds", _default_thresholds())]
    max_false_trigger_rate = float(
        calibration_cfg.get("max_false_trigger_rate", 0.1)
        if max_false_trigger_rate is None
        else max_false_trigger_rate
    )

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
    valid_subject_ids = dataset.subject_ids[[result.valid for result in quality_results]]
    split = make_train_test_split(y_valid, valid_subject_ids, config)

    result = calibrate_trigger_threshold(
        model=bundle["model"],
        X=X_valid[split.train_idx],
        y=y_valid[split.train_idx],
        sfreq=dataset.sfreq,
        ch_names=dataset.ch_names,
        config=config,
        thresholds=thresholds,
        max_false_trigger_rate=max_false_trigger_rate,
    )
    heldout_config = json.loads(json.dumps(config))
    heldout_config["online"]["trigger_threshold"] = result["selected_threshold"]
    heldout_report = simulate_session(
        model=bundle["model"],
        X=X_valid[split.test_idx],
        y=y_valid[split.test_idx],
        sfreq=dataset.sfreq,
        ch_names=dataset.ch_names,
        config=heldout_config,
    )
    result["heldout_test_metrics_at_selected_threshold"] = {
        "n_trials": heldout_report["n_trials"],
        "true_intention_trials": heldout_report["true_intention_trials"],
        "rest_trials": heldout_report["rest_trials"],
        "trigger_rate": heldout_report["trigger_rate"],
        "false_trigger_rate": heldout_report["false_trigger_rate"],
        "mean_trigger_delay_seconds": heldout_report["mean_trigger_delay_seconds"],
        "triggered_intention_trials": heldout_report["triggered_intention_trials"],
        "triggered_rest_trials": heldout_report["triggered_rest_trials"],
    }
    result["metadata"] = {
        "dataset": config["dataset"]["name"],
        "config_path": str(Path(config_path)),
        "model_path": outputs["model_path"],
        "calibration_split": split.strategy,
        "calibration_epochs": int(len(split.train_idx)),
        "heldout_test_subject_ids": split.test_subject_ids,
        "training_mode": bundle.get("training_mode", config["model"].get("training_mode", "epoch")),
    }

    output_path = Path(outputs.get("threshold_calibration_json", Path(outputs["dir"]) / "threshold_calibration.json"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _default_thresholds() -> list[float]:
    return [round(value, 2) for value in [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]]


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate BU100 trigger threshold on the training split.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float)
    parser.add_argument("--max-false-trigger-rate", type=float)
    args = parser.parse_args()

    result = calibrate(args.config, args.thresholds, args.max_false_trigger_rate)
    compact = {key: value for key, value in result.items() if key != "candidates"}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
