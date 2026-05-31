from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.online import simulate_session
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def evaluate_subject_report(
    config_path: str,
    output_path: str | None = None,
    use_calibrated_threshold: bool = True,
    threshold: float | None = None,
) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
    bundle = joblib.load(outputs["model_path"])
    model = bundle["model"]

    if use_calibrated_threshold:
        _apply_calibrated_threshold(config, outputs)
    if threshold is not None:
        config["online"]["trigger_threshold"] = float(threshold)

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
    for subject_id in sorted(map(str, np.unique(valid_subject_ids[split.test_idx]))):
        subject_mask = valid_subject_ids[split.test_idx].astype(str) == subject_id
        epoch_idx = split.test_idx[subject_mask]
        X_subject = X_valid[epoch_idx]
        y_subject = y_valid[epoch_idx]
        rows.append(
            {
                "subject_id": subject_id,
                "n_epochs": int(len(y_subject)),
                "label_counts": {
                    "rest": int(np.sum(y_subject == 0)),
                    "motor_intention": int(np.sum(y_subject == 1)),
                },
                "offline": _offline_metrics(model, X_subject, y_subject, dataset.sfreq, config),
                "session": _compact_session(
                    simulate_session(
                        model=model,
                        X=X_subject,
                        y=y_subject,
                        sfreq=dataset.sfreq,
                        ch_names=dataset.ch_names,
                        config=config,
                    )
                ),
            }
        )

    report = {
        "metadata": {
            "dataset": config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "model_path": outputs["model_path"],
            "threshold": float(config["online"]["trigger_threshold"]),
            "threshold_source": "override" if threshold is not None else ("calibrated" if use_calibrated_threshold else "config"),
            "split_strategy": split.strategy,
            "test_subject_ids": split.test_subject_ids,
        },
        "subjects": rows,
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "subject_report.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return report


def _apply_calibrated_threshold(config: dict, outputs: dict) -> None:
    path = Path(outputs.get("threshold_calibration_json", Path(outputs["dir"]) / "threshold_calibration.json"))
    if not path.exists():
        return
    calibration = json.loads(path.read_text(encoding="utf-8"))
    if "selected_threshold" in calibration:
        config["online"]["trigger_threshold"] = float(calibration["selected_threshold"])


def _offline_metrics(model, X: np.ndarray, y: np.ndarray, sfreq: float, config: dict) -> dict:
    if len(np.unique(y)) < 2:
        return {"skipped": True, "reason": "single_class_subject"}

    training_mode = str(config.get("model", {}).get("training_mode", "epoch")).lower()
    if training_mode == "online_windows":
        windows = make_online_windows(X, y, sfreq, config)
        X_eval, y_eval = windows.X, windows.y
    elif training_mode == "epoch":
        X_eval, y_eval = X, y
    else:
        raise ValueError(f"Unsupported training_mode: {training_mode}")

    y_pred = model.predict(X_eval)
    y_score = model.predict_proba(X_eval)[:, 1]
    return {
        "skipped": False,
        "n_samples": int(len(y_eval)),
        "balanced_accuracy": float(balanced_accuracy_score(y_eval, y_pred)),
        "auc": float(roc_auc_score(y_eval, y_score)),
        "f1": float(f1_score(y_eval, y_pred)),
        "confusion_matrix": confusion_matrix(y_eval, y_pred).tolist(),
    }


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate held-out test subjects one by one.")
    parser.add_argument("--config", default="configs/v2/figshare_stroke_riemannian.yaml")
    parser.add_argument("--output")
    parser.add_argument("--no-calibrated-threshold", action="store_true")
    parser.add_argument("--threshold", type=float)
    args = parser.parse_args()

    report = evaluate_subject_report(
        config_path=args.config,
        output_path=args.output,
        use_calibrated_threshold=not args.no_calibrated_threshold,
        threshold=args.threshold,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
