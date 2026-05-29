from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.model_selection import LeaveOneGroupOut

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.online import simulate_session
from stroke_bci_mvp.signal import assess_epoch_quality, notch_epochs


def _valid_mask(X: np.ndarray, sfreq: float, ch_names: list[str], quality_cfg: dict) -> tuple[np.ndarray, list[float]]:
    results = [assess_epoch_quality(epoch, sfreq, ch_names, quality_cfg) for epoch in X]
    return np.asarray([result.valid for result in results], dtype=bool), [result.score for result in results]


def evaluate(config_path: str) -> dict:
    config = load_config(config_path)
    dataset = load_dataset(config)
    X = notch_epochs(dataset.X, dataset.sfreq, config["preprocessing"].get("notch_hz"))
    keep, quality_scores = _valid_mask(X, dataset.sfreq, dataset.ch_names, config["quality"])
    X = X[keep]
    y = dataset.y[keep]
    groups = dataset.subject_ids[keep]

    logo = LeaveOneGroupOut()
    fold_reports = []
    y_true_all = []
    y_pred_all = []
    y_score_all = []
    online_trigger_rates = []
    online_false_trigger_rates = []
    online_delays = []

    for fold_idx, (train_idx, test_idx) in enumerate(logo.split(X, y, groups), start=1):
        heldout_subject = str(groups[test_idx][0])
        if len(np.unique(y[train_idx])) < 2 or len(np.unique(y[test_idx])) < 2:
            fold_reports.append(
                {
                    "fold": fold_idx,
                    "heldout_subject": heldout_subject,
                    "skipped": True,
                    "reason": "train_or_test_fold_has_single_class",
                }
            )
            continue

        model = build_model(config, dataset.sfreq)
        model.fit(X[train_idx], y[train_idx])
        y_pred = model.predict(X[test_idx])
        y_score = model.predict_proba(X[test_idx])[:, 1]
        online_report = simulate_session(
            model=model,
            X=X[test_idx],
            y=y[test_idx],
            sfreq=dataset.sfreq,
            ch_names=dataset.ch_names,
            config=config,
        )

        fold = {
            "fold": fold_idx,
            "heldout_subject": heldout_subject,
            "train_epochs": int(len(train_idx)),
            "test_epochs": int(len(test_idx)),
            "balanced_accuracy": float(balanced_accuracy_score(y[test_idx], y_pred)),
            "auc": float(roc_auc_score(y[test_idx], y_score)),
            "f1": float(f1_score(y[test_idx], y_pred)),
            "confusion_matrix": confusion_matrix(y[test_idx], y_pred).tolist(),
            "online_trigger_rate": float(online_report["trigger_rate"]),
            "online_false_trigger_rate": float(online_report["false_trigger_rate"]),
            "online_mean_trigger_delay_seconds": online_report["mean_trigger_delay_seconds"],
            "skipped": False,
        }
        fold_reports.append(fold)
        y_true_all.extend(y[test_idx].tolist())
        y_pred_all.extend(y_pred.tolist())
        y_score_all.extend(y_score.tolist())
        online_trigger_rates.append(float(online_report["trigger_rate"]))
        online_false_trigger_rates.append(float(online_report["false_trigger_rate"]))
        if online_report["mean_trigger_delay_seconds"] is not None:
            online_delays.append(float(online_report["mean_trigger_delay_seconds"]))

    valid_folds = [fold for fold in fold_reports if not fold.get("skipped")]
    if not valid_folds:
        raise RuntimeError("No valid subject-level folds were available.")

    summary = {
        "dataset": config["dataset"]["name"],
        "config_path": str(Path(config_path)),
        "split": "leave_one_subject_out",
        "n_subjects": int(len(np.unique(groups))),
        "n_epochs_total": int(len(dataset.y)),
        "n_epochs_valid": int(len(y)),
        "rejected_epochs": int(len(dataset.y) - len(y)),
        "quality_reject_rate": float(1.0 - len(y) / max(1, len(dataset.y))),
        "quality_score_mean": float(np.mean(quality_scores)),
        "valid_folds": int(len(valid_folds)),
        "balanced_accuracy_mean": float(np.mean([fold["balanced_accuracy"] for fold in valid_folds])),
        "balanced_accuracy_std": float(np.std([fold["balanced_accuracy"] for fold in valid_folds])),
        "auc_mean": float(np.mean([fold["auc"] for fold in valid_folds])),
        "auc_std": float(np.std([fold["auc"] for fold in valid_folds])),
        "f1_mean": float(np.mean([fold["f1"] for fold in valid_folds])),
        "f1_std": float(np.std([fold["f1"] for fold in valid_folds])),
        "pooled_balanced_accuracy": float(balanced_accuracy_score(y_true_all, y_pred_all)),
        "pooled_auc": float(roc_auc_score(y_true_all, y_score_all)),
        "pooled_f1": float(f1_score(y_true_all, y_pred_all)),
        "pooled_confusion_matrix": confusion_matrix(y_true_all, y_pred_all).tolist(),
        "online_trigger_rate_mean": float(np.mean(online_trigger_rates)),
        "online_false_trigger_rate_mean": float(np.mean(online_false_trigger_rates)),
        "online_mean_trigger_delay_seconds": None if not online_delays else float(np.mean(online_delays)),
        "folds": fold_reports,
    }

    output_dir = Path(config["outputs"]["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "subject_generalization_metrics.json"
    output_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate leave-one-subject-out BCI generalization.")
    parser.add_argument("--config", default="configs/figshare_stroke.yaml")
    args = parser.parse_args()

    summary = evaluate(args.config)
    compact = {key: value for key, value in summary.items() if key != "folds"}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

