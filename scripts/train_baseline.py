from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def train(config_path: str) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
    output_dir = Path(outputs["dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_dataset(config)
    X = notch_epochs(dataset.X, dataset.sfreq, config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, quality_results = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        config["quality"],
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=float(config["model"].get("test_size", 0.25)),
        random_state=int(config["dataset"].get("random_state", 7)),
        stratify=y_valid,
    )

    model = build_model(config, dataset.sfreq)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    metrics = {
        "dataset": config["dataset"]["name"],
        "config_path": str(Path(config_path)),
        "n_epochs_total": int(len(dataset.y)),
        "n_epochs_valid": int(len(y_valid)),
        "rejected_epochs": int(len(dataset.y) - len(y_valid)),
        "quality_reject_rate": float(1.0 - len(y_valid) / max(1, len(dataset.y))),
        "train_epochs": int(len(y_train)),
        "test_epochs": int(len(y_test)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "quality_score_mean": float(np.mean([result.score for result in quality_results])),
    }

    model_bundle = {
        "model": model,
        "sfreq": dataset.sfreq,
        "ch_names": dataset.ch_names,
        "label_names": dataset.label_names,
        "config": config,
    }
    joblib.dump(model_bundle, outputs["model_path"])
    Path(outputs["offline_metrics_path"]).write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the BCI MVP baseline decoder.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    metrics = train(args.config)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
