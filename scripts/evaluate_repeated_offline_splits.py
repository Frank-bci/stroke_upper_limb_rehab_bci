from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score, roc_auc_score

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def evaluate_repeated_offline_splits(
    config_path: str,
    seeds: list[int],
    output_path: str | None = None,
) -> dict:
    base_config = load_config(config_path)
    outputs = base_config["outputs"]

    dataset = load_dataset(base_config)
    X = notch_epochs(dataset.X, dataset.sfreq, base_config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, quality_results = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        base_config["quality"],
    )
    valid_mask = np.asarray([result.valid for result in quality_results], dtype=bool)
    valid_subject_ids = dataset.subject_ids[valid_mask]

    runs = []
    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["dataset"]["random_state"] = int(seed)
        split = make_train_test_split(y_valid, valid_subject_ids, config)

        train_data = _training_samples(X_valid[split.train_idx], y_valid[split.train_idx], dataset.sfreq, config)
        test_data = _training_samples(X_valid[split.test_idx], y_valid[split.test_idx], dataset.sfreq, config)

        model = build_model(config, dataset.sfreq)
        model.fit(train_data["X"], train_data["y"])
        offline = _offline_metrics(model, test_data["X"], test_data["y"])

        runs.append(
            {
                "seed": int(seed),
                "test_subject_ids": split.test_subject_ids,
                "train_epochs": int(len(split.train_idx)),
                "test_epochs": int(len(split.test_idx)),
                "train_samples": int(len(train_data["y"])),
                "test_samples": int(len(test_data["y"])),
                "offline": offline,
            }
        )

    result = {
        "metadata": {
            "dataset": base_config["dataset"]["name"],
            "task": base_config["dataset"].get("task", "rest_vs_mi"),
            "config_path": str(Path(config_path)),
            "seeds": seeds,
            "split_strategy": base_config.get("model", {}).get("split_strategy"),
            "training_mode": base_config.get("model", {}).get("training_mode", "epoch"),
            "n_epochs_total": int(len(dataset.y)),
            "n_epochs_valid": int(len(y_valid)),
            "quality_reject_rate": float(1.0 - len(y_valid) / max(1, len(dataset.y))),
            "label_names": {str(key): value for key, value in dataset.label_names.items()},
        },
        "summary": _summary(runs),
        "runs": runs,
    }

    if output_path is None:
        output_path = str(Path(outputs["dir"]) / "repeated_offline_splits.json")
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _training_samples(X, y, sfreq: float, config: dict) -> dict:
    training_mode = str(config.get("model", {}).get("training_mode", "epoch")).lower()
    if training_mode == "online_windows":
        windows = make_online_windows(X, y, sfreq, config)
        return {"X": windows.X, "y": windows.y}
    if training_mode == "epoch":
        return {"X": X, "y": y}
    raise ValueError(f"Unsupported training_mode: {training_mode}")


def _offline_metrics(model, X_test, y_test) -> dict:
    y_pred = model.predict(X_test)
    y_score = model.predict_proba(X_test)[:, 1]
    return {
        "balanced_accuracy": float(balanced_accuracy_score(y_test, y_pred)),
        "auc": float(roc_auc_score(y_test, y_score)),
        "f1": float(f1_score(y_test, y_pred)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }


def _summary(runs: list[dict]) -> dict:
    return {
        "offline_balanced_accuracy": _mean_std([run["offline"]["balanced_accuracy"] for run in runs]),
        "offline_auc": _mean_std([run["offline"]["auc"] for run in runs]),
        "offline_f1": _mean_std([run["offline"]["f1"] for run in runs]),
    }


def _mean_std(values: list[float]) -> dict:
    arr = np.asarray(values, dtype=float)
    return {
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr, ddof=0)),
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate repeated offline train/test splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27, 37, 47])
    parser.add_argument("--output")
    args = parser.parse_args()

    result = evaluate_repeated_offline_splits(
        config_path=args.config,
        seeds=args.seeds,
        output_path=args.output,
    )
    print(json.dumps({"metadata": result["metadata"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
