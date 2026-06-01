from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.evaluation import make_online_windows, make_train_test_split
from stroke_bci_mvp.models import build_model
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def audit_channel_selection(config_path: str, seeds: list[int], output_path: str | None = None) -> dict:
    base_config = load_config(config_path)
    dataset = load_dataset(base_config)
    X = notch_epochs(dataset.X, dataset.sfreq, base_config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, quality_results = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        base_config["quality"],
    )
    valid_subject_ids = dataset.subject_ids[np.asarray([result.valid for result in quality_results], dtype=bool)]

    runs = []
    channel_counts: Counter[str] = Counter()
    risk_counts: Counter[str] = Counter()
    total_selectors = 0
    total_selected_slots = 0
    risk_channels = set(map(str, base_config.get("model", {}).get("channel_selection", {}).get("risk_channels", [])))
    if not risk_channels:
        risk_channels = {"HEOL", "HEOR", ""}

    for seed in seeds:
        config = copy.deepcopy(base_config)
        config["dataset"]["random_state"] = int(seed)
        split = make_train_test_split(y_valid, valid_subject_ids, config)
        train_data = _training_samples(X_valid[split.train_idx], y_valid[split.train_idx], dataset.sfreq, config)

        model = build_model(config, dataset.sfreq, ch_names=dataset.ch_names)
        model.fit(train_data["X"], train_data["y"])
        selectors = _extract_selected_channels(model, dataset.ch_names)
        total_selectors += len(selectors)

        run_rows = []
        for selector_idx, channels in enumerate(selectors, start=1):
            total_selected_slots += len(channels)
            channel_counts.update(channels)
            selected_risk = [channel for channel in channels if channel in risk_channels]
            risk_counts.update(selected_risk)
            run_rows.append(
                {
                    "selector_idx": int(selector_idx),
                    "selected_channels": channels,
                    "risk_channels": selected_risk,
                }
            )

        runs.append(
            {
                "seed": int(seed),
                "test_subject_ids": split.test_subject_ids,
                "selectors": run_rows,
            }
        )

    result = {
        "metadata": {
            "dataset": base_config["dataset"]["name"],
            "config_path": str(Path(config_path)),
            "seeds": seeds,
            "risk_channels": sorted(risk_channels),
            "n_total_channels": int(len(dataset.ch_names)),
            "ch_names": dataset.ch_names,
            "n_epochs_total": int(len(dataset.y)),
            "n_epochs_valid": int(len(y_valid)),
            "quality_reject_rate": float(1.0 - len(y_valid) / max(1, len(dataset.y))),
        },
        "summary": {
            "total_selectors": int(total_selectors),
            "total_selected_slots": int(total_selected_slots),
            "risk_channel_selected_slots": int(sum(risk_counts.values())),
            "risk_channel_selected_slot_rate": float(sum(risk_counts.values()) / max(1, total_selected_slots)),
            "risk_channel_counts": dict(sorted(risk_counts.items())),
            "top_selected_channels": [
                {"channel": channel, "count": int(count), "selector_rate": float(count / max(1, total_selectors))}
                for channel, count in channel_counts.most_common()
            ],
        },
        "runs": runs,
    }

    if output_path is None:
        output_path = str(Path(base_config["outputs"]["dir"]) / "channel_selection_audit.json")
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


def _extract_selected_channels(model, ch_names: list[str]) -> list[list[str]]:
    rows = []
    estimators = []
    if hasattr(model, "calibrated_classifiers_"):
        estimators = [
            getattr(calibrated, "estimator", None)
            for calibrated in model.calibrated_classifiers_
        ]
    else:
        estimators = [model]

    for estimator in estimators:
        if estimator is None or not hasattr(estimator, "named_steps"):
            continue
        selector = estimator.named_steps.get("channel_selector")
        if selector is None:
            continue
        indices = getattr(selector, "selected_indices_", getattr(selector, "indices", []))
        rows.append([str(ch_names[int(idx)]) for idx in indices])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit model channel-selection frequency across repeated splits.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--output")
    args = parser.parse_args()

    result = audit_channel_selection(config_path=args.config, seeds=args.seeds, output_path=args.output)
    print(json.dumps({"metadata": result["metadata"], "summary": result["summary"]}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
