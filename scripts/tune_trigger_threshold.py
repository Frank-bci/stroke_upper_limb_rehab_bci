from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import joblib
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.config import load_config
from stroke_bci_mvp.datasets import load_dataset
from stroke_bci_mvp.online import simulate_session
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan trigger thresholds for the pseudo-online controller.")
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--thresholds", nargs="+", type=float, default=[0.7, 0.8, 0.85, 0.9, 0.95])
    args = parser.parse_args()

    config = load_config(args.config)
    bundle = joblib.load(config["outputs"]["model_path"])
    dataset = load_dataset(config)
    X = notch_epochs(dataset.X, dataset.sfreq, config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, _ = filter_valid_epochs(X, dataset.y, dataset.sfreq, dataset.ch_names, config["quality"])

    _, X_test, _, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=float(config["model"].get("test_size", 0.25)),
        random_state=int(config["dataset"].get("random_state", 7)),
        stratify=y_valid,
    )

    rows = []
    for threshold in args.thresholds:
        run_config = copy.deepcopy(config)
        run_config["online"]["trigger_threshold"] = threshold
        report = simulate_session(bundle["model"], X_test, y_test, dataset.sfreq, dataset.ch_names, run_config)
        rows.append(
            {
                "threshold": threshold,
                "trigger_rate": report["trigger_rate"],
                "false_trigger_rate": report["false_trigger_rate"],
                "mean_trigger_delay_seconds": report["mean_trigger_delay_seconds"],
            }
        )

    print(json.dumps(rows, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

