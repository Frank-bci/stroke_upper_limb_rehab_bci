from __future__ import annotations

import argparse
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
from stroke_bci_mvp.reports import write_session_report
from stroke_bci_mvp.signal import filter_valid_epochs, notch_epochs


def simulate(config_path: str) -> dict:
    config = load_config(config_path)
    outputs = config["outputs"]
    bundle = joblib.load(outputs["model_path"])
    model = bundle["model"]

    dataset = load_dataset(config)
    X = notch_epochs(dataset.X, dataset.sfreq, config["preprocessing"].get("notch_hz"))
    X_valid, y_valid, _ = filter_valid_epochs(
        X,
        dataset.y,
        dataset.sfreq,
        dataset.ch_names,
        config["quality"],
    )

    _, X_test, _, y_test = train_test_split(
        X_valid,
        y_valid,
        test_size=float(config["model"].get("test_size", 0.25)),
        random_state=int(config["dataset"].get("random_state", 7)),
        stratify=y_valid,
    )

    report = simulate_session(
        model=model,
        X=X_test,
        y=y_test,
        sfreq=dataset.sfreq,
        ch_names=dataset.ch_names,
        config=config,
    )
    report["metadata"] = {
        "dataset": config["dataset"]["name"],
        "config_path": str(Path(config_path)),
        "model_path": outputs["model_path"],
        "label_mapping": bundle.get("label_names", {}),
    }
    write_session_report(report, outputs["session_report_json"], outputs["session_report_md"])
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pseudo-online BU100 trigger simulation.")
    parser.add_argument("--config", default="configs/default.yaml")
    args = parser.parse_args()

    report = simulate(args.config)
    compact = {key: value for key, value in report.items() if key != "trials"}
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
