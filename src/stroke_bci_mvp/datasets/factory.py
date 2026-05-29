from __future__ import annotations

from stroke_bci_mvp.datasets.figshare_stroke import load_figshare_stroke
from stroke_bci_mvp.datasets.physionet import load_physionet_eegmmi
from stroke_bci_mvp.datasets.synthetic import make_synthetic_dataset
from stroke_bci_mvp.types import EpochDataset


def load_dataset(config: dict) -> EpochDataset:
    dataset_cfg = config["dataset"]
    name = dataset_cfg.get("name", "synthetic").lower()

    if name == "synthetic":
        return make_synthetic_dataset(dataset_cfg)

    if name == "physionet":
        return load_physionet_eegmmi(dataset_cfg)

    if name == "figshare_stroke":
        return load_figshare_stroke(dataset_cfg)

    raise ValueError(
        f"Unsupported dataset '{name}'. The MVP currently ships with 'synthetic', 'physionet', and 'figshare_stroke'; "
        "add public dataset adapters under src/stroke_bci_mvp/datasets/."
    )
