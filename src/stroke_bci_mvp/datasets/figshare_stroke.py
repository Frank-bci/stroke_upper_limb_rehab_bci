from __future__ import annotations

from pathlib import Path
import csv

import mne
import numpy as np

from stroke_bci_mvp.types import EpochDataset


def load_figshare_stroke(config: dict) -> EpochDataset:
    """Load locally extracted Figshare stroke MI EDF files.

    Expected source: Liu & Lv, "EEG datasets of stroke patients", figshare
    article 21679035. Download/extract `edffile.zip` into `dataset.path`, then
    point `configs/figshare_stroke.yaml` at that directory.
    """

    root = Path(config.get("path", "data/raw/figshare_stroke/edffile"))
    if not root.exists():
        raise FileNotFoundError(
            f"Figshare stroke path does not exist: {root}. "
            "Download/extract edffile.zip first, or run scripts/download_figshare_stroke.py --metadata-only."
        )

    edf_files = sorted(root.rglob("*.edf"))
    max_files = config.get("max_files")
    if max_files:
        edf_files = edf_files[: int(max_files)]
    if not edf_files:
        raise FileNotFoundError(f"No EDF files found under {root}")

    epoch_seconds = float(config.get("epoch_seconds", 4.0))
    sfreq_target = float(config.get("sfreq", 160))
    max_epochs = config.get("max_epochs")
    events_path = Path(config.get("events_path", "data/raw/figshare_stroke/task-motor-imagery_events.tsv"))
    shared_events = _read_figshare_events(events_path) if events_path.exists() else []

    all_epochs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_subject_ids: list[str] = []
    ch_names: list[str] | None = None

    for edf_path in edf_files:
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose="ERROR")
        raw.pick("eeg")
        raw.resample(sfreq_target, verbose="ERROR")
        if ch_names is None:
            ch_names = list(raw.ch_names)

        n_samples = int(round(epoch_seconds * raw.info["sfreq"]))
        subject_id = _infer_subject_id(edf_path)
        events = _events_from_annotations(raw)
        if not events and shared_events:
            events = shared_events

        for onset_seconds, label in events:
            start = raw.time_as_index(float(onset_seconds))[0]
            stop = start + n_samples
            if stop > raw.n_times:
                continue
            all_epochs.append(raw.get_data(start=start, stop=stop) * 1e6)
            all_labels.append(label)
            all_subject_ids.append(subject_id)
            if max_epochs and len(all_epochs) >= int(max_epochs):
                break
        if max_epochs and len(all_epochs) >= int(max_epochs):
            break

    if not all_epochs:
        raise RuntimeError(
            "No epochs could be built from EDF annotations or task-motor-imagery_events.tsv. "
            "Inspect the extracted Figshare files and event timing units."
        )

    return EpochDataset(
        X=np.asarray(all_epochs),
        y=np.asarray(all_labels, dtype=int),
        sfreq=sfreq_target,
        ch_names=ch_names or [],
        subject_ids=np.asarray(all_subject_ids),
        label_names={0: "rest", 1: "motor_intention"},
    )


def _infer_subject_id(path: Path) -> str:
    for part in path.parts:
        lowered = part.lower()
        if lowered.startswith("sub-") or lowered.startswith("subject"):
            return part
    return path.stem.split("_")[0]


def _map_annotation_to_label(description: str) -> int | None:
    desc = description.strip().lower()
    if desc in {"t0", "rest", "break", "baseline", "0"} or "rest" in desc:
        return 0
    motor_tokens = ("t1", "t2", "left", "right", "hand", "mi", "motor")
    if any(token in desc for token in motor_tokens):
        return 1
    return None


def _events_from_annotations(raw: mne.io.BaseRaw) -> list[tuple[float, int]]:
    events: list[tuple[float, int]] = []
    for annotation in raw.annotations:
        label = _map_annotation_to_label(str(annotation["description"]))
        if label is not None:
            events.append((float(annotation["onset"]), label))
    return events


def _read_figshare_events(path: Path) -> list[tuple[float, int]]:
    events: list[tuple[float, int]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            value = str(row.get("value", "")).strip()
            if value == "2":
                label = 1
            elif value == "3":
                label = 0
            else:
                continue
            onset_seconds = float(row["onset"]) / 1000.0
            events.append((onset_seconds, label))
    return events
