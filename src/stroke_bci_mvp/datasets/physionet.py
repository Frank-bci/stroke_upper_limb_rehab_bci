from __future__ import annotations

from pathlib import Path

import mne
import numpy as np

from stroke_bci_mvp.types import EpochDataset


def load_physionet_eegmmi(config: dict) -> EpochDataset:
    """Load PhysioNet EEG Motor Movement/Imagery runs through MNE.

    The loader maps T1/T2 motor-imagery annotations to label 1 and T0 rest
    annotations to label 0, matching the MVP's Rest vs Motor Intention target.
    """

    subjects = [int(subject) for subject in config.get("subjects", [1])]
    runs = [int(run) for run in config.get("runs", [4, 8, 12])]
    data_path = Path(config.get("path", "data/raw/physionet"))
    epoch_seconds = float(config.get("epoch_seconds", 4.0))
    sfreq_target = float(config.get("sfreq", 160))
    max_epochs = config.get("max_epochs")

    from mne.datasets import eegbci

    all_epochs: list[np.ndarray] = []
    all_labels: list[int] = []
    all_subject_ids: list[str] = []
    ch_names: list[str] | None = None

    for subject in subjects:
        files = eegbci.load_data(subject, runs, path=data_path, update_path=True)
        raws = []
        for file in files:
            raw = mne.io.read_raw_edf(file, preload=True, verbose="ERROR")
            eegbci.standardize(raw)
            raw.pick("eeg")
            raw.resample(sfreq_target, verbose="ERROR")
            raws.append(raw)
        raw = mne.concatenate_raws(raws, verbose="ERROR")
        if ch_names is None:
            ch_names = list(raw.ch_names)

        n_samples = int(round(epoch_seconds * raw.info["sfreq"]))
        for annotation in raw.annotations:
            if annotation["description"] not in {"T0", "T1", "T2"}:
                continue
            start = raw.time_as_index(float(annotation["onset"]))[0]
            stop = start + n_samples
            if stop > raw.n_times:
                continue
            epoch = raw.get_data(start=start, stop=stop) * 1e6
            label = 0 if annotation["description"] == "T0" else 1
            all_epochs.append(epoch)
            all_labels.append(label)
            all_subject_ids.append(f"S{subject:03d}")
            if max_epochs and len(all_epochs) >= int(max_epochs):
                break
        if max_epochs and len(all_epochs) >= int(max_epochs):
            break

    if not all_epochs:
        raise RuntimeError("No PhysioNet EEGMMI epochs were loaded. Check subjects, runs, and data path.")

    return EpochDataset(
        X=np.asarray(all_epochs),
        y=np.asarray(all_labels, dtype=int),
        sfreq=float(sfreq_target),
        ch_names=ch_names or [],
        subject_ids=np.asarray(all_subject_ids),
        label_names={0: "rest", 1: "motor_intention"},
    )
