from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import mne
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from stroke_bci_mvp.datasets.figshare_stroke import _infer_subject_id


def audit_figshare_events(
    root: str,
    events_path: str,
    output_path: str,
    epoch_seconds: float = 2.0,
    subjects: list[str] | None = None,
) -> dict:
    root_path = Path(root)
    event_rows = _read_events(Path(events_path))
    event_summary = _summarize_events(event_rows)

    edf_files = sorted(root_path.rglob("*.edf"))
    if subjects:
        subject_set = set(subjects)
        edf_files = [path for path in edf_files if _infer_subject_id(path) in subject_set]

    subject_rows = []
    for edf_path in edf_files:
        raw = mne.io.read_raw_edf(edf_path, preload=False, verbose="ERROR")
        duration_seconds = raw.n_times / raw.info["sfreq"]
        events_in_bounds = _events_in_bounds(event_rows, raw, epoch_seconds)
        loader_candidate_count = _loader_candidate_count(event_rows)
        nominal_overrun_count = _nominal_overrun_count(event_rows, duration_seconds, epoch_seconds)
        subject_rows.append(
            {
                "subject_id": _infer_subject_id(edf_path),
                "edf_path": str(edf_path),
                "sfreq": float(raw.info["sfreq"]),
                "n_times": int(raw.n_times),
                "duration_seconds": float(duration_seconds),
                "annotation_count": int(len(raw.annotations)),
                "shared_events_total": int(len(event_rows)),
                "current_loader_candidate_events": int(loader_candidate_count),
                "events_in_bounds": int(len(events_in_bounds)),
                "events_out_of_bounds": int(loader_candidate_count - len(events_in_bounds)),
                "nominal_epoch_stop_overruns": int(nominal_overrun_count),
                "current_loader_epochs": _current_loader_epoch_counts(events_in_bounds),
                "last_event_stop_seconds": _last_event_stop_seconds(events_in_bounds, epoch_seconds),
            }
        )

    result = {
        "metadata": {
            "root": str(root_path),
            "events_path": str(Path(events_path)),
            "epoch_seconds": float(epoch_seconds),
            "audited_edf_count": int(len(edf_files)),
        },
        "shared_events": event_summary,
        "edf_summary": _summarize_edfs(subject_rows),
        "subjects": subject_rows,
        "findings": _findings(event_summary, subject_rows),
    }

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def _read_events(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            rows.append(
                {
                    "onset_ms": float(row["onset"]),
                    "onset_seconds": float(row["onset"]) / 1000.0,
                    "duration_ms": float(row["duration"]),
                    "duration_seconds": float(row["duration"]) / 1000.0,
                    "trial_type": str(row.get("trial_type", "")).strip(),
                    "value": str(row.get("value", "")).strip(),
                    "stim_file": str(row.get("stim_file", "")).strip(),
                }
            )
    return rows


def _summarize_events(rows: list[dict]) -> dict:
    value_counts = Counter(row["value"] for row in rows)
    trial_type_counts = Counter(row["trial_type"] for row in rows)
    stim_counts = Counter(row["stim_file"] for row in rows)
    current_labels = [_current_label(row) for row in rows]
    loader_rows = [row for row, label in zip(rows, current_labels) if label is not None]
    label_counts = Counter(label for label in current_labels if label is not None)
    return {
        "n_rows": int(len(rows)),
        "value_counts": dict(value_counts),
        "trial_type_counts": dict(trial_type_counts),
        "stim_file_counts": dict(stim_counts),
        "first_onset_seconds": float(rows[0]["onset_seconds"]) if rows else None,
        "last_onset_seconds": float(rows[-1]["onset_seconds"]) if rows else None,
        "last_declared_stop_seconds": float(rows[-1]["onset_seconds"] + rows[-1]["duration_seconds"]) if rows else None,
        "current_loader_used_rows": int(len(loader_rows)),
        "current_loader_label_counts": {str(k): int(v) for k, v in label_counts.items()},
        "current_loader_first_onset_seconds": float(loader_rows[0]["onset_seconds"]) if loader_rows else None,
        "current_loader_last_onset_seconds": float(loader_rows[-1]["onset_seconds"]) if loader_rows else None,
    }


def _current_label(row: dict) -> int | None:
    if row["value"] == "2":
        return 1
    if row["value"] == "3":
        return 0
    return None


def _loader_candidate_count(rows: list[dict]) -> int:
    return sum(1 for row in rows if _current_label(row) is not None)


def _events_in_bounds(rows: list[dict], raw: mne.io.BaseRaw, epoch_seconds: float) -> list[dict]:
    used = []
    n_samples = int(round(epoch_seconds * raw.info["sfreq"]))
    for row in rows:
        label = _current_label(row)
        if label is None:
            continue
        start = raw.time_as_index(float(row["onset_seconds"]))[0]
        stop = start + n_samples
        if stop <= raw.n_times:
            used.append(row | {"label": label})
    return used


def _nominal_overrun_count(rows: list[dict], duration_seconds: float, epoch_seconds: float) -> int:
    count = 0
    for row in rows:
        if _current_label(row) is not None and row["onset_seconds"] + epoch_seconds > duration_seconds:
            count += 1
    return count


def _current_loader_epoch_counts(rows: list[dict]) -> dict:
    counts = Counter(row["label"] for row in rows)
    return {"rest": int(counts.get(0, 0)), "motor_intention": int(counts.get(1, 0))}


def _last_event_stop_seconds(rows: list[dict], epoch_seconds: float) -> float | None:
    if not rows:
        return None
    return float(max(row["onset_seconds"] + epoch_seconds for row in rows))


def _summarize_edfs(subject_rows: list[dict]) -> dict:
    if not subject_rows:
        return {}
    durations = np.asarray([row["duration_seconds"] for row in subject_rows], dtype=float)
    annotation_counts = Counter(row["annotation_count"] for row in subject_rows)
    events_in_bounds = Counter(row["events_in_bounds"] for row in subject_rows)
    rest_counts = Counter(row["current_loader_epochs"]["rest"] for row in subject_rows)
    motor_counts = Counter(row["current_loader_epochs"]["motor_intention"] for row in subject_rows)
    nominal_overruns = Counter(row["nominal_epoch_stop_overruns"] for row in subject_rows)
    return {
        "duration_seconds_min": float(np.min(durations)),
        "duration_seconds_max": float(np.max(durations)),
        "annotation_count_distribution": {str(k): int(v) for k, v in annotation_counts.items()},
        "events_in_bounds_distribution": {str(k): int(v) for k, v in events_in_bounds.items()},
        "nominal_epoch_stop_overrun_distribution": {str(k): int(v) for k, v in nominal_overruns.items()},
        "rest_epoch_count_distribution": {str(k): int(v) for k, v in rest_counts.items()},
        "motor_epoch_count_distribution": {str(k): int(v) for k, v in motor_counts.items()},
    }


def _findings(event_summary: dict, subject_rows: list[dict]) -> list[str]:
    findings = []
    if subject_rows and all(row["annotation_count"] == 0 for row in subject_rows):
        findings.append("All audited EDF files have zero embedded annotations; the loader relies on the shared events TSV.")
    if event_summary.get("current_loader_label_counts") == {"1": 40, "0": 40}:
        findings.append("Current loader maps value=2 to motor_intention and value=3 to rest, producing 40/40 epochs per full-length EDF.")
    if event_summary.get("value_counts", {}).get("1"):
        findings.append("Events with value=1 are instruction markers and are intentionally ignored by the current loader.")
    out_of_bounds = [row for row in subject_rows if row["events_out_of_bounds"]]
    if out_of_bounds:
        findings.append(f"{len(out_of_bounds)} EDF files have shared events that exceed the recording duration for the configured epoch length.")
    else:
        findings.append("All shared motor/rest events fit using the same sample-index logic as the current loader.")
    nominal_overruns = [row for row in subject_rows if row["nominal_epoch_stop_overruns"]]
    if nominal_overruns:
        findings.append(
            "The final shared rest event nominally ends about 1 ms after the EDF duration, "
            "but sample-index conversion keeps it in bounds for the current loader."
        )
    if subject_rows and len({row["events_in_bounds"] for row in subject_rows}) == 1:
        findings.append("All audited EDF files have the same number of usable shared events, so subject-level count imbalance is mostly from quality rejection, not event coverage.")
    return findings


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Figshare shared events against EDF recordings.")
    parser.add_argument("--root", default="data/raw/figshare_stroke/edffile")
    parser.add_argument("--events-path", default="data/raw/figshare_stroke/task-motor-imagery_events.tsv")
    parser.add_argument("--epoch-seconds", type=float, default=2.0)
    parser.add_argument("--subjects", nargs="+")
    parser.add_argument("--output", default="outputs/v2_figshare_stroke_full_riemannian/figshare_event_audit.json")
    args = parser.parse_args()

    result = audit_figshare_events(
        root=args.root,
        events_path=args.events_path,
        output_path=args.output,
        epoch_seconds=args.epoch_seconds,
        subjects=args.subjects,
    )
    compact = {
        "metadata": result["metadata"],
        "shared_events": result["shared_events"],
        "edf_summary": result["edf_summary"],
        "findings": result["findings"],
    }
    print(json.dumps(compact, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
