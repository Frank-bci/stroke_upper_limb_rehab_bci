from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def write_session_report(report: dict[str, Any], json_path: str | Path, md_path: str | Path) -> None:
    json_path = Path(json_path)
    md_path = Path(md_path)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    delay = report.get("mean_trigger_delay_seconds")
    delay_text = "N/A" if delay is None else f"{delay:.3f} s"
    metadata = report.get("metadata", {})
    md = f"""# Simulated BU100 BCI Session Report

## Provenance

- Dataset: {metadata.get("dataset", "unknown")}
- Config: {metadata.get("config_path", "unknown")}
- Model: {metadata.get("model_path", "unknown")}
- Training mode: {metadata.get("training_mode", "unknown")}
- Split strategy: {metadata.get("split_strategy", "unknown")}
- Test subjects: {", ".join(map(str, metadata.get("test_subject_ids", []))) or "unknown"}

## Summary

- Trials: {report["n_trials"]}
- Motor intention trials: {report["true_intention_trials"]}
- Rest trials: {report["rest_trials"]}
- Effective trigger rate: {report["trigger_rate"]:.1%}
- False trigger rate: {report["false_trigger_rate"]:.1%}
- Mean trigger delay: {delay_text}

## Decision Reasons

"""
    for reason, count in sorted(report.get("decision_reasons", {}).items(), key=lambda item: item[0]):
        md += f"- {reason}: {count}\n"

    md += """
## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
"""
    md_path.write_text(md, encoding="utf-8")
