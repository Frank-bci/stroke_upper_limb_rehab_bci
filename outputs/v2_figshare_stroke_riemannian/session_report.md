# Simulated BU100 BCI Session Report

## Provenance

- Dataset: figshare_stroke
- Config: configs/v2/figshare_stroke_riemannian.yaml
- Model: outputs/v2_figshare_stroke_riemannian/model.joblib
- Training mode: online_windows
- Split strategy: subject
- Test subjects: sub-04, sub-06

## Summary

- Trials: 130
- Motor intention trials: 62
- Rest trials: 68
- Effective trigger rate: 0.0%
- False trigger rate: 0.0%
- Mean trigger delay: N/A

## Decision Reasons

- low_intention_probability: 649
- low_signal_quality: 1

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
