# Simulated BU100 BCI Session Report

## Provenance

- Dataset: physionet
- Config: configs/v2/physionet_riemannian.yaml
- Model: outputs/v2_physionet_riemannian/model.joblib
- Training mode: online_windows
- Split strategy: subject
- Test subjects: S003

## Summary

- Trials: 7
- Motor intention trials: 4
- Rest trials: 3
- Effective trigger rate: 0.0%
- False trigger rate: 0.0%
- Mean trigger delay: N/A

## Decision Reasons

- low_intention_probability: 91

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
