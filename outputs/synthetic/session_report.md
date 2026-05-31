# Simulated BU100 BCI Session Report

## Provenance

- Dataset: synthetic
- Config: configs/default.yaml
- Model: outputs/synthetic/model.joblib
- Training mode: online_windows
- Split strategy: random
- Test subjects: S01, S02, S03, S04, S05, S06

## Summary

- Trials: 90
- Motor intention trials: 45
- Rest trials: 45
- Effective trigger rate: 64.4%
- False trigger rate: 4.4%
- Mean trigger delay: 1.534 s

## Decision Reasons

- low_intention_probability: 742
- refractory_period: 185
- trigger_bu100_assist_open_hand: 31
- waiting_for_confirmation: 212

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
