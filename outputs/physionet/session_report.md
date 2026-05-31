# Simulated BU100 BCI Session Report

## Provenance

- Dataset: physionet
- Config: configs/physionet.yaml
- Model: outputs/physionet/model.joblib
- Training mode: online_windows
- Split strategy: subject
- Test subjects: S003

## Summary

- Trials: 7
- Motor intention trials: 4
- Rest trials: 3
- Effective trigger rate: 50.0%
- False trigger rate: 0.0%
- Mean trigger delay: 2.000 s

## Decision Reasons

- low_intention_probability: 55
- refractory_period: 8
- trigger_bu100_assist_open_hand: 2
- waiting_for_confirmation: 26

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
