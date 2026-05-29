# Simulated BU100 BCI Session Report

## Provenance

- Dataset: physionet
- Config: configs\physionet.yaml
- Model: outputs/physionet/model.joblib

## Summary

- Trials: 37
- Motor intention trials: 19
- Rest trials: 18
- Effective trigger rate: 68.4%
- False trigger rate: 27.8%
- Mean trigger delay: 1.885 s

## Decision Reasons

- low_intention_probability: 289
- refractory_period: 98
- trigger_bu100_assist_open_hand: 18
- waiting_for_confirmation: 76

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
