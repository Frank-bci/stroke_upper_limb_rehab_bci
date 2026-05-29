# Simulated BU100 BCI Session Report

## Provenance

- Dataset: figshare_stroke
- Config: configs\figshare_stroke.yaml
- Model: outputs/figshare_stroke/model.joblib

## Summary

- Trials: 112
- Motor intention trials: 55
- Rest trials: 57
- Effective trigger rate: 29.1%
- False trigger rate: 12.3%
- Mean trigger delay: 1.047 s

## Decision Reasons

- low_intention_probability: 380
- low_signal_quality: 4
- refractory_period: 20
- trigger_bu100_assist_open_hand: 23
- waiting_for_confirmation: 133

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
