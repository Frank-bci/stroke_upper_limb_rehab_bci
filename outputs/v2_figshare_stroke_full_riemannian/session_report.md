# Simulated BU100 BCI Session Report

## Provenance

- Dataset: figshare_stroke
- Config: configs/v2/figshare_stroke_full_riemannian.yaml
- Model: outputs/v2_figshare_stroke_full_riemannian/model.joblib
- Training mode: online_windows
- Split strategy: subject
- Test subjects: sub-02, sub-11, sub-14, sub-16, sub-21, sub-23, sub-28, sub-31, sub-33, sub-35, sub-37, sub-43, sub-47

## Summary

- Trials: 964
- Motor intention trials: 468
- Rest trials: 496
- Effective trigger rate: 0.9%
- False trigger rate: 0.0%
- Mean trigger delay: 1.125 s

## Decision Reasons

- low_intention_probability: 4698
- low_signal_quality: 30
- refractory_period: 2
- trigger_bu100_assist_open_hand: 4
- waiting_for_confirmation: 86

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
