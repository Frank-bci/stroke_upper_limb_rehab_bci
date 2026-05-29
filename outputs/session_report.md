# Simulated BU100 BCI Session Report

## Summary

- Trials: 90
- Motor intention trials: 45
- Rest trials: 45
- Effective trigger rate: 57.8%
- False trigger rate: 6.7%
- Mean trigger delay: 1.654 s

## Decision Reasons

- low_intention_probability: 772
- refractory_period: 159
- trigger_bu100_assist_open_hand: 29
- waiting_for_confirmation: 210

## Clinical Interpretation Draft

The simulated session estimates whether a patient-specific decoder can trigger
upper-limb assistance only when motor intention is detected and signal quality is
acceptable. For the next hardware-connected version, replace the simulated
trigger event with the BU100 command interface and keep the same safety gate.
