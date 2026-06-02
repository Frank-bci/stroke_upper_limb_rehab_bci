# WCDA Paper Notes

Paper reviewed: `WCDA(含DOI).pdf`, "Weighted Conditional Distribution Adaptation for Motor Imagery Classification", ICIG 2021.

## Useful Ideas

- The paper frames cross-subject MI decoding as source-domain to target-domain transfer learning. This matches our Figshare problem, where training subjects are source domains and held-out stroke subjects are target domains.
- It uses tangent-space covariance features, which is aligned with our current Riemannian pipeline.
- It first applies Euclidean Alignment to reduce marginal distribution shift, then focuses on conditional distribution adaptation. We tested a lightweight Euclidean covariance alignment variant and found it close to, but not better than, the current mainline.
- Its core WCDA idea is to reduce same-class source-target discrepancy while increasing different-class discrepancy. This is promising, but it requires pseudo-label iteration and a new projection objective, so it should not be dropped directly into the current trigger pipeline without a controlled offline prototype.
- Its TSSS source-sample selection idea is practical: remove low-margin source samples near a classifier boundary to reduce transfer noise and computational cost. This is a good future candidate after calibration-window validation.

## Project Decisions

- Do not replace the current v2 mainline with full WCDA yet. The current project needs clinically interpretable trigger behavior, false-trigger control, and repeatable subject-split validation before adding a more complex transfer projection.
- Keep the current recommended model as `configs/v2/figshare_stroke_full_riemannian_train_channel_standardized.yaml`.
- Adopt the paper's target-domain calibration principle first: use a small calibration window from the held-out subject/session to select a subject-specific threshold, then evaluate only on later trials.
- Keep Euclidean covariance alignment as a documented control, not as mainline, because it did not improve AUC or personalized trigger rate over the current baseline.

## Next Candidates

- Calibration-window threshold adaptation with finer threshold grids and conservative subject-level selection.
- TSSS-style source-sample selection on tangent-space features.
- Full WCDA projection as an offline-only prototype once source-sample selection and calibration-window thresholding are stable.
