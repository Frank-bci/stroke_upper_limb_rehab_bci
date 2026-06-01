# Artifact Policy

This project keeps reproducible experiment evidence in Git and keeps large binary artifacts local by default.

## Tracked

- Experiment configs under `configs/`
- Source code under `src/` and `scripts/`
- Test code under `tests/`
- Human-readable docs under `docs/`
- Metrics and summaries such as:
  - `offline_metrics.json`
  - `repeated_subject_splits.json`
  - `repeated_offline_splits.json`
  - `subject_thresholds.json`
  - `threshold_grid_heldout.json`
  - `outputs/experiment_index.json`

## Not Tracked

- Raw EEG data under `data/raw/`
- Download archives such as `.zip`
- EDF files
- Trained model binaries such as `outputs/**/model.joblib`

## Rationale

The model binaries are reproducible from the tracked configs and scripts, while some models can exceed GitHub's recommended file size. Keeping them out of Git makes the repository easier to clone, review, and maintain.

If a trained model must be shared, use one of these approaches:

- Attach it to a release artifact.
- Store it in an external artifact bucket.
- Move model binaries to Git LFS intentionally.

## Rebuild Summary

After running experiments, refresh the tracked result index with:

```bash
python3 scripts/build_results_summary.py
```
