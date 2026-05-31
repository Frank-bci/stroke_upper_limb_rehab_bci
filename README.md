# Stroke Upper Limb Rehab BCI MVP

This MVP demonstrates a product-oriented BCI pipeline for stroke upper-limb rehabilitation:

```text
EEG epochs
  -> signal quality gate
  -> real-time-safe preprocessing/features
  -> motor intention decoding
  -> trigger decision
  -> simulated BU100 wrist/hand assistance
  -> session report
```

The default demo uses synthetic EEG-like data so the full pipeline can run without downloading a dataset. Public dataset loaders can be added under `src/stroke_bci_mvp/datasets/`.

## Quick Start

```powershell
pip install -r requirements.txt
python scripts/train_baseline.py --config configs/default.yaml
python scripts/simulate_online.py --config configs/default.yaml
python scripts/calibrate_trigger_threshold.py --config configs/default.yaml
```

To run on the public PhysioNet EEG Motor Movement/Imagery dataset:

```powershell
python scripts/train_baseline.py --config configs/physionet.yaml
python scripts/simulate_online.py --config configs/physionet.yaml
```

`configs/physionet.yaml` is a fast experiment config with 3 subjects. To run all PhysioNet subjects, use:

```powershell
python scripts/train_baseline.py --config configs/physionet_full.yaml
python scripts/simulate_online.py --config configs/physionet_full.yaml
```

To prepare the Figshare stroke patient MI dataset metadata or EDF archive:

```powershell
python scripts/download_figshare_stroke.py --metadata-only
python scripts/download_figshare_stroke.py --include-edf-zip
```

After extracting `edffile.zip`, point `configs/figshare_stroke.yaml` to the EDF directory and run:

```powershell
python scripts/train_baseline.py --config configs/figshare_stroke.yaml
python scripts/simulate_online.py --config configs/figshare_stroke.yaml
```

`configs/figshare_stroke.yaml` is a fast experiment config with the first 6 EDF files. To use all locally downloaded Figshare EDF files, run:

```powershell
python scripts/train_baseline.py --config configs/figshare_stroke_full.yaml
python scripts/simulate_online.py --config configs/figshare_stroke_full.yaml
```

For a more realistic cross-patient estimate, run leave-one-subject-out evaluation:

```powershell
python scripts/evaluate_subject_generalization.py --config configs/figshare_stroke.yaml
```

Outputs are written by dataset. The default synthetic demo writes to `outputs/synthetic/`, PhysioNet writes to `outputs/physionet/`, and the Figshare stroke dataset writes to `outputs/figshare_stroke/`:

- `model.joblib`: trained baseline decoder
- `offline_metrics.json`: balanced accuracy, AUC, F1, confusion matrix
- `session_report.json`: pseudo-online trigger metrics
- `session_report.md`: clinician-facing summary draft
- `threshold_calibration.json`: selected threshold and candidate metrics from the training split

Training uses `online_windows` mode by default: training samples are generated with the same window length, step, and task timing used by the pseudo-online controller, avoiding a mismatch between full-epoch training and 1-second window inference. Public real-data configs use subject-aware train/test splits by default so the same subject does not appear in both train and test sets. The default synthetic demo keeps a random stratified split for fast smoke testing.

Threshold calibration runs cross-validation inside the training split and also reports performance on the held-out test split. With very few real-data subjects, calibrated thresholds can be unstable; prefer the full configs for final experiments. Full configs use fewer calibration folds and threshold candidates by default to keep runtime manageable.

Run the minimal test suite:

```powershell
pytest
```

## MVP Scope

The first implementation focuses on one narrow closed-loop action:

```text
Rest vs affected-hand motor intention -> simulated wrist/hand opening assistance
```

It is intentionally conservative:

- FBCSP + LDA baseline instead of a deep model
- signal quality gate before decoding
- continuous-window trigger confirmation
- refractory period after each trigger
- explicit rejection/failed-trigger reasons
