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
python scripts/train_baseline.py --config configs/default.yaml
python scripts/simulate_online.py --config configs/default.yaml
```

To run on the public PhysioNet EEG Motor Movement/Imagery dataset:

```powershell
python scripts/train_baseline.py --config configs/physionet.yaml
python scripts/simulate_online.py --config configs/physionet.yaml
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

For a more realistic cross-patient estimate, run leave-one-subject-out evaluation:

```powershell
python scripts/evaluate_subject_generalization.py --config configs/figshare_stroke.yaml
```

Outputs are written to `outputs/`:

- `model.joblib`: trained baseline decoder
- `offline_metrics.json`: balanced accuracy, AUC, F1, confusion matrix
- `session_report.json`: pseudo-online trigger metrics
- `session_report.md`: clinician-facing summary draft

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
