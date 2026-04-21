# ATADD 48-Hour Baseline Sprint

This repository implements a reproducible baseline pipeline for the AT-ADD challenge:

- Unified flow: `data -> model -> train -> eval -> report`
- Standard entry points: `run_train` and `run_eval`
- Baselines: `WAVLM`, `MERT`, `XLSR`
- One enhancement setting: waveform augmentation (noise + gain)

## 1) Environment

```powershell
python -m venv .venv
. .\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## 2) Data Manifest Format

Each split is a CSV file with at least:

- `audio_path`: absolute path, or path relative to the manifest file directory
- `label`: integer class id (`0 ... num_classes-1`)

Example:

```csv
audio_path,label
train/audio_0001.wav,0
train/audio_0002.wav,1
```

See examples in `data/manifests/*.example.csv`.

## 3) Baseline Configs

- `configs/baselines/wavlm_base.yaml`
- `configs/baselines/mert_base.yaml`
- `configs/baselines/xlsr_base.yaml`
- `configs/experiments/wavlm_base_aug_noise.yaml` (enhancement)

## 4) Standard Commands

### Windows PowerShell wrappers

Train:

```powershell
.\scripts\run_train.ps1 `
  -Config .\configs\baselines\wavlm_base.yaml `
  -TrainManifest .\data\manifests\train.csv `
  -ValManifest .\data\manifests\val.csv `
  -OutputDir .\outputs\wavlm_base `
  -Device cuda `
  -Seed 42
```

Eval:

```powershell
.\scripts\run_eval.ps1 `
  -Config .\configs\baselines\wavlm_base.yaml `
  -Checkpoint .\outputs\wavlm_base\best.pt `
  -Manifest .\data\manifests\val.csv `
  -OutputDir .\outputs\wavlm_base `
  -Device cuda
```

### Python module entry points (cross-platform)

Train:

```bash
PYTHONPATH=src python -m atadd.train \
  --config configs/baselines/wavlm_base.yaml \
  --train-manifest data/manifests/train.csv \
  --val-manifest data/manifests/val.csv \
  --output-dir outputs/wavlm_base \
  --device cuda \
  --seed 42
```

Eval:

```bash
PYTHONPATH=src python -m atadd.eval \
  --config configs/baselines/wavlm_base.yaml \
  --checkpoint outputs/wavlm_base/best.pt \
  --manifest data/manifests/val.csv \
  --output-dir outputs/wavlm_base \
  --device cuda
```

## 5) Output Structure

Each run writes:

- `best.pt`: best checkpoint by configured main metric
- `training_log.csv`: epoch metrics
- `run_summary.json`: final summary with timing and GPU memory
- `eval_summary.json`: evaluation output from `run_eval`

Reporting templates:

- `reports/results_table.csv`
- `reports/troubleshooting_log.md`

## 6) Two-Day Acceptance Checklist

- At least 2 baselines trained and evaluated with the same split/eval script.
- At least 1 enhancement-vs-baseline comparison.
- Every reported number traceable to a config + output directory.

## 7) Competition Brief Template

Use `docs/competition_brief.md` to fill:

- task input/output
- official metric
- split and submission format
- hard constraints

The challenge source link is already tracked in `memory.md`.

