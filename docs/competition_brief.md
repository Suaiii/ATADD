# Competition Brief (Fill Before Full Training)

Source:
- https://mp.weixin.qq.com/s/AarYQIgp6Ni712m69iPHiA
- https://www.at-add.com/
- https://www.at-add.com/instructions
- https://www.at-add.com/rules

## Fixed Elements

- Task name: `AT-ADD Challenge 2026 Track 2: All-Type Audio Deepfake Detection`
- Input format: single input audio clip of unknown type (`speech / sound / singing / music`)
- Output format: deterministic binary prediction for each test audio, `real` or `fake`
- Main metric: Track 2 official metric is Macro-F1 balanced first within each type (`real/fake`) and then averaged across the four audio types
- Secondary metrics: no official secondary metric listed on the rules page
- Data split rule: train/dev may be split internally or merged for training; progress/evaluation sets must not be used for training, fine-tuning, pseudo-labeling, threshold tuning, or any test-time adaptation
- Submission rule: upload an arbitrary `.zip` file to Codabench; the zip must contain exactly one file named `predict.csv` with header `name,predict`
- Deadline:
  - Progress evaluation stage: already open as of 2026-05-13
  - Final evaluation stage opens: 2026-06-08
  - Final leaderboard freeze: 2026-06-15
  - Metadata & technical report submission: 2026-06-17

## Constraints

- Allowed external data: external labeled or unlabeled audio data is prohibited for training, calibration, pseudo-labeling, or distillation; public augmentation resources such as MUSAN/RIR are allowed only as augmentation sources
- Allowed pretraining models: publicly available and traceable pretrained models are allowed, provided they were not supervisedly trained or fine-tuned for audio deepfake detection or closely related authenticity classification tasks
- Runtime / memory limits: not listed on the official rules page as of 2026-05-13; verify directly on Codabench before final submission
- Number of submissions per day: not listed on the official rules page as of 2026-05-13; verify directly on Codabench before final submission

## Validation Mapping

- Local validation metric name: currently repo default is `accuracy`, while reports also record `macro_f1`
- Relationship to official metric: current local metric is only an approximation; to match Track 2 officially, local validation should compute per-type Macro-F1 and then average across `speech / sound / singing / music`
- Expected tolerance: unknown until a Track 2-compatible local metric and a progress-set submission are compared
