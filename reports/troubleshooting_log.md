# Troubleshooting Log

## Template

- Date:
- Model:
- Config:
- Stage: (data/train/eval)
- Symptom:
- Root cause:
- Fix:
- Status:

---

## 2026-04-22: Track2 subset bring-up

- Date: 2026-04-22
- Model: wavlm_base / wavlm_base_aug_noise
- Config: configs/baselines/wavlm_base.yaml, configs/experiments/wavlm_base_aug_noise.yaml
- Stage: data/train
- Symptom: `torchaudio` import failed with `OSError: libcudart.so.13: cannot open shared object file` and blocked FLAC loading on track2.
- Root cause: remote base environment had PyTorch 2.8.0+cu128, but the installed `torchaudio` binary could not load its CUDA runtime dependency; dataset code also imported `torchaudio` eagerly at module import time.
- Fix: installed `soundfile` and `scipy`, then patched `src/atadd/dataset.py` to lazy-fallback from `torchaudio` to `soundfile + scipy.signal.resample_poly`; generated balanced track2 subset manifests instead of using placeholder manifests.
- Status: resolved for current subset experiments.
