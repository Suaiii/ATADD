# Troubleshooting Log

## Template

- Date:
- Experiment ID:
- Model:
- Config:
- Train Manifest:
- Val Manifest:
- Stage: (data/train/eval)
- Device:
- Symptom:
- Root cause:
- Fix:
- Status:
- Log path:
- Output dir:
- Next action:

---

## Usage Notes

- 每次只记录一个明确问题，避免把多个失败原因混在同一条里。
- 如果问题发生在云端后台任务，务必同时记录日志文件路径和输出目录。
- 如果某次实验最终可复现，请在 `Fix` 中写明最终采用的命令、关键参数或环境变更。

---

## 2026-05-06：本地 git bash `ssh-copy-id` 反复 Permission denied

- Date: 2026-05-06
- Experiment ID: -
- Model: -（环境层问题）
- Config: -
- Stage: setup
- Device: 本地 Windows git bash → autodl 服务器（`connect.bjb1.seetacloud.com:47192`）
- Symptom: `ssh-copy-id` 输入密码 `/np3mmpoWqsI` 后两次返回 `Permission denied, please try again`，最后 `Permission denied (publickey,password)`。后续所有 ssh/scp/git push 都不通。
- Root cause: 密码以 `/` 起头，git bash 里 `ssh-copy-id` 借助 OpenSSH 与 SSH agent 转发输入时疑似把首字符解释错。无 `sshpass` 与 `plink` 可用，无法非交互注入密码。
- Fix: 用 `E:\anaconda\python.exe` 的 paramiko 4.0 直接以密码连入服务器，把本地 `~/.ssh/id_ed25519.pub` 追加到 `~/.ssh/authorized_keys`（`grep -qxF` 防重复）。之后所有 `ssh connect.bjb1.seetacloud.com ...` 全程免密。
- Status: 已解决，公钥已上线
- Log path: -
- Output dir: -
- Next action: 后续若服务器重租，paramiko 一次性脚本可以复用：读取 `.local/assistant_secrets.md` 中的 password，把 `id_ed25519.pub` push 上去。

---

## 2026-05-06：服务器 `import torchaudio` OSError libcudart.so.13

- Date: 2026-05-06
- Experiment ID: exp_track2_ast_audioset（首次触发）
- Model: ast_audioset（也影响所有需要 torchaudio 的路径）
- Config: configs/baselines/ast_audioset.yaml
- Stage: train（启动期 backbone/FE 加载）
- Device: server cuda
- Symptom: 干跑 `AutoFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")` 抛 `OSError: libcudart.so.13: cannot open shared object file`，单独 `python -c "import torchaudio"` 也复现。
- Root cause: 服务器装的是 `torch==2.8.0+cu128`（CUDA 12.8 / `libcudart.so.12`），但 `torchaudio==2.11.0`（需要 CUDA 13 / `libcudart.so.13`），版本错配。`dataset.py` 已用 `try/except` 把 torchaudio 包成可选，但 transformers 的 AST FE 链路里仍然会触发 torchaudio 导入，绕不过去。
- Fix: `pip install --index-url https://download.pytorch.org/whl/cu128 torchaudio==2.8.0` 降级到与 torch 匹配的版本。`python -c "import torchaudio; print(torchaudio.__version__)"` 返回 `2.8.0+cu128` 即修复。
- Status: 已解决
- Log path: -
- Output dir: -
- Next action: 后续如果服务器重装 torch/torchaudio，先校验 `torch.__version__` 与 `torchaudio.__version__` 的 CUDA 后缀一致，再跑实验。

---

## 2026-05-06：服务器 `git fetch/pull origin` 到 GitHub 443 端口超时

- Date: 2026-05-06
- Experiment ID: -
- Model: -
- Stage: git sync
- Device: server
- Symptom: 服务器 `git -c http.version=HTTP/1.1 fetch origin` 报 `curl 28 Failed to connect to github.com port 443 after 130510 ms: Connection timed out`，本地仓库 `c722e01/ea2f7d9/fba3337` 推不上服务器走 GitHub 这条路。
- Root cause: autodl 容器到 github.com 的 443 端口被网络策略拦截（HTTP/2 framing 错与 timeout 都来自同一原因）。
- Fix: 跳过 GitHub，走 SSH 直推。一次性在服务器侧 `git config receive.denyCurrentBranch updateInstead`，本地 `git push connect.bjb1.seetacloud.com:/root/autodl-tmp/ATADD main` 直接更新 server 工作树。注意推送前要把服务器侧未提交的本地路径覆写（`dataset.py` 软回退、`./pretrained/...` yaml）`git stash` 起来，推完再 `git stash pop`。
- Status: 已解决（流程化为 stash → push → pop）
- Log path: -
- Output dir: -
- Next action: 后续做服务器同步默认走"本地 push 到服务器"，不依赖 server 出网到 GitHub。如果将来 autodl 放开 443 再切回 `git pull origin main`。
---

## 2026-05-12: `ast_audioset_ft_lr1e5_long` underperformed on the larger balanced subset

- Date: 2026-05-12
- Experiment ID: exp_track2_ast_audioset_ft_lr1e5_long_v2
- Model: ast_audioset_ft_lr1e5_long
- Config: configs/experiments/ast_audioset_ft_lr1e5_long.yaml
- Train Manifest: data/manifests/track2_train_balanced_2000.csv
- Val Manifest: data/manifests/track2_dev_balanced_500.csv
- Stage: train
- Device: server cuda
- Symptom: The lower-learning-rate AST run stayed below the stronger `ast_audioset_ft` recipe on the same larger balanced subset.
- Root cause: Lowering the learning rate from `2e-5` to `1e-5` and extending training did not improve adaptation in this setting.
- Fix: Stop the low-LR run and reallocate the single-GPU slot to a higher-value multi-seed stability rerun of `ast_audioset_ft`.
- Status: closed without promotion
- Log path: /root/autodl-tmp/ATADD/logs/ast_audioset_ft_lr1e5_long_v2.log
- Output dir: /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_lr1e5_long_v2
- Next action: Keep `ast_audioset_ft` as the main AST recipe and prioritize stability checks before trying new AST-specific augmentation.

---

## 2026-05-12: `ast_audioset_ft` seed7 rerun validated stability

- Date: 2026-05-12
- Experiment ID: exp_track2_ast_audioset_ft_v2_seed7
- Model: ast_audioset_ft
- Config: configs/experiments/ast_audioset_ft.yaml
- Train Manifest: data/manifests/track2_train_balanced_2000.csv
- Val Manifest: data/manifests/track2_dev_balanced_500.csv
- Stage: train+eval
- Device: server cuda
- Symptom: Need to confirm whether the strong AST result on the larger balanced subset was a single-seed fluctuation.
- Root cause: Not a failure. This run was launched as a stability verification after the seed42 run reached `accuracy=0.97625`.
- Fix: Re-run the same AST fine-tuning recipe with `seed=7` and compare the best validation checkpoint and final eval summary.
- Status: validated
- Log path: /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v2_seed7.log
- Output dir: /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_seed7
- Next action: Promote AST full fine-tuning as the current primary recipe and decide whether the next single-GPU slot goes to another seed or to AST-specific augmentation such as SpecAugment.

---

## 2026-05-12: Large-batch AST run improved both throughput and accuracy

- Date: 2026-05-12
- Experiment ID: exp_track2_ast_audioset_ft_v2_bs128_seed42
- Model: ast_audioset_ft
- Config: configs/experiments/ast_audioset_ft.yaml
- Train Manifest: data/manifests/track2_train_balanced_2000.csv
- Val Manifest: data/manifests/track2_dev_balanced_500.csv
- Stage: train+eval
- Device: server cuda, single NVIDIA RTX PRO 6000 Blackwell Server Edition
- Symptom: The original AST fine-tuning recipe used only about 4-5 GB GPU memory, leaving most of the single-card server idle.
- Root cause: Batch size was conservative for compatibility, not tuned for the available 97.9 GB GPU.
- Fix: Increase batch size to 128 and rerun the same AST full fine-tuning recipe with seed 42.
- Status: promoted
- Log path: /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v2_bs128_seed42.log
- Output dir: /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_seed42
- Next action: Treat `/root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_seed42/best.pt` as the high-throughput baseline; best epoch is 4 with `accuracy=0.9895`, `macro_f1=0.9895`, and peak GPU memory about 91.2 GB.

---

## 2026-05-13: Large-batch seed7 confirms the need for best-checkpoint selection

- Date: 2026-05-13
- Experiment ID: exp_track2_ast_audioset_ft_v2_bs128_seed7
- Model: ast_audioset_ft
- Config: configs/experiments/ast_audioset_ft.yaml
- Train Manifest: data/manifests/track2_train_balanced_2000.csv
- Val Manifest: data/manifests/track2_dev_balanced_500.csv
- Stage: train+eval
- Device: server cuda, single NVIDIA RTX PRO 6000 Blackwell Server Edition
- Symptom: Need to check whether the `bs128` improvement was stable beyond seed 42.
- Root cause: Not a failure. This was a seed stability check for the large-batch recipe.
- Fix: Re-run the same `batch_size=128` AST fine-tuning recipe with `seed=7`.
- Status: validated but not promoted over seed42
- Log path: /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v2_bs128_seed7.log
- Output dir: /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_seed7
- Next action: Keep `bs128 seed42` plus optional `bs32 lr=1e-6` calibration as the current best recipe. The seed7 run reached `accuracy=0.98725` at epoch 3, then dropped to about `0.979` at epochs 4-5, so large-batch training must rely on best checkpoint / early stopping rather than the final epoch.

---

## 2026-05-12: Disjoint dev checks reduce the overfitting concern for `bs128`

- Date: 2026-05-12
- Experiment ID: exp_track2_ast_audioset_ft_v2_bs128_eval_alt30 / exp_track2_ast_audioset_ft_v2_bs128_eval_cap500
- Model: ast_audioset_ft
- Config: configs/experiments/ast_audioset_ft.yaml
- Train Manifest: data/manifests/track2_train_balanced_2000.csv
- Val Manifest: data/manifests/track2_dev_disjoint_alt30.csv and data/manifests/track2_dev_disjoint_cap500.csv
- Stage: eval
- Device: server cuda
- Symptom: After `bs128` reached `accuracy=0.9895`, the main concern was whether the score came from overfitting the current balanced dev subset.
- Root cause: Not a failure. The original dev manifest is a balanced sample from official dev; an independent dev slice was needed to check whether the checkpoint still generalizes.
- Fix: Evaluate the same `bs128` checkpoint on two disjoint official-dev manifests: a fully balanced `30 x type x label` slice and a larger capped remainder slice.
- Status: validated
- Log path: /root/autodl-tmp/ATADD/logs/eval_ast_bs128_seed42_disjoint_alt30.log and /root/autodl-tmp/ATADD/logs/eval_ast_bs128_seed42_disjoint_cap500.log
- Output dir: /root/autodl-tmp/ATADD/outputs/eval_ast_bs128_seed42_disjoint_alt30 and /root/autodl-tmp/ATADD/outputs/eval_ast_bs128_seed42_disjoint_cap500
- Next action: Keep `bs128` as a valid main checkpoint; disjoint results were `alt30 accuracy=0.9875`, `cap500 accuracy=0.9895`, so the high score is unlikely to be only a current-dev artifact.

---

## 2026-05-12: Small-batch refinement is schedule-sensitive

- Date: 2026-05-12
- Experiment ID: exp_track2_ast_audioset_ft_v2_bs128_to_bs8_seed42 / exp_track2_ast_audioset_ft_v2_bs128_to_bs32_lr1e6_seed42
- Model: ast_audioset_ft
- Config: configs/experiments/ast_audioset_ft.yaml
- Train Manifest: data/manifests/track2_train_balanced_2000.csv
- Val Manifest: data/manifests/track2_dev_balanced_500.csv
- Stage: train+eval
- Device: server cuda
- Symptom: A direct `bs128 -> bs8` continuation dropped from the `bs128` best score, raising an overfitting or refinement-instability concern.
- Root cause: The `bs8, lr=5e-6, epochs=3` refinement likely disturbed an already good checkpoint; the issue was not simply that any post-training continuation overfits.
- Fix: Add `--init-checkpoint` support in `src/atadd/train.py`, then compare two staged schedules from the same `bs128` checkpoint: aggressive `bs8, lr=5e-6, 3 epochs` versus gentle `bs32, lr=1e-6, 1 epoch`.
- Status: partially validated
- Log path: /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v2_bs128_to_bs8_seed42.log and /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v2_bs128_to_bs32_lr1e6_seed42.log
- Output dir: /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_to_bs8_seed42 and /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_to_bs32_lr1e6_seed42
- Next action: Prefer the gentle refinement if using a two-stage recipe. `bs8` reached only `accuracy=0.9875`, while `bs32, lr=1e-6, 1 epoch` reached `accuracy=0.98975` on the main dev and improved disjoint checks to `alt30 accuracy=0.9917`, `cap500 accuracy=0.9926`.

---

## 2026-05-13: Seed7 low-lr bs32 calibration confirms the two-stage AST recipe

- Date: 2026-05-13
- Experiment ID: exp_track2_ast_audioset_ft_v2_bs128_seed7_to_bs32_lr1e6_seed7
- Model: ast_audioset_ft
- Config: configs/experiments/ast_audioset_ft.yaml
- Train Manifest: data/manifests/track2_train_balanced_2000.csv
- Val Manifest: data/manifests/track2_dev_balanced_500.csv
- Stage: train+eval
- Device: server cuda, single NVIDIA RTX PRO 6000 Blackwell Server Edition
- Symptom: Need to verify whether the successful `bs128 -> bs32, lr=1e-6, 1 epoch` calibration was specific to the seed42 checkpoint.
- Root cause: Not a failure. This was a cross-seed validation of the two-stage AST recipe.
- Fix: Start from `/root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_seed7/best.pt`, then run `batch_size=32`, `lr=1e-6`, `epochs=1`.
- Status: promoted
- Log path: /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v2_bs128_seed7_to_bs32_lr1e6_seed7.log
- Output dir: /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_seed7_to_bs32_lr1e6_seed7
- Next action: Treat `AST bs128 + gentle bs32 calibration` as the current primary recipe. The seed7 calibrated checkpoint reached `accuracy=0.9930`, `macro_f1=0.9930` on the main dev, and stayed strong on disjoint checks with `alt30 accuracy=0.9875`, `cap500 accuracy=0.9921`.

---

## 2026-05-13: Expanding the balanced train subset to 32000 pushes AST higher again

- Date: 2026-05-13
- Experiment ID: exp_track2_ast_audioset_ft_v3_train4000_bs128_seed7 / exp_track2_ast_audioset_ft_v3_train4000_bs128_to_bs32_lr1e6_seed7
- Model: ast_audioset_ft
- Config: configs/experiments/ast_audioset_ft.yaml
- Train Manifest: data/manifests/track2_train_balanced_4000.csv
- Val Manifest: data/manifests/track2_dev_balanced_500_v3.csv
- Stage: train+eval
- Device: server cuda, single NVIDIA RTX PRO 6000 Blackwell Server Edition
- Symptom: After the 16000-train recipe reached `0.9930`, the next uncertainty was whether AST still benefits from more balanced training data or had already saturated on the smaller subset.
- Root cause: Not a failure. This was a scale-up validation of the current AST main line under the largest balanced train subset that still fits the server disk budget.
- Fix: Build a larger balanced subset with `train_per_group=4000` and `dev_per_group=500`, then run `bs128, seed=7` followed by the same gentle `bs32, lr=1e-6, 1 epoch` calibration.
- Status: promoted
- Log path: /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v3_train4000_bs128_seed7.log and /root/autodl-tmp/ATADD/logs/ast_audioset_ft_v3_train4000_bs128_to_bs32_lr1e6_seed7.log
- Output dir: /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v3_train4000_bs128_seed7 and /root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v3_train4000_bs128_to_bs32_lr1e6_seed7
- Next action: Treat the `32000-train bs128 + gentle bs32 calibration` checkpoint as the new best overall recipe. The large-batch stage reached `accuracy=0.9940` at epoch 4, the calibrated checkpoint reached `accuracy=0.99625`, and disjoint checks stayed extremely strong with `alt30 accuracy=1.0000`, `cap500 accuracy=0.9960`.
