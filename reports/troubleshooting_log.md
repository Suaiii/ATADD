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

