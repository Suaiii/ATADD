# Track2 云端最小实验快速启动说明

## 1. 当前默认

- 真实数据不在本地跑，正式训练与评测默认都在云端执行。
- 当前只做 `track2`，暂不考虑 `track1`。
- 云端已知资源为两张 `RTX 5090`，建议优先并行跑两组最小 baseline。

## 2. 数据访问与下载

Hugging Face 相关下载默认先走国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
hf auth login --token <your token>
```

`AT-ADD-Track2` 数据集当前是 Hugging Face gated dataset，需要先同意条款并获得访问权限：

- 数据集页：<https://huggingface.co/datasets/xieyuankun/AT-ADD-Track2>
- 已公开信息显示总大小约 `36.8 GB`

获得访问权限后，可在云端下载到本地目录，例如：

```bash
huggingface-cli download xieyuankun/AT-ADD-Track2 \
  --repo-type dataset \
  --local-dir data/raw/track2
```

## 3. 云端环境准备

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

如果云端使用 `conda`，只要保证最终安装的是仓库 `requirements.txt` 中的依赖即可。

## 4. 推荐目录结构

```text
ATADD/
  configs/
  data/
    raw/track2/
    manifests/
      train.csv
      val.csv
  logs/
  outputs/
  reports/
```

## 5. Manifest 要求

当前代码要求 `train.csv` 和 `val.csv` 至少包含两列：

- `audio_path`
- `label`

`audio_path` 可以是绝对路径，也可以是相对 manifest 文件所在目录的路径。

## 6. 第一轮最小实验建议

优先级：

1. `wavlm_base`
2. `xlsr_base`
3. `wavlm_base_aug_noise`

第一轮目标不是正式横评，而是确认：

- `track2` 数据链路无误
- 训练和验证脚本可用
- `WAVLM / XLSR` 这两个前端在当前框架下可被快速验证

## 7. 单次运行命令

### 训练

```bash
bash scripts/run_train.sh \
  configs/baselines/wavlm_base.yaml \
  data/manifests/train.csv \
  data/manifests/val.csv \
  outputs/wavlm_base \
  cuda
```

### 评测

```bash
bash scripts/run_eval.sh \
  configs/baselines/wavlm_base.yaml \
  outputs/wavlm_base/best.pt \
  data/manifests/val.csv \
  outputs/wavlm_base \
  cuda
```

## 8. 双卡并行建议

仓库已补充一个云端启动脚本：

```bash
bash scripts/launch_track2_pair.sh \
  data/manifests/train.csv \
  data/manifests/val.csv \
  outputs
```

它会默认：

- `GPU 0` 跑 `wavlm_base`
- `GPU 1` 跑 `xlsr_base`
- 每个实验自动生成单独日志
- 训练结束后自动执行对应评测

如果第一轮结果正常，再追加：

```bash
nohup bash -lc 'CUDA_VISIBLE_DEVICES=0 bash scripts/run_train.sh \
  configs/experiments/wavlm_base_aug_noise.yaml \
  data/manifests/train.csv \
  data/manifests/val.csv \
  outputs/wavlm_base_aug_noise \
  cuda && \
  CUDA_VISIBLE_DEVICES=0 bash scripts/run_eval.sh \
  configs/experiments/wavlm_base_aug_noise.yaml \
  outputs/wavlm_base_aug_noise/best.pt \
  data/manifests/val.csv \
  outputs/wavlm_base_aug_noise \
  cuda' > logs/wavlm_base_aug_noise.log 2>&1 &
```

## 9. 结果回填要求

每次实验结束后，至少回填以下文件：

- `reports/results_table.csv`
- `reports/troubleshooting_log.md`

并保留：

- 输出目录中的 `best.pt`
- `training_log.csv`
- `run_summary.json`
- `eval_summary.json`

## 10. 当前已知限制

- 本地仓库没有真实 `track2` manifest，因此当前机器无法直接跑正式实验。
- `CSAM` 当前不应作为第一轮 backbone 实现目标，而应继续作为训练增强策略调研项。
