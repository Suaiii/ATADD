# Sync Status 2026-05-25

## Summary

本地仓库已完成一次从云端 `/root/autodl-tmp/ATADD` 到工作区的保护性同步。

- 同步时间：`2026-05-25`
- 云端入口：`connect.bjb1.seetacloud.com:21190`
- 保护性备份目录：`E:\aNB\TECH\ATADD\.local\sync_backup_20260525_210537`
- 同步方式：先备份本地同名文件，再覆盖写入云端版本

当前判断：云端实验进展已经显著超出最初的 `WAVLM / XLSR / MERT` 初步对照阶段，项目主线已转向 `AST AudioSet`。

## Main Line

当前主提交方向应视为：

- Backbone：`AST AudioSet`
- 主训练 recipe：
  - 大平衡子集：`track2_train_balanced_4000.csv`
  - 第一阶段：`batch_size=128`
  - 第二阶段：从 best checkpoint 做 `batch_size=32, lr=1e-6, epochs=1` 温和校准
- 当前最佳记录：
  - Experiment ID: `exp_track2_ast_audioset_ft_v3_train4000_bs128_to_bs32_lr1e6_seed7`
  - Accuracy: `0.9963`
  - Macro F1: `0.99625`

对应输出目录见：

- `/root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v3_train4000_bs128_to_bs32_lr1e6_seed7`

## Backup Lines

当前可保留的备选或对照路线：

- `AST AudioSet` specialist / focus 路线
  - `configs/experiments/ast_audioset_ft_music_sound_focus.yaml`
  - `configs/experiments/ast_audioset_ft_music_specialist.yaml`
- `MERT 330M`
  - 作为通用音频 SSL 强备选
- `HuBERT`
  - 作为中等强度语音 SSL 对照
- `WAVLM / XLSR`
  - 保留作早期基线对照，不再是当前主线

## Files Synced From Server

这次同步回本地的关键文件类别包括：

- 配置：
  - `configs/baselines/*`
  - `configs/experiments/ast_*`
  - `configs/experiments/wavlm_base_aug_noise.yaml`
- 代码：
  - `src/atadd/dataset.py`
  - `src/atadd/train.py`
  - `src/atadd/eval.py`
  - `src/atadd/metrics.py`
  - `src/atadd/predict.py`
- 脚本：
  - `scripts/build_focus_manifest.py`
  - `scripts/build_track2_balanced_subset.py`
  - `scripts/filter_manifest_by_type.py`
  - `scripts/oracle_specialist_eval.py`
  - `scripts/run_predict.ps1`
  - `scripts/run_predict.sh`
- 文档与结果：
  - `docs/competition_brief.md`
  - `docs/feature_extractor_survey.md`
  - `reports/results_table.csv`
  - `reports/troubleshooting_log.md`
- manifests：
  - `data/manifests/track2_*`
  - `data/manifests/train.csv`
  - `data/manifests/val.csv`

## Keep In Git

建议纳入版本控制的内容：

- AST 相关实验配置
- `predict.py` 与预测脚本
- manifest 构建脚本
- oracle specialist 评估脚本
- `track2` 平衡子集与 disjoint dev manifests
- 最新文档与结果表
- 训练 / 评测 / 指标逻辑改动

## Keep Local Only

建议继续只保留本地，不纳入版本控制的内容：

- `.claude/settings.local.json`
  - 本地工具权限配置
- `.claude/scheduled_tasks.lock`
  - 本地运行时锁文件
- `data/train_reversed/`
  - 本地私有数据目录
- `.local/`
  - 本地备份、凭据和临时记录

## Current Risks

当前最主要的风险不是模型性能，而是仓库整理度：

- 云端仍是脏工作区，很多关键结果还没经过正式提交整理
- 本地与云端虽然已同步文件，但还没有形成一组干净、可审阅、可提交的 commit
- 当前本地验证指标虽已扩展，但仍需持续注意与官方 `Track2 Macro-F1` 口径完全一致

## Recommended Next Step

最合理的下一步是分两段完成：

1. 先把当前工作区整理成“可提交集合”
   - 保留主线 AST、specialist、predict、manifests、results
   - 排除本地私有环境和数据目录
2. 再做一次提交前审视
   - 主方案
   - 备选 specialist 方案
   - 指标口径
   - 提交流程是否可复现
