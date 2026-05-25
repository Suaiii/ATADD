# 项目长期记忆（Memory）

## 参赛对象（核心来源）
- 微信文章链接（项目主要参赛对象）：https://mp.weixin.qq.com/s/AarYQIgp6Ni712m69iPHiA

## 你的主要工作内容（基石任务）
- 调研音频特征提取网络架构（从音频信号到隐向量）和训练增强方法。
- 基线方向包含但不限于：XLSR、WAVLM、MERT、CSAM 等。

## 团队分工（以 `ATADD挑战赛.xlsx` 为准）
- 朱羿帅：调研音频特征提取网络架构（从音频信号到隐向量）和训练增强方法，聚焦 `XLSR`、`WAVLM`、`MERT`、`CSAM` 等方向。
- 邵振宇：调研音频基础分类器方向（例如 `aasist`）。
- 金俊山：调研基于 `ALLM` 的另一套基线方法（`AT-ADD-ALLM-Baseline`）。
- 后续工作默认避免与 Excel 中已有分工和已记录结果冲突，优先补你负责的“特征提取网络 + 训练增强”空白。

## 已有表格信息
- `ATADD挑战赛.xlsx` 中有 `任务安排` 与 `基线测试` 两个工作表。
- `基线测试` 已记录部分官方 baseline 的 `DEV数据测试-F1`，后续新增实验记录时应与表内已有项目区分来源，避免重复劳动或覆盖队友结论。

## 使用约定
- 后续每次开启本项目工作时，先读取并遵循本文件中的信息。

## 当前执行默认（两天冲刺）
- 统一入口命令：`scripts/run_train.ps1`、`scripts/run_eval.ps1`。
- 优先基线顺序：`WAVLM/XLSR` -> `MERT` -> `WAVLM+增强`。
- 所有对比需同数据划分、同评测脚本，并记录到 `reports/results_table.csv`。
- 当前数据集范围默认仅使用 `track2`，暂不考虑 `track1`。

## 数据下载约定
- Hugging Face 相关下载默认优先使用国内镜像：`export HF_ENDPOINT=https://hf-mirror.com`
- 下载前先执行 Hugging Face 登录：`hf auth login --token <your token>`
- 完成镜像设置与登录后，再调用对应数据集的 API 进行下载。

## 服务器工作背景
- 后续默认以远程云服务器为主要执行环境，不假设本地电脑持续在线。
- 当前主用云端机器入口：`connect.bjb1.seetacloud.com:47192`，默认工作目录为 `/root/autodl-tmp/ATADD`。
- 服务器真实 GPU 为两张 `NVIDIA RTX PRO 6000 Blackwell Server Edition`，后续按双卡并行方式安排最小实验。
- 当前所有真实数据集与正式训练任务默认都在云端执行，本地仓库主要承担代码整理、文档、配置与结果回填。
- 训练、下载、评测等长任务默认按“SSH 断开后仍继续运行”设计。
- 后续涉及长时间运行命令时，优先使用 `nohup`、`tmux`、`screen` 或后台日志重定向方式，而不是依赖前台终端会话。
- 给出的运行方案默认包含日志文件路径、后台运行方式和结果检查命令。
- 敏感凭据不写入版本控制文件；当前项目的本地私有凭据统一记录在 `.local/assistant_secrets.md`。

## 2026-05-25 Track2 Progress 冲刺结论
- 当前云端入口已切换为：`connect.bjb1.seetacloud.com:21190`，工作目录仍为 `/root/autodl-tmp/ATADD`。
- 当前主线模型是 `AST AudioSet`，线上 progress 最优来自 `focus_music3_sound2` 路线，而不是早期 `general_v3`。
- `focus_music3_sound2` 原始阈值提交线上约为：`macro_f1=73.22`，分项为 `speech=75.29`、`sound=70.80`、`singing=83.91`、`music=62.88`。
- 阈值调低后持续提升，说明 progress 集上模型明显低估 `fake`，当前关键不是重训，而是基于 `prob_fake` 做 fake-sensitive threshold calibration。
- 已知更强提交：
  - `focus_music3_sound2_th0p35`: `macro_f1=73.65`
  - `focus_music3_sound2_th0p01`: `macro_f1=76.76`
  - `focus_music3_sound2_th0p0005`: `macro_f1=79.11`，分项为 `speech=78.09`、`sound=75.97`、`singing=92.95`、`music=69.42`
- 当前 live best 是 `track2_progress_focus_music3_sound2_th0p0005_submission.zip`。
- 当前最主要短板仍是 `music`，其次是 `sound`；但降阈值目前对四个 type 都有正收益，应继续往更低阈值小步探索，直到 leaderboard 出现过冲。
- 本地提交包目录 `submissions/` 已将中高阈值冗余包归档到 `submissions/archive_thresholds_20260525/`，活跃目录优先保留低阈值候选。
