# 项目长期记忆（Memory）

## 参赛对象（核心来源）
- 微信文章链接（项目主要参赛对象）：https://mp.weixin.qq.com/s/AarYQIgp6Ni712m69iPHiA

## 你的主要工作内容（基石任务）
- 调研音频特征提取网络架构（从音频信号到隐向量）和训练增强方法。
- 基线方向包含但不限于：XLSR、WAVLM、MERT、CSAM 等。

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
- 训练、下载、评测等长任务默认按“SSH 断开后仍继续运行”设计。
- 后续涉及长时间运行命令时，优先使用 `nohup`、`tmux`、`screen` 或后台日志重定向方式，而不是依赖前台终端会话。
- 给出的运行方案默认包含日志文件路径、后台运行方式和结果检查命令。
