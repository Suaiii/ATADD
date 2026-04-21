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
