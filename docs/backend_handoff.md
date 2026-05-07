# Track2 后端接入交接：冻结 backbone × frame 级特征

本文档面向后端（AASIST / attentive pooling / 其他 head）的负责人，目的是把"前端 backbone"和"后端 head"的接口锁死，让你这边迭代 head 时**完全不用再跑 backbone 前向**。

## 1. 流水线在哪一刀切开

```
audio waveform (B, T_wav)
        ↓ AudioBackboneClassifier.extract_features
frame features (B, T, D)        ← 冻结，离线 dump 到磁盘
        ↓ <你的 HEAD>
logits (B, num_classes)
        ↓ cross-entropy
```

当前 `AudioBackboneClassifier.forward` 在 `extract_features` 之上做的是 mean-pool + Linear，作为最便宜的 baseline 保留。你的 head 替换的是下半段。

## 2. `extract_features` 接口契约

定义见 `src/atadd/modeling.py`（方法 `extract_features`）。

- **输入**：`input_values: torch.Tensor`，shape `(B, T_wav)`，dtype `float32`，采样率取自 config 的 `data.sample_rate`（当前所有 backbone 都是 16000）。
- **输出**：`(B, T, D)`，HF backbone 的 `last_hidden_state`。
- 波形 → 频谱图（AST）的转换被封装在方法内部，你看到的永远是 `(B, T, D)`，不需要关心 backbone 类型。
- backbone 在 `freeze_backbone=True` 下构造时 `requires_grad=False`，不会回传梯度。

## 3. 各 backbone 的 (T, D) 表（max_seconds=5.0, sample_rate=16000）

数值均为实测（除 `wavlm_base` / `xlsr_base` / `mert_base` 未 dump，但同架构系列推断一致）。

| backbone | kind | T | D | 备注 |
| --- | --- | --- | --- | --- |
| `wavlm_base` | waveform | 249 | 768 | 语音 SSL，wav2vec2-style 50Hz（推断） |
| `xlsr_base` | waveform | 249 | 1024 | 语音 SSL，50Hz（推断） |
| `hubert_base` | waveform | 249 | 768 | 语音 SSL，50Hz（实测） |
| `mert_base` (95M) | waveform | 249 | 768 | 音乐 SSL，50Hz（按 MERT-330M 实测推断） |
| `mert_v1_330m` | waveform | 249 | 1024 | 音乐 SSL，50Hz（实测） |
| `ast_audioset` | spectrogram | 1214 | 768 | AudioSet，索引 0/1 是 `[CLS]` / 蒸馏 token（共 2 个 prepended token + 1212 patch）；T 由 AST FeatureExtractor 的 `max_length` 决定，**与 max_seconds 无直接关系** |

> 之前文档曾误写 MERT 是 75Hz/T~374，实际 MERT-v1 也是 wav2vec2-style 卷积前端，输出 50Hz/T=249。

如有疑问，对任意 dump `.pt` 跑 `python -m scripts.inspect_features <path>.pt` 即可看实际 T。

## 4. Dump 文件 schema

每次 `scripts/dump_features.py --output X.pt` 产出一个 `torch.save` 的 dict：

| key | type | shape / 值 |
| --- | --- | --- |
| `features` | `torch.Tensor` | `(N, T, D)`，默认 `float16`（也可 `float32`） |
| `labels` | `torch.Tensor` | `(N,)`, `int64` |
| `audio_paths` | `list[str]` | 长度 N，与 manifest 顺序一致 |
| `model_name` | str | 例如 `ast_audioset` |
| `pretrained_name` | str | HF id |
| `kind` | str | `waveform` or `spectrogram` |
| `sample_rate` | int | 16000 |
| `max_seconds` | float | 5.0 |
| `manifest` | str | source CSV 的绝对路径 |
| `config_path` | str | source YAML 的绝对路径 |
| `dtype` | str | `float16` or `float32` |
| `shape` | `list[int]` | 与 features.shape 一致 |

**`features` / `labels` / `audio_paths` 三者顺序严格对齐 manifest**（DataLoader 跑的是 `shuffle=False`）。

## 5. 消费示例

```python
import torch
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

bundle = torch.load("features/ast_audioset__train_balanced.pt", map_location="cpu")
X = bundle["features"].float()   # 训练时建议 cast 回 fp32
y = bundle["labels"]

train_loader = DataLoader(
    TensorDataset(X, y), batch_size=32, shuffle=True, num_workers=2
)

for feats, labels in train_loader:
    logits = your_head(feats)              # feats: (B, T, D), fp32
    loss = F.cross_entropy(logits, labels)
    ...
```

注意：

- 整个 `features` 张量驻留 CPU，按 batch 再 `.to(device)`。AST 在子集上是 ~7GB（fp16），fp32 翻倍。
- head 训练时**不再需要 backbone**，纯 CPU 数据 → GPU 只需要喂 head。可以在笔记本上迭代。
- 当前所有样本都已 pad 到同一个 `T`，**不需要 attention mask**。如果以后切到变长输入，schema 会增加 `lengths` 字段，会另行通知。

## 6. 首批 dump（已完成 2026-05-07）

实测大小如下，全部位于服务器 `/root/autodl-tmp/ATADD/features/`：

| 优先级 | 文件名 | 形状 | 大小 (fp16) | 备注 |
| --- | --- | --- | --- | --- |
| P0 | `ast_audioset__track2_train_balanced_500.pt` | (4000, 1214, 768) | 7.0 GB | AST 当前最强 backbone，含 [CLS]/dist 双 token |
| P0 | `ast_audioset__track2_dev_balanced_150.pt` | (1200, 1214, 768) | 2.1 GB | |
| P1 | `mert_v1_330m__track2_train_balanced_500.pt` | (4000, 249, 1024) | 2.0 GB | 音乐 SSL 最强 |
| P1 | `mert_v1_330m__track2_dev_balanced_150.pt` | (1200, 249, 1024) | 584 MB | |
| P2 | `hubert_base__track2_train_balanced_500.pt` | (4000, 249, 768) | 1.5 GB | 语音 SSL 系列代表 |
| P2 | `hubert_base__track2_dev_balanced_150.pt` | (1200, 249, 768) | 438 MB | |

合计 ~14 GB。后端实验**先在 P0 上跑通**再向下扩展（AST 的预期收益空间最大）。

如需新 backbone（WAVLM、MERT-95M、XLSR）或更大 split，告诉前端这边重新 dump，**不要在你这边重跑 backbone**。

## 7. 命名规范与重新 dump

输出路径约定：

```
features/<model_name>__<manifest_stem>.pt
```

重新 dump（例：换 backbone、改窗口）：

```bash
python -m scripts.dump_features \
  --config configs/baselines/ast_audioset.yaml \
  --manifest data/manifests/track2_train_balanced_500.csv \
  --output features/ast_audioset__track2_train_balanced_500.pt \
  --device cuda --batch-size 8 --dtype float16 \
  --overwrite
```

dump 由前端这边负责。需要新 backbone / 新 split 直接告诉我，我跑完同步给你，**不要在你这边重新跑 backbone 前向**——会浪费 GPU 时间，也会污染版本一致性（HF 权重缓存可能不一样）。

## 8. 留作后续讨论的点

- **AST 的 `[CLS]` token 在索引 0**：你的 head 可以包含也可以剥掉，但请在代码里写清楚选了哪个，否则后面跨 backbone 比较时会混淆。
- 现在所有样本同 `T`，未来切到变长，schema 会带 `lengths`。
- fp16 存盘对绝大多数 head 没问题；如果 head 走混合精度训练，先 cast 到 fp32 再过 LayerNorm/BatchNorm，避免下溢。
