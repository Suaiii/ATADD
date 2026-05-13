# Track2 音频特征提取网络与增强策略阶段总结

## 1. 当前任务边界

当前工作只针对 `track2`，暂不考虑 `track1`。这一部分的目标是：

- 比较不同音频前端 backbone 在 `track2` 上的表现。
- 验证哪些增强策略值得继续投入。
- 给团队形成明确的主线和备选路线。

团队分工上，本文档只覆盖“特征提取网络 + 训练增强方法”，不重复展开 `aasist` 和 `ALLM` 主线。

## 2. 关键 backbone 定位

| 方向 | 核心定位 | 适合 Track2 的原因 | 当前判断 |
| --- | --- | --- | --- |
| `XLSR / XLS-R` | 多语种语音 SSL 前端 | 强语音表征，对跨语言语音任务有基础优势 | 当前明显偏弱，仅保留作对照 |
| `WAVLM` | 语音 SSL 前端，兼顾说话人和噪声信息 | 更贴近语音 deepfake 检测直觉 | 已被更强路线超越 |
| `HuBERT` | 经典语音 SSL 前端 | 作为语音 SSL 中等强度对照很合适 | 优于 `WAVLM`，但不够强 |
| `MERT` | 通用音频 / 音乐 SSL 前端 | 对 `music / singing / sound` 这类非纯语音更友好 | 明显强于语音 SSL |
| `AST AudioSet` | 频谱图 backbone，带 AudioSet 预训练 | 更善于捕获音频事件和谱图伪造痕迹 | 当前最强主线 |
| `CSAM` | 训练策略，不是独立 backbone | 可作为后续域泛化或增强策略 | 先继续调研，不纳入第一轮 backbone 排序 |

## 3. 已完成实验

所有结果目前都基于同一套平衡子集：

- `train=4000`
- `dev=1200`
- 采样方式：`type x label` 平衡抽样
- 任务：`track2`

### 第一轮 baseline / 增强结果

| 实验 | Accuracy | Macro F1 | 结论 |
| --- | --- | --- | --- |
| `wavlm_base` | `0.5942` | `0.5876` | 最小语音 baseline 跑通 |
| `wavlm_base_aug_noise` | `0.5967` | `0.5909` | `noise + gain` 对 `WAVLM` 有小幅正收益 |
| `xlsr_base` | `0.5000` | `0.3333` | 当前设置下明显失败 |
| `mert_base` | `0.6542` | `0.6531` | 通用音频前端明显强于语音 SSL |
| `mert_base_aug_noise` | `0.6358` | `0.6345` | `noise + gain` 对 `MERT` 无益 |
| `hubert_base` | `0.6183` | `0.6168` | 语音 SSL 中优于 `WAVLM` |
| `mert_v1_330m` | `0.6792` | `0.6720` | `MERT 95M -> 330M` 放大有效 |
| `ast_audioset` | `0.7600` | `0.7562` | 第一版 AST 已是最强 baseline |

### AST 上限探索

为了确认 AST 是否只是“偶然更高”，又额外做了两组实验：

| 实验 | 设置 | Accuracy | Macro F1 | 训练耗时 | 峰值显存 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `ast_audioset_long` | 冻结 backbone，训练 `8 epochs` | `0.8100` | `0.8094` | `1216.89s` | `551.55MB` | 仅靠更长训练就继续涨点 |
| `ast_audioset_ft` | 解冻 backbone，全量微调 `5 epochs` | `0.9533` | `0.9533` | `520.19s` | `4143.09MB` | 当前子集上的最高点，说明 AST 真正上限远高于冻结线性头 |

### AST 大子集稳定性与 batch 策略

后续实验切到更大的 track2 平衡子集：

- `train=16000`
- `dev=4000`
- train/dev 分别来自官方 train/dev，不混合官方划分
- manifest: `track2_train_balanced_2000.csv` / `track2_dev_balanced_500.csv`

| 实验 | 设置 | Accuracy | Macro F1 | 结论 |
| --- | --- | --- | --- | --- |
| `ast_audioset_ft_v2` | `batch_size=4`, `seed=42` | `0.9763` | `0.9762` | 大子集上依然稳定强 |
| `ast_audioset_ft_v2_seed7` | `batch_size=4`, `seed=7` | `0.9833` | `0.9832` | 换 seed 后继续高分 |
| `ast_audioset_ft_v2_seed123` | `batch_size=4`, `seed=123` | `0.9783` | `0.9782` | 第三组 seed 仍稳定 |
| `ast_audioset_ft_v2_bs128_seed42` | `batch_size=128`, `seed=42` | `0.9895` | `0.9895` | 大 batch 同时提升吞吐和指标 |
| `ast_audioset_ft_v2_bs128_seed7` | `batch_size=128`, `seed=7` | `0.9873` | `0.9872` | 大 batch 换 seed 仍强，但未超过 seed42 |
| `ast_audioset_ft_v2_bs128_to_bs8_seed42` | 从 `bs128 best.pt` 接 `bs8, lr=5e-6, 3 epochs` | `0.9875` | `0.9875` | 过强小 batch 精修会扰动最优点 |
| `ast_audioset_ft_v2_bs128_to_bs32_lr1e6_seed42` | 从 `bs128 best.pt` 接 `bs32, lr=1e-6, 1 epoch` | `0.9898` | `0.9897` | 温和精修略高于 bs128 |
| `ast_audioset_ft_v2_bs128_seed7_to_bs32_lr1e6_seed7` | 从 `bs128 seed7 best.pt` 接 `bs32, lr=1e-6, 1 epoch` | `0.9930` | `0.9930` | 跨 seed 复验后成为当前最佳主 dev 结果 |
| `ast_audioset_ft_v3_train4000_bs128_seed7` | `train=32000`, `batch_size=128`, `seed=7` | `0.9940` | `0.9940` | 扩大平衡训练集后继续提升 |
| `ast_audioset_ft_v3_train4000_bs128_to_bs32_lr1e6_seed7` | `train=32000` 下从 `bs128 best.pt` 接 `bs32, lr=1e-6, 1 epoch` | `0.9963` | `0.9963` | 当前最佳主 dev 结果，说明大子集 + 温和校准仍有效 |

为了检查 `0.9895+` 是否只是当前 dev 子集过拟合，又用官方 dev 中未进入当前 dev manifest 的样本做了两份 disjoint eval：

| Checkpoint | Disjoint manifest | Accuracy | Macro F1 | 结论 |
| --- | --- | --- | --- | --- |
| `bs128 best.pt` | `track2_dev_disjoint_alt30.csv` | `0.9875` | `0.9875` | 完全不重叠的平衡小切片仍高分 |
| `bs128 best.pt` | `track2_dev_disjoint_cap500.csv` | `0.9895` | `0.9894` | 较大不重叠 dev remainder 仍高分 |
| `bs128 -> bs32 lr1e-6 best.pt` | `track2_dev_disjoint_alt30.csv` | `0.9917` | `0.9917` | 温和精修在 disjoint 小切片上更好 |
| `bs128 -> bs32 lr1e-6 best.pt` | `track2_dev_disjoint_cap500.csv` | `0.9926` | `0.9925` | 温和精修在较大 disjoint remainder 上也更好 |
| `bs128 seed7 -> bs32 lr1e-6 best.pt` | `track2_dev_disjoint_alt30.csv` | `0.9875` | `0.9875` | 换 seed 的校准模型在完全不重叠小切片上仍高分 |
| `bs128 seed7 -> bs32 lr1e-6 best.pt` | `track2_dev_disjoint_cap500.csv` | `0.9921` | `0.9920` | 换 seed 的校准模型在较大 disjoint remainder 上保持同一高水平 |
| `32000-train bs128 -> bs32 lr1e-6 best.pt` | `track2_dev_disjoint_alt30.csv` | `1.0000` | `1.0000` | 更大训练集上的校准模型在完全不重叠小切片上达到满分 |
| `32000-train bs128 -> bs32 lr1e-6 best.pt` | `track2_dev_disjoint_cap500.csv` | `0.9960` | `0.9960` | 更大训练集上的校准模型在较大 disjoint remainder 上继续提升 |

阶段判断：`bs128` 不是单纯“把显存吃满”的工程优化，而是当前最有效的 AST 训练策略之一；`bs32, lr=1e-6, 1 epoch` 可以作为轻量校准尾巴保留。`bs128 seed7 -> bs32 lr1e-6` 先把成绩推到 `0.9930`，而扩大到 `train=32000` 后，`bs128 -> bs32 lr1e-6` 进一步达到 `0.9963`，说明“更大平衡训练集 + 温和校准”仍然有效。`bs128 seed7` 在较小训练集上 epoch 3 后会下降到 `0.979` 左右，说明大 batch 也不能无脑多训，必须保留 best checkpoint / early stopping。`bs8, lr=5e-6, 3 epochs` 的下降说明两阶段训练可行，但小 batch 接管必须非常温和。

## 4. 当前结论

### 当前排序

1. `AST AudioSet full fine-tuning + batch_size=128 + optional bs32 low-lr calibration`
2. `AST AudioSet full fine-tuning, batch_size=4, multi-seed stable`
3. `AST AudioSet frozen backbone + longer training`
4. `AST AudioSet baseline`
5. `MERT-v1-330M`
6. `MERT-base`
7. `HuBERT-base`
8. `WAVLM + noise/gain`
9. `WAVLM-base`
10. `XLSR-base`

### 直接结论

- `AST` 已经不是“可试试的备选”，而是当前最明确的主线。
- `AST` 的上限不在冻结线性头；一旦解冻 backbone，当前平衡子集上能直接到 `0.9533 / 0.9533`，扩大平衡训练集并采用大 batch + 温和校准后已推进到 `0.9963`。
- `batch_size=128` 当前应视为主训练策略，而不是单纯提速技巧；它在单卡上吃满显存并刷新了主 dev 指标。
- 对过拟合的担心目前被 disjoint dev 检查进一步缓解：更大训练集上的 `bs128 -> bs32 lr1e-6` 在未重叠 dev 样本上达到 `alt30=1.0000`、`cap500=0.9960`，说明提升不只是当前主 dev 偶然抬高。
- `MERT-v1-330M` 仍然是最值得保留的 waveform 路线备选。
- `HuBERT / WAVLM / XLSR` 这一组现在主要承担对照作用，不再是第一优先。
- `noise + gain` 这种波形增强不能简单外推到所有 backbone，至少对 `MERT` 不成立。

## 5. 研究方向判断

当前研究方向已经非常清楚：

- 主线应转向：`AST` 这类谱图 backbone 的上限挖掘。
- 备线保留：`MERT-v1-330M` 作为通用音频 waveform backbone。
- 语音 SSL 不再是主押注对象。

更具体地说，现在的问题已经不是“谁比谁强”，而是：

- `AST` 的高分是稳定可复现，还是小子集上的过拟合高点。
- `AST` 在更大子集、更多 epoch、不同 seed 下还能不能维持明显优势。
- `AST` 适合继续加哪一类增强，而不是盲目继续做 waveform 增强。

## 6. 后续计划

### 第一优先级

- 对 `AST` 做复验，而不是立刻换模型。
- 建议先做：
  - 更大平衡子集
  - 多 seed 重复
  - 更长 epoch 的稳定性检查

目标：确认 `AST` 的 `0.95` 级别表现不是偶然波动。

### 第二优先级

- 给 `AST` 加谱域增强，而不是 waveform 增强。
- 优先方向：
  - `SpecAugment`
  - `time mask`
  - `frequency mask`

目标：判断 `AST` 是否还能在高基线之上继续稳步提升。

### 第三优先级

- 保留 `MERT-v1-330M` 作为主备方案中的备线。
- 后续可以给它单独试更适配的增强，而不是复用 `WAVLM` 的 `noise + gain`。

### 第四优先级

- 当 `AST` 的稳定性确认后，再考虑：
  - 是否扩到更大数据规模
  - 是否做提交前主备方案固化
  - 是否把 `CSAM` 接入到主线训练中

## 7. 当前阶段判断

你负责的这部分工作现在已经不只是“初步完成”，而是已经进入**明确主线阶段**：

- backbone 方向已经筛出来了。
- 增强方向的错误路线也被排掉了一部分。
- 当前已经能给团队一个明确建议：

**主线：`AST`**  
**备线：`MERT-v1-330M`**

## 8. 参考

- AT-ADD challenge evaluation plan: [arXiv:2604.08184](https://arxiv.org/abs/2604.08184)
- `WavLM`: [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)
- `XLSR`: [arXiv:2006.13979](https://arxiv.org/abs/2006.13979)
- `XLS-R`: [arXiv:2111.09296](https://arxiv.org/abs/2111.09296)
- `MERT`: [arXiv:2306.00107](https://arxiv.org/abs/2306.00107)
- `AST`: [arXiv:2104.01778](https://arxiv.org/abs/2104.01778)
- `CSAM / CodecFake`: [arXiv:2405.04880](https://arxiv.org/abs/2405.04880)
