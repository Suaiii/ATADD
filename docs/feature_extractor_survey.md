# Track2 音频特征提取网络与训练增强方法调研

## 1. 任务背景

当前工作只面向 `track2`，暂不考虑 `track1`。`track2` 的目标不是单一语音场景，而是覆盖 `speech / sound / singing / music` 的 all-type audio deepfake detection，因此这一部分工作的核心是两件事：

- 判断哪类前端特征提取网络更适合作为 `track2` 的统一 backbone。
- 判断哪些训练增强方法值得优先进入第一轮实验。

团队分工上，本文档只覆盖你负责的“特征提取网络 + 训练增强方法”部分，不重复展开 `aasist` 和 `ALLM` 主线。

## 2. 特征提取网络对比

| 方向 | 模型定位 | 输入与前端 | 主干结构 | 预训练目标 | 输出隐向量特性 | 适合 Track2 的原因 | 主要风险 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `XLSR / XLS-R` | 多语种语音 SSL 前端 | 16 kHz 波形，经卷积特征编码后送入 Transformer | `wav2vec 2.0` 风格卷积前端 + Transformer | 掩码对比学习 + 量化目标 | 偏语音表征，擅长跨语言、跨说话人共享声学模式 | 适合作为语音 SSL 对照组，帮助判断“语音表征是否已经足够” | 对 `music / sound` 这类非语音音频缺少专门先验 |
| `WAVLM` | 面向 full-stack speech tasks 的通用语音 SSL 前端 | 16 kHz 波形，经卷积前端后送入 Transformer | 卷积前端 + Transformer，带 gated relative position bias | 掩码预测 + 去噪联合预训练 | 对说话人、韵律、噪声和伪造纹理更敏感 | 更贴近 deepfake detection 对伪造痕迹和副语言信息的需求 | 仍以语音预训练为主，对强非语音样本的覆盖有限 |
| `MERT` | 面向 music/audio understanding 的通用音频 SSL 前端 | 音频波形输入，预训练中结合音乐教师信号 | Transformer 编码器 | 掩码建模 + 音乐相关教师目标 | 更重视音高、音色、结构和音乐属性 | 如果 `track2` 中 `music / singing / sound` 占比高，理论上更匹配任务 | 复现链路更依赖特定实现，且和纯语音 SSL 的比较要注意公平性 |
| `CSAM` | 更像训练策略，而不是独立前端 backbone | 依赖已有检测器 | 叠加在现有模型训练上 | 来自 `CodecFake`，强调跨域泛化 | 不直接产出新的前端 embedding，而是改善训练稳定性和域外泛化 | 适合放到“增强策略”序列中，而不是第一轮 backbone 对比中 | 不能直接与 `WAVLM / XLSR / MERT` 当成同级前端比较 |

## 3. 从“音频信号到隐向量”的路径理解

### `XLSR`

- 从原始 16 kHz 波形开始。
- 经卷积特征编码器得到低帧率局部声学表示。
- 再由 Transformer 建模长时上下文。
- 通过掩码对比学习得到适合语音内容和说话人模式的隐向量。

结论：`XLSR` 更像强语音 SSL 对照前端，适合作为比较基线，但不一定是 `track2` 的第一优先方案。

### `WAVLM`

- 从原始波形开始，经卷积前端抽取局部表征。
- 使用 Transformer 建模全局上下文。
- 预训练不只做掩码预测，还显式结合去噪任务。
- 输出隐向量对说话人、噪声、韵律和伪造细节更加敏感。

结论：`WAVLM` 对 deepfake detection 的适配性强，适合优先作为首轮语音前端。

### `MERT`

- 也从原始音频出发，但训练目标面向通用音频和音乐理解。
- 隐向量更强调音高、音色、旋律和结构信息。
- 对 `music / singing / sound` 这类非纯语音样本理论上更友好。

结论：如果 `track2` 的非语音比例较高，`MERT` 很可能是比纯语音 SSL 更强的前端候选。

### `CSAM`

- `CSAM` 不定义新的前端特征提取流程。
- 它解决的是“已有检测器如何训练得更稳、更泛化”。

结论：当前阶段应把 `CSAM` 放入增强策略，而不是与 `WAVLM / XLSR / MERT` 同级对比。

## 4. 训练增强方法对比

| 增强方法 | 作用位置 | 预期收益 | 额外开销 | 主要风险 | 当前建议 |
| --- | --- | --- | --- | --- | --- |
| `加噪 (noise injection)` | 波形级 | 提高对录制噪声、压缩噪声和环境扰动的鲁棒性 | 很低 | 噪声过强会淹没伪造细节 | 第一批优先做 |
| `增益扰动 (gain perturbation)` | 波形级 | 降低模型对响度分布的过拟合 | 很低 | 扰动过大可能破坏真实/伪造的幅度线索 | 第一批优先做 |
| `SpecAugment / 时频遮挡` | 频谱级或中间特征级 | 提升局部缺失下的稳定性 | 低到中 | 遮挡过强可能抹掉关键伪造痕迹 | 第二优先 |
| `loss 调整` | 优化目标 | 处理类别不平衡或强化难例学习 | 低 | 在平衡子集上容易引入额外优化噪声 | 基线稳定后再做 |
| `学习率策略` | 优化器调度 | 改善收敛稳定性 | 低 | 搜索空间容易膨胀 | 第二优先，单变量验证 |
| `正则化` | 模型头或优化层面 | 抑制过拟合 | 很低 | 过强会拖慢收敛 | 作为配套微调项 |

## 5. 已完成的小实验结果

本轮实验在云端完成，使用 `track2` 平衡子集做最小验证：

- 训练集：`4000`
- 验证集：`1200`
- 抽样方式：按 `type x label` 平衡抽样
- 训练轮数：`3 epochs`
- 目标：先确认 backbone 方向和最低成本增强是否值得保留

| 实验 | 配置 | Accuracy | Macro F1 | 训练耗时 | 峰值显存 | 结论 |
| --- | --- | --- | --- | --- | --- | --- |
| `WAVLM baseline` | `wavlm_base` | `0.5942` | `0.5876` | `104.81s` | `2779.21MB` | 最小 baseline 跑通 |
| `WAVLM + noise/gain` | `wavlm_base_aug_noise` | `0.5967` | `0.5909` | `102.83s` | `2779.21MB` | 比 `WAVLM baseline` 小幅更好 |
| `XLSR baseline` | `xlsr_base` | `0.5000` | `0.3333` | `137.81s` | `4845.05MB` | 当前设置下明显弱于 `WAVLM`，疑似偏单类预测 |
| `MERT baseline` | `mert_base` | `0.6542` | `0.6531` | `66.35s` | `2469.81MB` | 当前子集上整体最佳 |
| `MERT + noise/gain` | `mert_base_aug_noise` | `0.6358` | `0.6345` | `68.16s` | `2469.81MB` | 未超过 `MERT baseline` |
| `HuBERT baseline` | `hubert_base` | `0.6183` | `0.6168` | `73.04s` | `2557.66MB` | 介于 `WAVLM` 与 `MERT` 之间，纯 SSL 略胜 `WAVLM` |
| `MERT-330M baseline` | `mert_v1_330m` | `0.6792` | `0.6720` | `199.70s` | `2880.48MB` | 比 `MERT-95M` 高 2.5 acc / 1.9 f1，验证同系列放大有正收益 |
| `AST AudioSet baseline` | `ast_audioset` | `0.7600` | `0.7562` | `154.16s` | `551.55MB` | 当前所有 backbone 最强，且峰值显存比波形 SSL 小 5 倍 |

### 实验解读

- `WAVLM` 已经证明自己是可运行、可增强、可复现的语音前端。
- `XLSR` 在当前子集和训练轮数下明显落后，不适合作为当前第一优先方案。
- `MERT` 在当前平衡子集上长期可靠，且 `MERT-95M -> MERT-330M` 还能继续涨点，说明音乐/通用音频 SSL 路线没到天花板。
- `HuBERT` 单纯 SSL 配方比 `WAVLM` 略好，说明 `WAVLM` 的 denoise 联合预训练在 deepfake 子集上不是关键差异化。
- `AST` 把"频谱图 + AudioSet 音频事件预训练"路线打到 0.76，**比所有波形 SSL 都明显高一个台阶**，且因 backbone 冻结后等价于一次前向，显存只有 551MB。说明 track2 的伪造痕迹可能更适合在频谱级而不是波形级捕捉。
- `noise + gain` 对 `WAVLM` 是正收益，但对 `MERT` 没有带来提升，因此增强策略不应直接跨 backbone 平移结论。

## 6. 当前结论

### 当前推荐顺序

1. `AST` (AudioSet 频谱图)
2. `MERT-330M`
3. `MERT-95M`
4. `HuBERT`
5. `WAVLM`
6. `XLSR`

### 当前判断

- `AST` 已从"备选频谱图路线"上升为"当前第一优先候选"，在 track2 平衡子集上 acc=0.7600/f1=0.7562，比第二名 MERT-330M 高约 8 个点，且峰值显存只有 551MB。下一步的训练增强、调度、扩数据都应优先以 AST 作为主线背骨。
- `MERT-330M` 是当前波形 SSL 的最强项，比 `MERT-95M` 涨 2.5 acc / 1.9 f1，确认"同系列放大"在这一阶段还有红利；但代价是训练耗时翻倍（66s -> 200s）。作为非 AST 路线的次优选项保留。
- `MERT-95M` 仍是性价比最佳的小模型 baseline，适合做快速增强对照（changes-per-hour 高）。
- `HuBERT` 略胜 `WAVLM`（+2.4 acc），说明 `WAVLM` 的 denoise 联合预训练在当前子集和 3 epoch 下不构成显著优势，可以把语音 SSL 主线收窄到 `HuBERT`。
- `WAVLM` 退居"已有正收益增强证据"的二线，不再作为第一优先 backbone。
- `XLSR` 暂时保留为对照组，不建议投入更多优先资源。
- `CSAM` 仍然作为训练增强策略调研，不进入第一轮 backbone 实现优先级。

### 第一批最值得保留的增强

当前优先保留 `waveform noise + gain`，但结论限定在 `WAVLM` 上更成立。

原因：

- 仓库里已实现，接入成本最低。
- 在 `WAVLM` 上已经看到正向信号。
- 在 `MERT` 上没有提升，说明下一轮增强设计要区分 backbone。

## 7. 初步任务完成度判断

如果把你的初步任务定义为：

- 完成 `XLSR / WAVLM / MERT / CSAM` 的方法调研
- 给出明确的前端优先级建议
- 用至少一组到两组最小实验支撑判断

那么现在可以判断为：**初步任务已经完成第一阶段，并且比之前更扎实。**

已经完成的部分：

- 四个方向的文献和方法定位已整理完成。
- `WAVLM vs XLSR vs MERT` 的第一轮波形 SSL 对照已经形成。
- `HuBERT` 的语音 SSL 第二代表已加入对照，并验证其略胜 `WAVLM`。
- `MERT-95M -> MERT-330M` 的同系列放大对照已经形成，确认放大有正收益。
- `AST AudioSet` 作为频谱图路线已经接入并跑通，建立了远超波形 SSL 的新主线。
- `WAVLM baseline vs WAVLM + noise/gain`、`MERT baseline vs MERT + noise/gain` 两组增强证据线已经形成。
- 已经可以给团队一个明确的当前建议顺序：`AST > MERT-330M > MERT-95M > HuBERT > WAVLM > XLSR`。

尚未完成的部分：

- 这些结果仍然来自平衡子集和 3 epoch 的最小实验，不是最终官方排名依据。
- `SpecAugment`、`loss`、`scheduler` 等第二批增强还未验证。
- `CSAM` 还停留在方法定位阶段，尚未进入实验接入设计。

## 8. 后续任务规划

### 第一优先级

- 在 `AST` 上做更长 epoch（5-10 ep）和更大子集复验，确认 0.76 不是 3 epoch 偶然结果。
- 在 `AST` 上验证频域增强（`SpecAugment`、time/freq mask），先于波形增强。
- 目标：判断 AST 在跨数据规模下的稳定性与可继续提升空间。

### 第二优先级

- 在 `MERT-330M` 与 `AST` 上分别加 `SpecAugment`，做"波形 SSL × 频谱增强"和"频谱 backbone × 频谱增强"两条线的横向对照。
- 改 pooling：当前对 AST/MERT 都用 `last_hidden_state.mean(dim=1)`，AST 的 `[CLS]` token 与多层加权（layer-wise attention pooling）值得各跑一组。

### 第三优先级

- 继续调研 `CSAM` 的接入方式，明确它是作为训练目标增强、域泛化策略还是数据组织策略插入现有框架。
- 评估 `BEATs` 是否值得手工接入（HF 无官方权重，需 Microsoft 原仓库），当 AST 在更大子集上保持优势时再启动。

### 第四优先级

- 当 `AST` 与 `MERT-330M` 的主线结论稳定后，再决定是否投入全量数据或重复实验做参赛化固化。

## 9. 与队友工作的边界

- 本文档只关注“前端特征提取网络”和“训练增强策略”。
- 不重复设计 `aasist` 分类后端，不与队友已有 Excel 结果冲突。
- 不重复展开 `ALLM` 路线，只在后续团队汇总时做横向对照。

## 10. 参考来源

- AT-ADD challenge evaluation plan: [arXiv:2604.08184](https://arxiv.org/abs/2604.08184)
- `WavLM`: [arXiv:2110.13900](https://arxiv.org/abs/2110.13900)
- `XLSR`: [arXiv:2006.13979](https://arxiv.org/abs/2006.13979)
- `XLS-R`: [arXiv:2111.09296](https://arxiv.org/abs/2111.09296)
- `MERT`: [arXiv:2306.00107](https://arxiv.org/abs/2306.00107)
- `CSAM / CodecFake`: [arXiv:2405.04880](https://arxiv.org/abs/2405.04880)
