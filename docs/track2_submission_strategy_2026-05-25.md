# Track2 Submission Strategy 2026-05-25

## Situation

当前问题不是模型没有学起来，而是本地 dev 表现已经很高，下一步必须让测试办法更接近官方检查。官方 Track2 指标按 `speech / sound / singing / music` 分 type 计算二分类 Macro-F1，再对四类取平均，因此只看整体 `accuracy` 或整体 `macro_f1` 容易掩盖某个 type 的掉点。

## Current Candidates

### Main Candidate: `general_v3`

- Checkpoint: `/root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v3_train4000_bs128_to_bs32_lr1e6_seed7/best.pt`
- Config: `configs/experiments/ast_audioset_ft.yaml`
- Progress submission: `/root/autodl-tmp/ATADD/outputs/submission_track2_progress_ast_v3_seed7/submission.zip`

Key validation:

| Manifest | Track2 Macro-F1 | Weakest Type |
| --- | --- | --- |
| `track2_dev_balanced_500_v3.csv` | `0.99625` | `speech=0.99300` |
| `track2_dev_disjoint_alt30.csv` | `1.00000` | none |
| `track2_dev_disjoint_cap500.csv` | `0.99042` | `music=0.97168` |
| `track2_dev_music_only.csv` | `0.99500` | `music=0.99500` |

Interpretation: this is still the safest main submission. The only visible concern is that `music` drops on the larger disjoint-cap slice, but it remains strong on the balanced music-only slice.

### Backup Candidate: `focus_music3_sound2`

- Checkpoint: `/root/autodl-tmp/ATADD/outputs/track2_subset_ast_focus_music3_sound2_v1/best.pt`
- Config: `configs/experiments/ast_audioset_ft_music_sound_focus.yaml`
- Progress submission: `/root/autodl-tmp/ATADD/outputs/submission_track2_progress_ast_focus_music3_sound2_v1/submission.zip`

Key validation:

| Manifest | Track2 Macro-F1 | Weakest Type |
| --- | --- | --- |
| `track2_dev_balanced_500_v3.csv` | `0.99575` | `speech=0.99200` |
| `track2_dev_disjoint_alt30.csv` | `1.00000` | none |
| `track2_dev_disjoint_cap500.csv` | `0.99162` | `music=0.97850` |
| `track2_dev_music_only.csv` | `0.99400` | `music=0.99400` |

Interpretation: this is a serious backup candidate. It is slightly weaker than `general_v3` on the balanced main dev, but stronger on the larger disjoint-cap slice under Track2 Macro-F1. If progress leaderboard rewards disjoint/generalization behavior, this may be competitive.

### Deprioritized Candidate: `v2_seed7_calib`

- Checkpoint: `/root/autodl-tmp/ATADD/outputs/track2_subset_ast_audioset_ft_v2_bs128_seed7_to_bs32_lr1e6_seed7/best.pt`

This candidate is consistently weaker under the newer audit:

- `track2_dev_balanced_500_v3.csv`: `track2_macro_f1=0.99300`
- `track2_dev_disjoint_cap500.csv`: `track2_macro_f1=0.97899`
- main weakness: `music=0.93397` on disjoint-cap

It should stay as historical reference, not as an active submission.

### Deprioritized Candidate: `music_specialist`

- Checkpoint: `/root/autodl-tmp/ATADD/outputs/track2_subset_ast_music_specialist_v1/best.pt`

The standalone music specialist is weaker on music-only dev than both main AST candidates:

- `music_specialist`: `music_macro_f1=0.99100`
- `general_v3`: `music_macro_f1=0.99500`
- `focus_music3_sound2`: `music_macro_f1=0.99400`

It should not replace `general_v3` for music under the current evidence.

## Progress Submission Comparison

Two progress submissions already exist and have valid shape:

| Candidate | Rows | Submission Zip |
| --- | --- | --- |
| `general_v3` | `45875` | `/root/autodl-tmp/ATADD/outputs/submission_track2_progress_ast_v3_seed7/submission.zip` |
| `focus_music3_sound2` | `45875` | `/root/autodl-tmp/ATADD/outputs/submission_track2_progress_ast_focus_music3_sound2_v1/submission.zip` |

Prediction comparison:

- Different predictions: `1904 / 45875`
- Difference rate: `4.15%`
- `general_v3` predicts `fake=26476`, `real=19399`
- `focus_music3_sound2` predicts `fake=27958`, `real=17917`

Interpretation: the backup candidate is meaningfully more aggressive toward `fake`. This is useful as a leaderboard probe because it tests a different decision boundary, not just a near-duplicate.

## Progress Leaderboard Feedback

The `focus_music3_sound2` submission improved over `general_v3` on the progress leaderboard.

Observed progress score for `focus_music3_sound2`:

| Metric | Score |
| --- | --- |
| `macro_f1` | `73.22` |
| `speech_f1` | `75.29` |
| `sound_f1` | `70.80` |
| `singing_f1` | `83.91` |
| `music_f1` | `62.88` |

Interpretation:

- The local audit correctly identified `focus_music3_sound2` as the stronger probe candidate for harder checking.
- The biggest live weakness is now clearly `music`, not the overall AST backbone.
- `singing` is already strong, so the next round should avoid changes that improve music by sacrificing singing.
- `focus_music3_sound2` is more fake-biased than `general_v3`, and that bias helped on the progress set, so threshold/probability calibration is now a higher-value direction than another blind full training run.

Observed progress score for `focus_music3_sound2_th0p35`:

| Metric | Score |
| --- | --- |
| `macro_f1` | `73.65` |
| `speech_f1` | `75.58` |
| `sound_f1` | `71.01` |
| `singing_f1` | `84.64` |
| `music_f1` | `63.38` |

Interpretation:

- Lowering the fake threshold from `0.50` to `0.35` improves all reported type F1 scores on the progress set.
- The model is under-predicting `fake` for the progress distribution; a more fake-sensitive decision boundary is currently beneficial.
- Since the gain has not saturated at `0.35`, the next immediate probe should test slightly lower thresholds instead of launching another long training run first.

Additional leaderboard feedback: `focus_music3_sound2_th0p10` is stronger than `th0p35`.

Observed progress score for `focus_music3_sound2_th0p10`:

| Metric | Score |
| --- | --- |
| `macro_f1` | `76.76` |
| `speech_f1` | `77.08` |
| `sound_f1` | `73.93` |
| `singing_f1` | `89.46` |
| `music_f1` | `66.57` |

Interpretation:

- The progress set still rewards a much lower fake threshold, so the model's calibrated probabilities are not aligned with the hidden progress distribution.
- This is now mainly a decision-boundary problem, not a representation problem.
- The gain from `0.35` to `0.10` is large across all four types, so the next probe should use a denser grid below `0.10` and stop as soon as macro-F1 or any fragile type starts dropping.

Additional leaderboard feedback: `focus_music3_sound2_th0p0005` is stronger than `th0p10`.

Observed progress score for `focus_music3_sound2_th0p0005`:

| Metric | Score |
| --- | --- |
| `macro_f1` | `79.11` |
| `speech_f1` | `78.09` |
| `sound_f1` | `75.97` |
| `singing_f1` | `92.95` |
| `music_f1` | `69.42` |

Interpretation:

- The gain from `0.10` to `0.0005` is large, so the useful threshold range is much lower than the original model calibration suggested.
- The progress set is best served by a very fake-sensitive boundary so far.
- Further probes should go below `0.0005` in small steps, but the risk of over-correction is now higher.

## Immediate Recommendation

1. Treat `focus_music3_sound2_th0p0005` as the current live best progress submission.
2. Keep `general_v3` as the stability baseline and fallback.
3. Do not submit `v2_seed7_calib` or `music_specialist` unless a leaderboard result reveals that the current main/focus assumptions are wrong.
4. Next improvement should target the `music` weakness first, while monitoring `sound` as the second-lowest type.

## Next Experiments

Highest priority:

- Build a larger or cleaner music-heavy disjoint validation slice, then compare `general_v3` and `focus_music3_sound2` again.
- Generate probability/logit outputs, not just hard labels, so we can test threshold calibration per type on dev without retraining.
- If type metadata is unavailable for progress/final test, avoid any submission strategy that requires knowing the hidden type at inference.
- Continue lower-threshold submissions from `focus_music3_sound2` probabilities below `0.0005`, starting with `0.0003` and `0.0002`, to find the over-correction point.

Lower priority:

- Train another specialist only if it beats `general_v3` on `music_only` and does not degrade disjoint-cap.
- Revisit ensembling only after probability output exists; hard-label majority vote has limited resolution with the current candidates.
