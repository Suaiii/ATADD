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
