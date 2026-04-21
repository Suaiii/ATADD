from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from atadd.config import load_experiment_config
from atadd.dataset import AudioClassificationDataset, collate_audio_batch
from atadd.metrics import classification_metrics
from atadd.modeling import AudioBackboneClassifier
from atadd.utils import ensure_dir, save_json


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate ATADD model checkpoint.")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--manifest", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--batch-size", default=None, type=int)
    p.add_argument("--num-workers", default=None, type=int)
    return p.parse_args()


@torch.no_grad()
def run_eval(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
):
    model.eval()
    total_loss = 0.0
    all_true = []
    all_pred = []

    for batch in loader:
        x = batch["input_values"].to(device)
        y = batch["labels"].to(device)
        logits = model(x)
        loss = F.cross_entropy(logits, y)
        total_loss += float(loss.item()) * x.shape[0]

        pred = torch.argmax(logits, dim=1)
        all_true.append(y.cpu().numpy())
        all_pred.append(pred.cpu().numpy())

    y_true = np.concatenate(all_true)
    y_pred = np.concatenate(all_pred)
    metrics = classification_metrics(y_true, y_pred, num_classes=num_classes)
    metrics["loss"] = total_loss / len(loader.dataset)
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_experiment_config(args.config)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ds = AudioClassificationDataset(
        manifest_path=args.manifest,
        sample_rate=cfg.data.sample_rate,
        max_seconds=cfg.data.max_seconds,
        audio_column=cfg.data.audio_column,
        label_column=cfg.data.label_column,
        augment=None,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = AudioBackboneClassifier(
        pretrained_name=cfg.model.pretrained_name,
        num_classes=cfg.data.num_classes,
        dropout=cfg.model.dropout,
        freeze_backbone=False,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    metrics = run_eval(model, loader, device=device, num_classes=cfg.data.num_classes)
    summary = {
        "model": cfg.model.name,
        "config_path": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "manifest": str(Path(args.manifest).resolve()),
        "metrics": metrics,
    }
    save_json(out_dir / "eval_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()

