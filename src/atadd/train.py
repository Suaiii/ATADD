from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from atadd.augment import WaveformAugment
from atadd.config import load_experiment_config
from atadd.dataset import AudioClassificationDataset, collate_audio_batch
from atadd.metrics import classification_metrics
from atadd.modeling import AudioBackboneClassifier
from atadd.utils import append_csv_row, ensure_dir, now_seconds, save_json, set_seed


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train ATADD audio baseline.")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--train-manifest", required=True, type=str)
    p.add_argument("--val-manifest", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--epochs", default=None, type=int)
    p.add_argument("--batch-size", default=None, type=int)
    p.add_argument("--lr", default=None, type=float)
    p.add_argument("--num-workers", default=None, type=int)
    p.add_argument("--enable-augment", action="store_true")
    return p.parse_args()


@torch.no_grad()
def evaluate(model, loader, device: torch.device, num_classes: int) -> Dict[str, float]:
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

    if args.epochs is not None:
        cfg.train.epochs = args.epochs
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.lr is not None:
        cfg.train.lr = args.lr
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers
    if args.enable_augment:
        cfg.augment.enable = True

    set_seed(args.seed)

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    out_dir = ensure_dir(args.output_dir)
    log_csv = out_dir / "training_log.csv"
    run_start = now_seconds()

    augment = WaveformAugment(
        enable=cfg.augment.enable,
        noise_prob=cfg.augment.noise_prob,
        noise_scale=cfg.augment.noise_scale,
        gain_prob=cfg.augment.gain_prob,
        gain_db=cfg.augment.gain_db,
    )

    train_ds = AudioClassificationDataset(
        manifest_path=args.train_manifest,
        sample_rate=cfg.data.sample_rate,
        max_seconds=cfg.data.max_seconds,
        audio_column=cfg.data.audio_column,
        label_column=cfg.data.label_column,
        augment=augment,
    )
    val_ds = AudioClassificationDataset(
        manifest_path=args.val_manifest,
        sample_rate=cfg.data.sample_rate,
        max_seconds=cfg.data.max_seconds,
        audio_column=cfg.data.audio_column,
        label_column=cfg.data.label_column,
        augment=None,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_audio_batch,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds,
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
        freeze_backbone=cfg.model.freeze_backbone,
        kind=cfg.model.kind,
        feature_extractor_name=cfg.model.feature_extractor_name,
    ).to(device)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.train.lr,
        weight_decay=cfg.train.weight_decay,
    )

    best_metric = -math.inf
    best_epoch = -1
    bad_epochs = 0

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for epoch in range(1, cfg.train.epochs + 1):
        model.train()
        train_loss_sum = 0.0
        train_true = []
        train_pred = []
        pbar = tqdm(train_loader, desc=f"epoch {epoch}/{cfg.train.epochs}")
        for batch in pbar:
            x = batch["input_values"].to(device)
            y = batch["labels"].to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
            optimizer.step()

            train_loss_sum += float(loss.item()) * x.shape[0]
            pred = torch.argmax(logits, dim=1)
            train_true.append(y.detach().cpu().numpy())
            train_pred.append(pred.detach().cpu().numpy())

        y_true = np.concatenate(train_true)
        y_pred = np.concatenate(train_pred)
        train_metrics = classification_metrics(y_true, y_pred, num_classes=cfg.data.num_classes)
        train_loss = train_loss_sum / len(train_loader.dataset)

        val_metrics = evaluate(model, val_loader, device=device, num_classes=cfg.data.num_classes)
        score = val_metrics[cfg.train.main_metric]

        if score > best_metric:
            best_metric = score
            best_epoch = epoch
            bad_epochs = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "config_path": str(Path(args.config).resolve()),
                    "model_name": cfg.model.name,
                    "pretrained_name": cfg.model.pretrained_name,
                    "num_classes": cfg.data.num_classes,
                    "dropout": cfg.model.dropout,
                },
                out_dir / "best.pt",
            )
        else:
            bad_epochs += 1

        append_csv_row(
            log_csv,
            fieldnames=[
                "epoch",
                "train_loss",
                "train_accuracy",
                "train_macro_f1",
                "val_loss",
                "val_accuracy",
                "val_macro_f1",
                "main_metric",
                "main_metric_value",
            ],
            row={
                "epoch": epoch,
                "train_loss": f"{train_loss:.6f}",
                "train_accuracy": f"{train_metrics['accuracy']:.6f}",
                "train_macro_f1": f"{train_metrics['macro_f1']:.6f}",
                "val_loss": f"{val_metrics['loss']:.6f}",
                "val_accuracy": f"{val_metrics['accuracy']:.6f}",
                "val_macro_f1": f"{val_metrics['macro_f1']:.6f}",
                "main_metric": cfg.train.main_metric,
                "main_metric_value": f"{score:.6f}",
            },
        )

        print(
            f"[epoch {epoch}] train_loss={train_loss:.4f} "
            f"val_acc={val_metrics['accuracy']:.4f} val_f1={val_metrics['macro_f1']:.4f}"
        )

        if bad_epochs >= cfg.train.early_stop_patience:
            print(f"Early stopping triggered at epoch {epoch}")
            break

    elapsed_sec = now_seconds() - run_start
    gpu_mem_mb = 0.0
    if device.type == "cuda":
        gpu_mem_mb = float(torch.cuda.max_memory_allocated(device) / (1024 * 1024))

    summary = {
        "model": cfg.model.name,
        "config_path": str(Path(args.config).resolve()),
        "seed": args.seed,
        "epochs_run": epoch,
        "best_epoch": best_epoch,
        "best_metric_name": cfg.train.main_metric,
        "best_metric_value": best_metric,
        "elapsed_seconds": elapsed_sec,
        "peak_gpu_memory_mb": gpu_mem_mb,
        "output_dir": str(out_dir.resolve()),
    }
    save_json(out_dir / "run_summary.json", summary)
    print("Training completed.")
    print(summary)


if __name__ == "__main__":
    main()

