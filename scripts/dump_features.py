from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from atadd.config import load_experiment_config
from atadd.dataset import AudioClassificationDataset, collate_audio_batch
from atadd.modeling import AudioBackboneClassifier
from atadd.utils import ensure_dir


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Dump frozen-backbone frame-level features for one manifest."
    )
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--manifest", required=True, type=str)
    p.add_argument("--output", required=True, type=str, help="Output .pt path")
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--batch-size", default=None, type=int)
    p.add_argument("--num-workers", default=None, type=int)
    p.add_argument("--dtype", default="float16", choices=["float16", "float32"])
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--log-every", default=20, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output)
    if out_path.exists() and not args.overwrite:
        raise FileExistsError(f"{out_path} exists. Use --overwrite to replace.")
    ensure_dir(str(out_path.parent))

    cfg = load_experiment_config(args.config)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    device = torch.device(
        args.device if (torch.cuda.is_available() or args.device == "cpu") else "cpu"
    )
    out_dtype = torch.float16 if args.dtype == "float16" else torch.float32

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
        freeze_backbone=True,
        kind=cfg.model.kind,
        feature_extractor_name=cfg.model.feature_extractor_name,
    ).to(device)
    model.eval()

    feats_chunks = []
    label_chunks = []
    t0 = time.time()
    n_total = len(ds)
    with torch.no_grad():
        for i, batch in enumerate(loader):
            x = batch["input_values"].to(device)
            y = batch["labels"]
            hidden = model.extract_features(x)
            feats_chunks.append(hidden.detach().to("cpu", dtype=out_dtype))
            label_chunks.append(y.detach().cpu())
            if (i + 1) % args.log_every == 0:
                done = min((i + 1) * cfg.train.batch_size, n_total)
                elapsed = time.time() - t0
                print(f"[{done}/{n_total}] elapsed={elapsed:.1f}s", flush=True)

    features = torch.cat(feats_chunks, dim=0)
    labels = torch.cat(label_chunks, dim=0)
    audio_paths = [str(r.audio_path) for r in ds.rows]

    payload = {
        "features": features,
        "labels": labels,
        "audio_paths": audio_paths,
        "model_name": cfg.model.name,
        "pretrained_name": cfg.model.pretrained_name,
        "kind": cfg.model.kind,
        "sample_rate": cfg.data.sample_rate,
        "max_seconds": cfg.data.max_seconds,
        "manifest": str(Path(args.manifest).resolve()),
        "config_path": str(Path(args.config).resolve()),
        "dtype": args.dtype,
        "shape": list(features.shape),
    }
    torch.save(payload, out_path)
    elapsed = time.time() - t0
    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(
        f"Saved features={tuple(features.shape)} dtype={features.dtype} "
        f"labels={tuple(labels.shape)} -> {out_path} "
        f"({size_mb:.1f} MB, {elapsed:.1f}s)",
        flush=True,
    )


if __name__ == "__main__":
    main()
