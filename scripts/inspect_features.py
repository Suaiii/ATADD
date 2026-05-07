from __future__ import annotations

import argparse
from pathlib import Path

import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Print a summary of a .pt produced by scripts/dump_features.py."
    )
    p.add_argument("path", type=str)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    path = Path(args.path)
    bundle = torch.load(path, map_location="cpu")

    feats = bundle["features"]
    labels = bundle["labels"]
    paths = bundle.get("audio_paths", [])

    print(f"file:           {path}")
    print(f"size:           {path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"model_name:     {bundle.get('model_name')}")
    print(f"pretrained:     {bundle.get('pretrained_name')}")
    print(f"kind:           {bundle.get('kind')}")
    print(f"sample_rate:    {bundle.get('sample_rate')}")
    print(f"max_seconds:    {bundle.get('max_seconds')}")
    print(f"dtype:          {bundle.get('dtype')}")
    print(f"manifest:       {bundle.get('manifest')}")
    print(f"config_path:    {bundle.get('config_path')}")
    print(f"features:       shape={tuple(feats.shape)} dtype={feats.dtype}")
    feats_f = feats.float()
    print(
        f"  mean={feats_f.mean().item():+.4f}  "
        f"std={feats_f.std().item():+.4f}  "
        f"min={feats_f.min().item():+.4f}  "
        f"max={feats_f.max().item():+.4f}"
    )
    print(f"labels:         shape={tuple(labels.shape)} dtype={labels.dtype}")
    counts = torch.bincount(labels)
    for cls, c in enumerate(counts.tolist()):
        print(f"  class {cls}: {c}")
    print(f"audio_paths:    n={len(paths)}")
    if paths:
        print(f"  first: {paths[0]}")
        print(f"  last : {paths[-1]}")


if __name__ == "__main__":
    main()
