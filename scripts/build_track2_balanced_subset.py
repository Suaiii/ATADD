from __future__ import annotations

import argparse
import csv
import random
import tarfile
from collections import defaultdict
from pathlib import Path


LABEL_MAP = {"real": 0, "fake": 1}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a balanced Track2 subset and manifests.")
    p.add_argument("--train-label-csv", required=True, type=Path)
    p.add_argument("--dev-label-csv", required=True, type=Path)
    p.add_argument("--train-tar", required=True, type=Path)
    p.add_argument("--dev-tar", required=True, type=Path)
    p.add_argument("--output-root", required=True, type=Path)
    p.add_argument("--manifests-dir", required=True, type=Path)
    p.add_argument("--train-per-group", required=True, type=int)
    p.add_argument("--dev-per-group", required=True, type=int)
    p.add_argument("--seed", default=42, type=int)
    p.add_argument("--train-manifest-name", default="track2_train_balanced.csv", type=str)
    p.add_argument("--dev-manifest-name", default="track2_dev_balanced.csv", type=str)
    return p.parse_args()


def read_rows(label_csv: Path) -> list[dict[str, str]]:
    with label_csv.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def choose_rows(rows: list[dict[str, str]], per_group: int, seed: int) -> list[dict[str, str]]:
    buckets: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        buckets[(row["type"], row["label"])].append(row)

    rng = random.Random(seed)
    chosen: list[dict[str, str]] = []
    for key in sorted(buckets):
        group_rows = buckets[key]
        if len(group_rows) < per_group:
            raise ValueError(f"Group {key} only has {len(group_rows)} rows, need {per_group}")
        chosen.extend(rng.sample(group_rows, per_group))
    chosen.sort(key=lambda x: x["name"])
    return chosen


def extract_subset(rows: list[dict[str, str]], tar_path: Path, target_dir: Path) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    wanted = {row["name"] for row in rows}
    remaining = set(wanted)

    with tarfile.open(tar_path, "r") as tf:
        members_by_name = {}
        for member in tf.getmembers():
            if not member.isfile():
                continue
            name = Path(member.name).name
            if name in wanted:
                members_by_name[name] = member

        missing = sorted(wanted - set(members_by_name))
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(f"Missing {len(missing)} members in {tar_path}: {preview}")

        for row in rows:
            name = row["name"]
            out_path = target_dir / name
            if out_path.exists():
                remaining.discard(name)
                continue
            member = members_by_name[name]
            with tf.extractfile(member) as src, out_path.open("wb") as dst:
                if src is None:
                    raise FileNotFoundError(f"Failed to extract {name} from {tar_path}")
                dst.write(src.read())
            remaining.discard(name)

    if remaining:
        raise RuntimeError(f"Subset extraction incomplete: {len(remaining)} files still missing")


def write_manifest(rows: list[dict[str, str]], subset_dir: Path, manifest_path: Path) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["audio_path", "label", "type", "generator", "source_name"]
    with manifest_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            label_text = row["label"].strip().lower()
            if label_text not in LABEL_MAP:
                raise ValueError(f"Unsupported label: {row['label']!r}")
            writer.writerow(
                {
                    "audio_path": str((subset_dir / row["name"]).resolve()),
                    "label": LABEL_MAP[label_text],
                    "type": row["type"],
                    "generator": row.get("generator", ""),
                    "source_name": row["name"],
                }
            )


def main() -> None:
    args = parse_args()

    train_rows = read_rows(args.train_label_csv)
    dev_rows = read_rows(args.dev_label_csv)
    chosen_train = choose_rows(train_rows, per_group=args.train_per_group, seed=args.seed)
    chosen_dev = choose_rows(dev_rows, per_group=args.dev_per_group, seed=args.seed)

    train_dir = args.output_root / "train"
    dev_dir = args.output_root / "dev"
    extract_subset(chosen_train, tar_path=args.train_tar, target_dir=train_dir)
    extract_subset(chosen_dev, tar_path=args.dev_tar, target_dir=dev_dir)

    train_manifest = args.manifests_dir / args.train_manifest_name
    dev_manifest = args.manifests_dir / args.dev_manifest_name
    write_manifest(chosen_train, subset_dir=train_dir, manifest_path=train_manifest)
    write_manifest(chosen_dev, subset_dir=dev_dir, manifest_path=dev_manifest)

    print(
        {
            "train_rows": len(chosen_train),
            "dev_rows": len(chosen_dev),
            "train_manifest": str(train_manifest.resolve()),
            "dev_manifest": str(dev_manifest.resolve()),
            "output_root": str(args.output_root.resolve()),
        }
    )


if __name__ == "__main__":
    main()
