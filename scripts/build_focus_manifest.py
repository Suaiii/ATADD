from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path


def parse_type_multipliers(items: list[str]) -> dict[str, int]:
    out: dict[str, int] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"Expected TYPE=MULTIPLIER, got: {item!r}")
        audio_type, multiplier = item.split("=", 1)
        audio_type = audio_type.strip()
        mult = int(multiplier)
        if mult < 1:
            raise ValueError(f"Multiplier must be >= 1, got {mult} for {audio_type}")
        out[audio_type] = mult
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Build a type-focused manifest without duplicating audio files.")
    p.add_argument("--source-manifest", required=True, type=Path)
    p.add_argument("--output-manifest", required=True, type=Path)
    p.add_argument(
        "--type-multiplier",
        action="append",
        default=[],
        help="Repeat rows from a given type. Example: --type-multiplier music=3 --type-multiplier sound=2",
    )
    p.add_argument("--seed", default=42, type=int)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    multipliers = parse_type_multipliers(args.type_multiplier)
    rng = random.Random(args.seed)

    with args.source_manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if not fieldnames:
            raise ValueError(f"Empty manifest header: {args.source_manifest}")
        if "type" not in fieldnames:
            raise ValueError(f"Manifest must contain 'type': {args.source_manifest}")
        rows = list(reader)

    expanded_rows: list[dict[str, str]] = []
    for row in rows:
        audio_type = row["type"]
        repeat = multipliers.get(audio_type, 1)
        for _ in range(repeat):
            expanded_rows.append(dict(row))

    rng.shuffle(expanded_rows)
    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(expanded_rows)

    type_counts: dict[str, int] = {}
    for row in expanded_rows:
        audio_type = row["type"]
        type_counts[audio_type] = type_counts.get(audio_type, 0) + 1

    print(
        {
            "source_manifest": str(args.source_manifest.resolve()),
            "output_manifest": str(args.output_manifest.resolve()),
            "rows": len(expanded_rows),
            "type_counts": type_counts,
            "multipliers": multipliers,
        }
    )


if __name__ == "__main__":
    main()
