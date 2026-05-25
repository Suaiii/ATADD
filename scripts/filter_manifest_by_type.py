from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Filter a manifest by one or more audio types.")
    p.add_argument("--source-manifest", required=True, type=Path)
    p.add_argument("--output-manifest", required=True, type=Path)
    p.add_argument("--type", action="append", required=True, dest="types")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    wanted = {x.strip() for x in args.types}

    with args.source_manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        fieldnames = list(reader.fieldnames or [])
        if "type" not in fieldnames:
            raise ValueError(f"Manifest must contain 'type': {args.source_manifest}")
        rows = [row for row in reader if row.get("type") in wanted]

    args.output_manifest.parent.mkdir(parents=True, exist_ok=True)
    with args.output_manifest.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(
        {
            "source_manifest": str(args.source_manifest.resolve()),
            "output_manifest": str(args.output_manifest.resolve()),
            "types": sorted(wanted),
            "rows": len(rows),
        }
    )


if __name__ == "__main__":
    main()
