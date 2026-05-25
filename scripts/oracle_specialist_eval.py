from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def load_predictions(path: Path) -> dict[str, str]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        return {row["name"]: row["predict"] for row in reader}


def label_to_int(text: str) -> int:
    t = text.strip().lower()
    if t == "real":
        return 0
    if t == "fake":
        return 1
    raise ValueError(f"Unsupported label: {text!r}")


def macro_f1(y_true: list[int], y_pred: list[int]) -> float:
    eps = 1e-12
    scores = []
    for c in (0, 1):
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp == c)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt != c and yp == c)
        fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == c and yp != c)
        precision = tp / (tp + fp + eps)
        recall = tp / (tp + fn + eps)
        scores.append(2 * precision * recall / (precision + recall + eps))
    return float(sum(scores) / len(scores))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate an oracle router that uses specialists for chosen types.")
    p.add_argument("--manifest", required=True, type=Path)
    p.add_argument("--base-predict-csv", required=True, type=Path)
    p.add_argument("--specialist-predict-csv", required=True, type=Path)
    p.add_argument("--specialist-type", action="append", required=True, dest="specialist_types")
    p.add_argument("--output-json", required=True, type=Path)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    specialist_types = {x.strip() for x in args.specialist_types}
    base_pred = load_predictions(args.base_predict_csv)
    specialist_pred = load_predictions(args.specialist_predict_csv)

    type_true: dict[str, list[int]] = {}
    type_pred: dict[str, list[int]] = {}
    total_true: list[int] = []
    total_pred: list[int] = []
    routed_count = 0
    total_rows = 0

    with args.manifest.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            total_rows += 1
            name = row.get("source_name") or Path(row["audio_path"]).name
            audio_type = row.get("type", "")
            y_true = int(row["label"])
            chosen = specialist_pred if audio_type in specialist_types else base_pred
            if audio_type in specialist_types:
                routed_count += 1
            if name not in chosen:
                raise KeyError(f"Missing prediction for {name} in selected predictor")
            y_pred = label_to_int(chosen[name])

            total_true.append(y_true)
            total_pred.append(y_pred)
            type_true.setdefault(audio_type, []).append(y_true)
            type_pred.setdefault(audio_type, []).append(y_pred)

    type_scores = {
        audio_type: macro_f1(type_true[audio_type], type_pred[audio_type])
        for audio_type in sorted(type_true)
    }
    summary = {
        "rows": total_rows,
        "routed_rows": routed_count,
        "specialist_types": sorted(specialist_types),
        "macro_f1": macro_f1(total_true, total_pred),
        "type_macro_f1": type_scores,
    }

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    with args.output_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)

    print(summary)


if __name__ == "__main__":
    main()
