from __future__ import annotations

import argparse
import csv
import os
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset

from atadd.config import load_experiment_config
from atadd.modeling import AudioBackboneClassifier
from atadd.utils import ensure_dir, save_json


LABEL_TEXT = {0: "real", 1: "fake"}


@dataclass
class PredictRow:
    audio_path: Path
    name: str


class AudioPredictionDataset(Dataset):
    def __init__(
        self,
        rows: List[PredictRow],
        sample_rate: int,
        max_seconds: float,
    ) -> None:
        self.rows = rows
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_seconds)

    def __len__(self) -> int:
        return len(self.rows)

    def _fix_length(self, wav: torch.Tensor) -> torch.Tensor:
        n = wav.shape[-1]
        if n == self.max_length:
            return wav
        if n > self.max_length:
            return wav[: self.max_length]
        pad = self.max_length - n
        return torch.nn.functional.pad(wav, (0, pad))

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor | str]:
        row = self.rows[idx]
        wav, sr = torchaudio.load(row.audio_path)
        wav = wav.mean(dim=0)
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = self._fix_length(wav)
        return {"input_values": wav, "name": row.name}


def collate_predict_batch(items: List[Dict[str, torch.Tensor | str]]) -> Dict[str, torch.Tensor | List[str]]:
    inputs = torch.stack([x["input_values"] for x in items], dim=0)
    names = [str(x["name"]) for x in items]
    return {"input_values": inputs, "names": names}


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Generate AT-ADD challenge predictions.")
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--output-dir", required=True, type=str)
    p.add_argument("--manifest", default=None, type=str)
    p.add_argument("--audio-dir", default=None, type=str)
    p.add_argument("--device", default="cuda", type=str)
    p.add_argument("--batch-size", default=None, type=int)
    p.add_argument("--num-workers", default=None, type=int)
    p.add_argument("--pattern", default="*.flac", type=str)
    p.add_argument("--offline", action="store_true")
    p.add_argument("--fake-threshold", default=0.5, type=float)
    p.add_argument("--save-probs", action="store_true")
    return p.parse_args()


def read_rows_from_manifest(manifest_path: Path) -> List[PredictRow]:
    rows: List[PredictRow] = []
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        if "audio_path" not in (reader.fieldnames or []):
            raise ValueError(f"Manifest must contain 'audio_path': {manifest_path}")
        for item in reader:
            raw_path = item["audio_path"]
            audio_path = Path(raw_path)
            if not audio_path.is_absolute():
                audio_path = manifest_path.parent / audio_path
            name = item.get("name") or audio_path.name
            rows.append(PredictRow(audio_path=audio_path, name=name))
    return rows


def read_rows_from_audio_dir(audio_dir: Path, pattern: str) -> List[PredictRow]:
    rows = [
        PredictRow(audio_path=path, name=path.name)
        for path in sorted(audio_dir.rglob(pattern))
        if path.is_file()
    ]
    return rows


@torch.no_grad()
def run_predict(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
    fake_threshold: float,
    save_probs: bool,
) -> List[Dict[str, str]]:
    model.eval()
    predictions: List[Dict[str, str]] = []
    for batch in loader:
        x = batch["input_values"].to(device)
        names = batch["names"]
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        fake_probs = probs[:, 1]
        pred = (fake_probs >= fake_threshold).long().tolist()
        probs_cpu = probs.detach().cpu().tolist()
        for name, pred_id, prob_row in zip(names, pred, probs_cpu):
            label_text = LABEL_TEXT.get(int(pred_id))
            if label_text is None:
                raise ValueError(f"Unsupported predicted label id: {pred_id}")
            row = {"name": name, "predict": label_text}
            if save_probs:
                row["prob_real"] = f"{float(prob_row[0]):.8f}"
                row["prob_fake"] = f"{float(prob_row[1]):.8f}"
            predictions.append(row)
    return predictions


def write_predict_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    fieldnames = ["name", "predict"]
    if rows and "prob_fake" in rows[0]:
        fieldnames.extend(["prob_real", "prob_fake"])
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_submission_csv(path: Path, rows: List[Dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["name", "predict"])
        writer.writeheader()
        for row in rows:
            writer.writerow({"name": row["name"], "predict": row["predict"]})


def write_submission_zip(zip_path: Path, csv_path: Path) -> None:
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write(csv_path, arcname="predict.csv")


def main() -> None:
    args = parse_args()
    if bool(args.manifest) == bool(args.audio_dir):
        raise ValueError("Provide exactly one of --manifest or --audio-dir.")

    if args.offline:
        os.environ["HF_HUB_OFFLINE"] = "1"
        os.environ["TRANSFORMERS_OFFLINE"] = "1"

    cfg = load_experiment_config(args.config)
    if args.batch_size is not None:
        cfg.train.batch_size = args.batch_size
    if args.num_workers is not None:
        cfg.train.num_workers = args.num_workers

    if args.manifest:
        rows = read_rows_from_manifest(Path(args.manifest))
        input_source = str(Path(args.manifest).resolve())
    else:
        rows = read_rows_from_audio_dir(Path(args.audio_dir), pattern=args.pattern)
        input_source = str(Path(args.audio_dir).resolve())

    if not rows:
        raise ValueError("No audio files found for prediction.")

    out_dir = ensure_dir(args.output_dir)
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    ds = AudioPredictionDataset(
        rows=rows,
        sample_rate=cfg.data.sample_rate,
        max_seconds=cfg.data.max_seconds,
    )
    loader = DataLoader(
        ds,
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.train.num_workers,
        collate_fn=collate_predict_batch,
        pin_memory=(device.type == "cuda"),
    )

    model = AudioBackboneClassifier(
        pretrained_name=cfg.model.pretrained_name,
        num_classes=cfg.data.num_classes,
        dropout=cfg.model.dropout,
        freeze_backbone=False,
        kind=cfg.model.kind,
        feature_extractor_name=cfg.model.feature_extractor_name,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state"], strict=True)

    predictions = run_predict(
        model,
        loader=loader,
        device=device,
        fake_threshold=args.fake_threshold,
        save_probs=args.save_probs,
    )

    predict_csv = out_dir / "predict.csv"
    submission_zip = out_dir / "submission.zip"
    write_predict_csv(predict_csv, predictions)
    submission_csv = predict_csv
    if args.save_probs:
        submission_csv = out_dir / "submission_predict.csv"
        write_submission_csv(submission_csv, predictions)
    write_submission_zip(submission_zip, submission_csv)

    summary = {
        "model": cfg.model.name,
        "config_path": str(Path(args.config).resolve()),
        "checkpoint": str(Path(args.checkpoint).resolve()),
        "input_source": input_source,
        "num_files": len(predictions),
        "predict_csv": str(predict_csv.resolve()),
        "submission_zip": str(submission_zip.resolve()),
        "fake_threshold": args.fake_threshold,
        "save_probs": args.save_probs,
    }
    save_json(out_dir / "predict_summary.json", summary)
    print(summary)


if __name__ == "__main__":
    main()
