from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torchaudio
from torch.utils.data import Dataset


@dataclass
class ManifestRow:
    audio_path: Path
    label: int


class AudioClassificationDataset(Dataset):
    def __init__(
        self,
        manifest_path: str,
        sample_rate: int,
        max_seconds: float,
        audio_column: str = "audio_path",
        label_column: str = "label",
        augment=None,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.sample_rate = sample_rate
        self.max_length = int(sample_rate * max_seconds)
        self.audio_column = audio_column
        self.label_column = label_column
        self.augment = augment
        self.rows = self._read_manifest()

    def _read_manifest(self) -> List[ManifestRow]:
        rows: List[ManifestRow] = []
        with self.manifest_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for item in reader:
                raw_audio_path = item[self.audio_column]
                audio_path = Path(raw_audio_path)
                if not audio_path.is_absolute():
                    audio_path = self.manifest_path.parent / audio_path
                label = int(item[self.label_column])
                rows.append(ManifestRow(audio_path=audio_path, label=label))
        if not rows:
            raise ValueError(f"Empty manifest: {self.manifest_path}")
        return rows

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

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.rows[idx]
        wav, sr = torchaudio.load(row.audio_path)
        wav = wav.mean(dim=0)  # mono
        if sr != self.sample_rate:
            wav = torchaudio.functional.resample(wav, sr, self.sample_rate)
        wav = self._fix_length(wav)
        if self.augment is not None:
            wav = self.augment(wav)
        return {
            "input_values": wav,
            "label": torch.tensor(row.label, dtype=torch.long),
        }


def collate_audio_batch(items: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    inputs = torch.stack([x["input_values"] for x in items], dim=0)
    labels = torch.stack([x["label"] for x in items], dim=0)
    return {"input_values": inputs, "labels": labels}

