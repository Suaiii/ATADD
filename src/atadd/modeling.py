from __future__ import annotations

from typing import Optional

import torch
from torch import nn
from transformers import AutoFeatureExtractor, AutoModel


class AudioBackboneClassifier(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        num_classes: int,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
        kind: str = "waveform",
        feature_extractor_name: Optional[str] = None,
    ) -> None:
        super().__init__()
        # Some local pretrained bundles, such as MERT, ship custom model code.
        self.backbone = AutoModel.from_pretrained(pretrained_name, trust_remote_code=True)
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        self.kind = kind
        if kind == "spectrogram":
            fe_name = feature_extractor_name or pretrained_name
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(fe_name)
            self.fe_sampling_rate = int(getattr(self.feature_extractor, "sampling_rate", 16000))
        elif kind == "waveform":
            self.feature_extractor = None
            self.fe_sampling_rate = None
        else:
            raise ValueError(f"Unsupported model kind: {kind!r}")

    def extract_features(self, input_values: torch.Tensor) -> torch.Tensor:
        if self.feature_extractor is not None:
            device = input_values.device
            wav_list = [w.detach().cpu().numpy() for w in input_values]
            fe_out = self.feature_extractor(
                wav_list,
                sampling_rate=self.fe_sampling_rate,
                return_tensors="pt",
            )
            backbone_input = fe_out["input_values"].to(device)
        else:
            backbone_input = input_values

        out = self.backbone(input_values=backbone_input)
        return out.last_hidden_state

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        hidden = self.extract_features(input_values)
        pooled = hidden.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits
