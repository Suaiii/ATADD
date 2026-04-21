from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel


class AudioBackboneClassifier(nn.Module):
    def __init__(
        self,
        pretrained_name: str,
        num_classes: int,
        dropout: float = 0.1,
        freeze_backbone: bool = True,
    ) -> None:
        super().__init__()
        self.backbone = AutoModel.from_pretrained(pretrained_name)
        hidden_size = int(self.backbone.config.hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, input_values: torch.Tensor) -> torch.Tensor:
        out = self.backbone(input_values=input_values)
        pooled = out.last_hidden_state.mean(dim=1)
        logits = self.classifier(self.dropout(pooled))
        return logits

