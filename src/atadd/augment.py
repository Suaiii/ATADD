from __future__ import annotations

import math
import random

import torch


class WaveformAugment:
    def __init__(
        self,
        enable: bool = False,
        noise_prob: float = 0.5,
        noise_scale: float = 0.003,
        gain_prob: float = 0.5,
        gain_db: float = 6.0,
    ) -> None:
        self.enable = enable
        self.noise_prob = noise_prob
        self.noise_scale = noise_scale
        self.gain_prob = gain_prob
        self.gain_db = gain_db

    def __call__(self, wav: torch.Tensor) -> torch.Tensor:
        if not self.enable:
            return wav

        out = wav
        if random.random() < self.noise_prob:
            noise = torch.randn_like(out) * self.noise_scale
            out = out + noise

        if random.random() < self.gain_prob:
            db = random.uniform(-self.gain_db, self.gain_db)
            out = out * (10.0 ** (db / 20.0))

        return torch.clamp(out, min=-1.0, max=1.0)

