from __future__ import annotations

import torch

EPS = 1e-12


def loss_from_field_sum(field_sum: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    intensity = field_sum.abs().to(target.dtype) ** 2
    if intensity.shape != target.shape:
        raise ValueError(f"Intensity shape {tuple(intensity.shape)} does not match target {tuple(target.shape)}.")
    # scale = (torch.sum(intensity * target) / (torch.sum(intensity * intensity) + EPS)).detach()
    intensity /= intensity.sum().clamp_min(EPS)
    target /= target.sum().clamp_min(EPS)
    return torch.norm(intensity - target) ** 2
