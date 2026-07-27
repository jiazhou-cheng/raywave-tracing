"""Adam-based phase-only retrieval using angular-spectrum propagation."""

from __future__ import annotations

import math

import numpy as np
import torch

from src_lightweight.optics.asm import asm


def _pupil_amplitude(h: int, w: int, device: torch.device, mask_radius: float | None) -> torch.Tensor:
    if mask_radius is None:
        return torch.ones((h, w), dtype=torch.float32, device=device)
    yy, xx = torch.meshgrid(
        torch.arange(h, device=device),
        torch.arange(w, device=device),
        indexing="ij",
    )
    cy, cx = h // 2, w // 2
    return (((xx - cx) ** 2 + (yy - cy) ** 2) <= (mask_radius**2)).float()


def _focal_plane_amplitude(
    phase_rad: torch.Tensor,
    pupil_amp: torch.Tensor,
    *,
    dist_mm: float,
    dx_mm: float,
    dy_mm: float,
    wavelength_um: float,
    device: torch.device,
) -> torch.Tensor:
    field = pupil_amp.to(torch.float32) * torch.exp(1j * phase_rad.to(torch.float32))
    wvl_mm = float(wavelength_um) * 1e-3
    propagated = asm(field, dist_mm, dx_mm, dy_mm, wvl_mm, padding=True, device=str(device))
    return torch.abs(propagated)


def retrieve_phase(
    target_intensity: np.ndarray | torch.Tensor,
    *,
    wavelength_um: float,
    dx_mm: float,
    dy_mm: float,
    dist_mm: float,
    mask_radius: float | None = None,
    max_iter: int = 500,
    learning_rate: float = 1e-3,
    device: str | torch.device = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, list[float]]:
    """
    Normal phase-only retrieval (no aberration correction term).

    Optimizes pupil phase with Adam so the ASM-propagated amplitude matches
    ``target_intensity`` using scale-invariant amplitude matching.

    Args:
        target_intensity: Target intensity map ``[H, W]`` (square).
        max_iter: Adam iterations.
        learning_rate: Adam learning rate.
        wavelength_um: Vacuum wavelength [µm].
        dx_mm, dy_mm: Sample pitch on the SLM/pupil grid [mm].
        dist_mm: Propagation distance [mm].
        mask_radius: Circular pupil radius in pixels, or ``None`` for uniform illumination.
        device: Torch device.

    Returns:
        pm_result: Retrieved phase ``[H, W]`` [rad] on CPU.
        apm_result: Complex pupil field ``am * exp(i * phase)`` ``[H, W]`` on CPU.
        loss_history: Per-iteration loss values.
    """
    device_t = torch.device(device)
    target = torch.as_tensor(target_intensity, dtype=torch.float32, device=device_t)
    if target.ndim != 2:
        raise ValueError(f"target_intensity must be 2D, got shape {tuple(target.shape)}.")
    h, w = int(target.shape[0]), int(target.shape[1])
    if h != w:
        raise ValueError(f"Phase retrieval expects a square grid; got {(h, w)}.")
    if not math.isfinite(float(dist_mm)):
        raise ValueError("Only finite dist_mm is supported for ASM phase retrieval.")

    eps = 1e-12
    target_I = target / (target.sum() + eps)
    target_A = torch.sqrt(target_I + eps)
    w_fg = 0.2 + 0.8 * (target_I / (target_I.max() + eps))

    pm = (2 * math.pi * torch.rand(h, w, device=device_t) - math.pi).requires_grad_(True)

    pupil_amp = _pupil_amplitude(h, w, device_t, mask_radius)
    opt = torch.optim.Adam([pm], lr=float(learning_rate))
    loss_history: list[float] = []

    for _ in range(int(max_iter)):
        opt.zero_grad(set_to_none=True)
        pred_A = _focal_plane_amplitude(
            pm,
            pupil_amp,
            dist_mm=float(dist_mm),
            dx_mm=float(dx_mm),
            dy_mm=float(dy_mm),
            wavelength_um=float(wavelength_um),
            device=device_t,
        )

        pa = pred_A / (pred_A.norm() + eps)
        ta = target_A / (target_A.norm() + eps)
        loss = (w_fg * (pa - ta).pow(2)).mean()

        loss.backward()
        opt.step()
        with torch.no_grad():
            pm[:] = (pm + math.pi) % (2 * math.pi) - math.pi
        loss_history.append(float(loss.detach().cpu()))

    with torch.no_grad():
        pm_result = pm.detach().cpu()
        apm_result = (pupil_amp * torch.exp(1j * pm)).detach().cpu()

    return pm_result, apm_result, loss_history
