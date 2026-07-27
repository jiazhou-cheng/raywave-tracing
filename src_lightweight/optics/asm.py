from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    import numpy as np


def FT(data: torch.Tensor) -> torch.Tensor:
    """Centered 2D Fourier transform."""
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(data, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def iFT(data: torch.Tensor) -> torch.Tensor:
    """Centered 2D inverse Fourier transform."""
    return torch.fft.ifftshift(
        torch.fft.ifftn(torch.fft.fftshift(data, dim=(-2, -1)), dim=(-2, -1)),
        dim=(-2, -1),
    )


def asm(
    field: torch.Tensor,
    z_mm: float,
    dx_mm: float,
    dy_mm: float,
    wavelength_mm: float,
    *,
    padding: bool = True,
    device: str = "cpu",
) -> torch.Tensor:
    """Propagate a complex field [Ny, Nx] using the angular spectrum method."""
    field = field.to(torch.complex64)
    ny, nx = field.shape
    if nx != ny:
        raise ValueError("asm() expects a square field.")

    zn = nx // 2
    if padding:
        field = F.pad(field, (zn, zn, zn, zn), mode="constant", value=0)

    ny_pad, nx_pad = field.shape
    fx = torch.fft.fftshift(torch.fft.fftfreq(nx_pad, d=dx_mm)).to(device)
    fy = torch.fft.fftshift(torch.fft.fftfreq(ny_pad, d=dy_mm)).to(device)
    fxx, fyy = torch.meshgrid(fx, fy, indexing="xy")
    fxx = fxx.to(torch.float32)
    fyy = fyy.to(torch.float32)

    spectrum = FT(field)
    invlam = torch.tensor(1.0 / wavelength_mm, device=device, dtype=torch.float32)
    gamma = invlam**2 - fxx**2 - fyy**2
    kz = torch.sqrt(torch.clamp(gamma, min=0.0))
    transfer = torch.exp(1j * 2 * math.pi * kz * z_mm).to(torch.complex64)
    transfer = torch.where(gamma >= 0, transfer, torch.zeros_like(transfer))
    out = iFT(spectrum * transfer)

    if padding:
        out = out[zn : ny_pad - zn, zn : nx_pad - zn]
    return out
