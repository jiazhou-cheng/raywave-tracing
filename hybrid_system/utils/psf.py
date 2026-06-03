from __future__ import annotations

from typing import Optional
import math
import torch
from .common import ifft2c, complex_dtype_for

def huygens_psf_gridded(
    o: torch.Tensor,
    d: torch.Tensor,
    opl: torch.Tensor,
    amp: Optional[torch.Tensor],
    wavelength_mm: float,
    sensor_pitch_mm: float,
    sensor_grid_size: int,
    oversamp: float = 1.0,
) -> torch.Tensor:
    device = o.device
    real_dtype = o.dtype
    cplx_dtype = complex_dtype_for(real_dtype)
    ks = int(sensor_grid_size)
    pitch = float(sensor_pitch_mm)
    extent = (ks // 2) * pitch

    on_sensor = (o[:, 0] >= -extent) & (o[:, 0] <= extent) & (o[:, 1] >= -extent) & (o[:, 1] <= extent)
    if not torch.any(on_sensor):
        return torch.zeros((ks, ks), device=device, dtype=cplx_dtype)

    o = o[on_sensor]
    d = d[on_sensor]
    opl = opl[on_sensor]
    if amp is None:
        amp = torch.ones((o.shape[0],), device=device, dtype=cplx_dtype)
    else:
        amp = amp[on_sensor].reshape(-1).to(cplx_dtype)

    k = 2.0 * math.pi / float(wavelength_mm)
    kx_s = k * d[:, 0]
    ky_s = k * d[:, 1]
    piston = -(kx_s * o[:, 0] + ky_s * o[:, 1])
    weights_c = amp * torch.exp(1j * (k * opl + piston).to(cplx_dtype))

    os_size = int(math.ceil(ks * float(oversamp)))
    dk = 2.0 * math.pi / (os_size * pitch)
    inband = (kx_s.abs() < math.pi / pitch) & (ky_s.abs() < math.pi / pitch)
    if not torch.any(inband):
        return torch.zeros((ks, ks), device=device, dtype=cplx_dtype)

    kx_s = kx_s[inband]
    ky_s = ky_s[inband]
    weights_c = weights_c[inband]
    ix_f = kx_s / dk + os_size * 0.5
    iy_f = ky_s / dk + os_size * 0.5
    ix0 = torch.floor(ix_f).long().clamp(0, os_size - 1)
    iy0 = torch.floor(iy_f).long().clamp(0, os_size - 1)
    ix1 = (ix0 + 1).clamp(0, os_size - 1)
    iy1 = (iy0 + 1).clamp(0, os_size - 1)
    tx = (ix_f - ix0.to(ix_f.dtype)).clamp(0, 1)
    ty = (iy_f - iy0.to(iy_f.dtype)).clamp(0, 1)

    ids = (iy0 * os_size + ix0, iy0 * os_size + ix1, iy1 * os_size + ix0, iy1 * os_size + ix1)
    ws = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)
    real = torch.zeros((os_size * os_size,), device=device, dtype=real_dtype)
    imag = torch.zeros((os_size * os_size,), device=device, dtype=real_dtype)
    for idx, weight in zip(ids, ws):
        real.index_add_(0, idx, weight * weights_c.real.to(real_dtype))
        imag.index_add_(0, idx, weight * weights_c.imag.to(real_dtype))

    field_os = ifft2c(torch.complex(real, imag).reshape(os_size, os_size))
    if os_size == ks:
        return field_os
    start = (os_size - ks) // 2
    return field_os[start : start + ks, start : start + ks].contiguous()
