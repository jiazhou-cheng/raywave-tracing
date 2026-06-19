from __future__ import annotations

from typing import Optional
import math
import torch
from ..common import ifft2c, complex_dtype_for

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

    d_norm = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(1e-12)
    sensor_normal = torch.tensor([0.0, 0.0, 1.0], device=device, dtype=real_dtype)
    obliquity = torch.clamp(torch.sum(d_norm * sensor_normal[None, :], dim=-1), min=0.0)

    k = 2.0 * math.pi / float(wavelength_mm)
    kx_s = k * d[:, 0]
    ky_s = k * d[:, 1]
    piston = -(kx_s * o[:, 0] + ky_s * o[:, 1])
    phase = torch.exp(1j * (k * opl + piston).to(cplx_dtype))
    weights_c = amp * obliquity.to(cplx_dtype) * phase

    os_size = int(math.ceil(ks * float(oversamp)))
    fx = torch.fft.fftshift(torch.fft.fftfreq(os_size, d=pitch)).to(device, dtype=real_dtype)
    kx_grid = 2 * math.pi * fx
    dk = float(kx_grid[1] - kx_grid[0])
    kmax = math.pi / pitch
    kmax_eff = kmax - dk

    inband = (kx_s.abs() < kmax_eff) & (ky_s.abs() < kmax_eff)
    if not torch.any(inband):
        return torch.zeros((ks, ks), device=device, dtype=cplx_dtype)

    kx_s = kx_s[inband]
    ky_s = ky_s[inband]
    weights_c = weights_c[inband]

    center = os_size // 2
    ix_f = (kx_s / dk) + center
    iy_f = (ky_s / dk) + center

    valid_splat = (ix_f >= 0) & (ix_f <= (os_size - 2)) & (iy_f >= 0) & (iy_f <= (os_size - 2))
    if not torch.any(valid_splat):
        return torch.zeros((ks, ks), device=device, dtype=cplx_dtype)

    ix_f = ix_f[valid_splat]
    iy_f = iy_f[valid_splat]
    weights_c = weights_c[valid_splat]

    ix0 = torch.floor(ix_f).long()
    iy0 = torch.floor(iy_f).long()
    tx = ix_f - ix0.to(ix_f.dtype)
    ty = iy_f - iy0.to(iy_f.dtype)
    ix1 = ix0 + 1
    iy1 = iy0 + 1

    def lin(ix, iy):
        return iy * os_size + ix

    idx00 = lin(ix0, iy0)
    idx10 = lin(ix1, iy0)
    idx01 = lin(ix0, iy1)
    idx11 = lin(ix1, iy1)

    w00 = (1 - tx) * (1 - ty)
    w10 = tx * (1 - ty)
    w01 = (1 - tx) * ty
    w11 = tx * ty

    real = torch.zeros((os_size * os_size,), device=device, dtype=real_dtype)
    imag = torch.zeros((os_size * os_size,), device=device, dtype=real_dtype)
    wr = weights_c.real.to(real_dtype)
    wi = weights_c.imag.to(real_dtype)

    real.index_add_(0, idx00, w00 * wr)
    real.index_add_(0, idx10, w10 * wr)
    real.index_add_(0, idx01, w01 * wr)
    real.index_add_(0, idx11, w11 * wr)
    imag.index_add_(0, idx00, w00 * wi)
    imag.index_add_(0, idx10, w10 * wi)
    imag.index_add_(0, idx01, w01 * wi)
    imag.index_add_(0, idx11, w11 * wi)

    field_os = ifft2c(torch.complex(real, imag).reshape(os_size, os_size))
    if os_size == ks:
        return field_os
    start = (os_size - ks) // 2
    return field_os[start : start + ks, start : start + ks].contiguous()
