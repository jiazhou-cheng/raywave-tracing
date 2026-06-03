from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from .refractive import EPS, propagate_to_plane
from .common import nan_to_num_complex, ifft2c, empty_ray_bundle, fft2c, complex_dtype_for

def _centered_grid(
    nx: int,
    ny: int,
    dx_mm: float,
    dy_mm: float,
    *,
    device: torch.device,
    dtype: torch.dtype,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
) -> Tuple[torch.Tensor, torch.Tensor]:
    xs = (torch.arange(nx, device=device, dtype=dtype) - (nx - 1) / 2.0) * float(dx_mm) + float(origin_xy[0])
    ys = (torch.arange(ny, device=device, dtype=dtype) - (ny - 1) / 2.0) * float(dy_mm) + float(origin_xy[1])
    return torch.meshgrid(xs, ys, indexing="xy")


def _circular_mask_from_grid(X: torch.Tensor, Y: torch.Tensor, radius_mm: float) -> torch.Tensor:
    return X * X + Y * Y <= float(radius_mm) ** 2


def _phase_to_complex_field(phase_rad: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    cplx_dtype = complex_dtype_for(phase_rad.dtype)
    field = torch.exp(1j * phase_rad.to(cplx_dtype))
    if mask is not None:
        field = field * mask.to(device=phase_rad.device, dtype=cplx_dtype)
    return field


@dataclass
class FlatDoeSurface:
    """Ray-wave DOE plane: grid sampling, field synthesis, and primary/secondary ray emission."""

    z_mm: float
    pitch_mm: float
    wavelength_mm: float
    aperture_radius_mm: float
    origin_xy: Tuple[float, float] = (0.0, 0.0)
    out_primary_ray_count: int = 1024
    sampled_secondary_ray_count: int = 4
    preserve_energy: bool = True
    n_before: float = 1.0
    n_after: float = 1.0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        device = self.device if self.device is not None else torch.device("cpu")
        dtype = self.dtype if self.dtype is not None else torch.float32
        self.z_mm = float(self.z_mm)
        self.pitch_mm = float(self.pitch_mm)
        self.wavelength_mm = float(self.wavelength_mm)
        self.aperture_radius_mm = float(self.aperture_radius_mm)
        self.origin_xy = (float(self.origin_xy[0]), float(self.origin_xy[1]))
        self.device = device
        self.dtype = dtype

    @property
    def dx_mm(self) -> float:
        return self.pitch_mm

    @property
    def dy_mm(self) -> float:
        return self.pitch_mm

    @classmethod
    def plane(
        cls,
        *,
        z_mm: float,
        pitch_mm: float,
        wavelength_mm: float,
        aperture_radius_mm: float,
        origin_xy: Tuple[float, float] = (0.0, 0.0),
        out_primary_ray_count: int = 1024,
        sampled_secondary_ray_count: int = 4,
        preserve_energy: bool = True,
        n_before: float = 1.0,
        n_after: float = 1.0,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "FlatDoeSurface":
        return cls(
            z_mm,
            pitch_mm,
            wavelength_mm,
            aperture_radius_mm,
            origin_xy,
            out_primary_ray_count,
            sampled_secondary_ray_count,
            preserve_energy,
            n_before,
            n_after,
            device,
            dtype,
        )

    def grid(self, grid_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Centered sample coordinates and circular aperture mask on the DOE raster."""
        X, Y = _centered_grid(
            grid_size,
            grid_size,
            self.dx_mm,
            self.dy_mm,
            device=self.device,
            dtype=self.dtype,
            origin_xy=self.origin_xy,
        )
        mask = _circular_mask_from_grid(X - self.origin_xy[0], Y - self.origin_xy[1], self.aperture_radius_mm)
        return X, Y, mask

    def phase_to_field(self, phase_rad: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        return _phase_to_complex_field(phase_rad, mask)

    def rays_to_field(
        self,
        o: torch.Tensor,
        d: torch.Tensor,
        opl: torch.Tensor,
        amp: Optional[torch.Tensor],
        *,
        field_shape: tuple[int, int],
    ) -> torch.Tensor:
        return _ray_bundle_to_field(
            o,
            d,
            opl,
            amp,
            field_shape=field_shape,
            dx_mm=self.dx_mm,
            dy_mm=self.dy_mm,
            wavelength_mm=self.wavelength_mm,
            origin_xy=self.origin_xy,
            n_ref=self.n_before,
        )

    def field_to_rays(self, field: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Emit primary/secondary rays from a complex field on this DOE plane."""
        H, W = field.shape
        if H != W:
            raise ValueError("FlatDoeSurface.field_to_rays expects a square field.")
        if abs(self.dx_mm - self.dy_mm) > 1e-12:
            raise ValueError("FlatDoeSurface.field_to_rays expects dx_mm == dy_mm.")

        device = field.device
        real_dtype = field.real.dtype
        cplx_dtype = complex_dtype_for(real_dtype)
        primary_count = int(self.out_primary_ray_count)
        secondary_count = int(self.sampled_secondary_ray_count)
        ray_count = primary_count * secondary_count

        xs = (torch.arange(W, device=device, dtype=real_dtype) - (W - 1) / 2.0) * self.dx_mm + self.origin_xy[0]
        ys = (torch.arange(H, device=device, dtype=real_dtype) - (H - 1) / 2.0) * self.dy_mm + self.origin_xy[1]
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        xy_all = torch.stack([X.reshape(-1), Y.reshape(-1)], dim=-1)
        primary_idx = torch.randint(0, xy_all.shape[0], (primary_count,), device=device)
        xy_primary = xy_all[primary_idx]
        if primary_count > 0:
            xy_primary = xy_primary.clone()
            xy_primary[0] = torch.tensor(self.origin_xy, device=device, dtype=real_dtype)
        xy = xy_primary.repeat_interleave(secondary_count, dim=0)

        o = torch.cat([xy, torch.full((ray_count, 1), self.z_mm, device=device, dtype=real_dtype)], dim=-1)
        spectrum = fft2c(field)

        fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=self.dx_mm)).to(device=device, dtype=real_dtype)
        fy = torch.fft.fftshift(torch.fft.fftfreq(H, d=self.dy_mm)).to(device=device, dtype=real_dtype)
        Fx, Fy = torch.meshgrid(fx, fy, indexing="xy")
        Kx = 2.0 * math.pi * Fx
        Ky = 2.0 * math.pi * Fy
        k_exit = float(self.n_after) * (2.0 * math.pi / max(self.wavelength_mm, EPS))

        propagating = Kx * Kx + Ky * Ky < k_exit * k_exit
        pdf = torch.nan_to_num(spectrum.abs() * propagating, nan=0.0, posinf=0.0, neginf=0.0)
        if bool((pdf.sum() <= 0).detach()):
            pdf = torch.zeros_like(pdf)
            pdf[H // 2, W // 2] = 1.0
        flat_pdf = (pdf / pdf.sum().clamp_min(EPS)).reshape(-1).detach()

        sample_idx = torch.multinomial(flat_pdf, ray_count, replacement=True)
        y_idx = sample_idx // W
        x_idx = sample_idx % W
        kx_s = Kx[y_idx, x_idx]
        ky_s = Ky[y_idx, x_idx]
        spec_s = spectrum[y_idx, x_idx].to(cplx_dtype)
        pdf_s = flat_pdf[sample_idx].to(real_dtype).clamp_min(EPS)

        ux = kx_s / max(k_exit, EPS)
        uy = ky_s / max(k_exit, EPS)
        uz = torch.sqrt((1.0 - ux * ux - uy * uy).clamp_min(0.0))
        d = F.normalize(torch.stack([ux, uy, uz], dim=-1), dim=-1)

        x_rel = xy[:, 0] - self.origin_xy[0]
        y_rel = xy[:, 1] - self.origin_xy[1]
        shift_phase = torch.exp(1j * (kx_s * x_rel + ky_s * y_rel).to(cplx_dtype))
        amps = spec_s * shift_phase * (self.dx_mm * self.dy_mm / pdf_s.to(cplx_dtype)) / float(ray_count)
        opl = torch.zeros((ray_count,), device=device, dtype=real_dtype)
        return o, d, nan_to_num_complex(amps.reshape(-1)), opl

    def interact(
        self,
        phase_cplx: torch.Tensor,
        o: torch.Tensor,
        d: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Propagate rays to the DOE, apply phase, and re-sample outgoing rays."""
        hit, valid, opl_inc = propagate_to_plane(o, d, self.z_mm, n=self.n_before)
        device = phase_cplx.device
        real_dtype = phase_cplx.real.dtype
        if not torch.any(valid):
            return empty_ray_bundle(device, real_dtype)

        input_field = self.rays_to_field(
            hit[valid],
            d[valid],
            opl_inc[valid],
            amp=None,
            field_shape=phase_cplx.shape,
        )
        output_field = nan_to_num_complex(input_field * phase_cplx)
        if self.preserve_energy:
            pin = (input_field.abs() ** 2).sum().clamp_min(EPS)
            pout = (output_field.abs() ** 2).sum().clamp_min(EPS)
            output_field = output_field * torch.sqrt(pin / pout).to(output_field.real.dtype)

        return self.field_to_rays(output_field)


def _ray_bundle_to_field(
    o: torch.Tensor,
    d: torch.Tensor,
    opl: torch.Tensor,
    amp: Optional[torch.Tensor],
    *,
    field_shape: tuple[int, int],
    dx_mm: float,
    dy_mm: float,
    wavelength_mm: float,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    n_ref: float = 1.0,
) -> torch.Tensor:
    H, W = field_shape
    if H != W:
        raise ValueError("_ray_bundle_to_field expects a square field.")
    if abs(float(dx_mm) - float(dy_mm)) > 1e-12:
        raise ValueError("_ray_bundle_to_field expects dx_mm == dy_mm.")

    device = o.device
    real_dtype = o.dtype
    cplx_dtype = complex_dtype_for(real_dtype)
    k = float(n_ref) * (2.0 * math.pi / max(float(wavelength_mm), EPS))

    d_norm = d / torch.linalg.norm(d, dim=-1, keepdim=True).clamp_min(EPS)
    kx_s = k * d_norm[:, 0]
    ky_s = k * d_norm[:, 1]
    piston = -(kx_s * (o[:, 0] - float(origin_xy[0])) + ky_s * (o[:, 1] - float(origin_xy[1])))

    if amp is None:
        amp_c = torch.ones((o.shape[0],), device=device, dtype=cplx_dtype)
    elif torch.is_complex(amp):
        amp_c = amp.reshape(-1).to(device=device, dtype=cplx_dtype)
    else:
        amp_c = amp.reshape(-1).to(device=device, dtype=cplx_dtype)
    weights_c = amp_c * torch.exp(1j * (k * opl.reshape(-1) + piston).to(cplx_dtype))

    fx = torch.fft.fftshift(torch.fft.fftfreq(W, d=float(dx_mm))).to(device=device, dtype=real_dtype)
    kx_grid = 2.0 * math.pi * fx
    dk = (kx_grid[1] - kx_grid[0]).abs().clamp_min(EPS)
    kmax_eff = math.pi / float(dx_mm) - dk
    inband = (kx_s.abs() < kmax_eff) & (ky_s.abs() < kmax_eff)
    if not torch.any(inband):
        return torch.zeros((H, W), device=device, dtype=cplx_dtype)

    kx_s = kx_s[inband]
    ky_s = ky_s[inband]
    weights_c = weights_c[inband]
    center = W // 2
    ix_f = kx_s / dk + center
    iy_f = ky_s / dk + center
    valid = (ix_f >= 0) & (ix_f <= W - 2) & (iy_f >= 0) & (iy_f <= H - 2)
    if not torch.any(valid):
        return torch.zeros((H, W), device=device, dtype=cplx_dtype)

    ix_f = ix_f[valid]
    iy_f = iy_f[valid]
    weights_c = weights_c[valid]
    ix0 = torch.floor(ix_f).long()
    iy0 = torch.floor(iy_f).long()
    tx = ix_f - ix0.to(ix_f.dtype)
    ty = iy_f - iy0.to(iy_f.dtype)

    ids = (iy0 * W + ix0, iy0 * W + ix0 + 1, (iy0 + 1) * W + ix0, (iy0 + 1) * W + ix0 + 1)
    ws = ((1 - tx) * (1 - ty), tx * (1 - ty), (1 - tx) * ty, tx * ty)

    real = torch.zeros((H * W,), device=device, dtype=real_dtype)
    imag = torch.zeros((H * W,), device=device, dtype=real_dtype)
    for idx, weight in zip(ids, ws):
        real.index_add_(0, idx, weight * weights_c.real.to(real_dtype))
        imag.index_add_(0, idx, weight * weights_c.imag.to(real_dtype))
    return nan_to_num_complex(ifft2c(torch.complex(real, imag).reshape(H, W)))


def doe_raywave_plane(
    phase_cplx: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    *,
    z_mm: float,
    dx_mm: float,
    dy_mm: float,
    wavelength_mm: float,
    origin_xy: tuple[float, float] = (0.0, 0.0),
    out_primary_ray_count: int = 1024,
    sampled_secondary_ray_count: int = 4,
    preserve_energy: bool = True,
    n_before: float = 1.0,
    n_after: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    surface = FlatDoeSurface.plane(
        z_mm=z_mm,
        pitch_mm=dx_mm,
        wavelength_mm=wavelength_mm,
        aperture_radius_mm=0.5 * int(phase_cplx.shape[0]) * float(dx_mm),
        origin_xy=origin_xy,
        out_primary_ray_count=out_primary_ray_count,
        sampled_secondary_ray_count=sampled_secondary_ray_count,
        preserve_energy=preserve_energy,
        n_before=n_before,
        n_after=n_after,
        device=phase_cplx.device,
        dtype=phase_cplx.real.dtype,
    )
    return surface.interact(phase_cplx, o, d)