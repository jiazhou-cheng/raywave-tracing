from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F 
from ..common import complex_dtype_for

EPS = 1e-12
NEWTONS_MAXITER = 10
NEWTONS_TOL_TIGHT = 25e-6
NEWTONS_TOL_LOOSE = 50e-6
NEWTONS_STEP_BOUND = 5.0


@dataclass
class CurvedRefractiveSurface:
    radius_mm: float
    curvature_mm_inv: float = 0.0
    z_offset_mm: float = 0.0
    device: Optional[torch.device] = None
    dtype: Optional[torch.dtype] = None

    def __post_init__(self) -> None:
        device = self.device if self.device is not None else torch.device("cpu")
        dtype = self.dtype if self.dtype is not None else torch.float32
        self.radius_mm = float(self.radius_mm)
        self.device = device
        self.dtype = dtype
        self.c = torch.tensor(float(self.curvature_mm_inv), device=device, dtype=dtype)
        self.z0 = torch.tensor(float(self.z_offset_mm), device=device, dtype=dtype)

    @classmethod
    def sphere(
        cls,
        *,
        radius_mm: float,
        curvature_mm_inv: float,
        z_offset_mm: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "CurvedRefractiveSurface":
        return cls(radius_mm, curvature_mm_inv, z_offset_mm, device, dtype)

    @classmethod
    def flat(
        cls,
        *,
        radius_mm: float,
        z_offset_mm: float,
        device: Optional[torch.device] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> "CurvedRefractiveSurface":
        return cls(radius_mm, 0.0, z_offset_mm, device, dtype)

    def sag(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        r2 = x * x + y * y
        c = self.c.to(device=x.device, dtype=x.dtype)
        if torch.abs(c).item() == 0.0:
            return torch.zeros_like(r2) + self.z0.to(device=x.device, dtype=x.dtype)
        under = torch.clamp(1.0 - c * c * r2, min=EPS)
        return c * r2 / (1.0 + torch.sqrt(under)) + self.z0.to(device=x.device, dtype=x.dtype)

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        r2 = x * x + y * y
        c = self.c.to(device=x.device, dtype=x.dtype)
        if torch.abs(c).item() == 0.0:
            return torch.zeros_like(x), torch.zeros_like(y)
        under = torch.clamp(1.0 - c * c * r2, min=EPS)
        s = torch.sqrt(under)
        dsdr2 = c / (1.0 + s) + c * r2 * c * c / (2.0 * s * (1.0 + s) ** 2)
        return 2.0 * x * dsdr2, 2.0 * y * dsdr2

    def normal_vec_left(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dsdx, dsdy = self.grad(x, y)
        n = torch.stack([dsdx, dsdy, -torch.ones_like(dsdx)], dim=-1)
        return F.normalize(n, p=2, dim=-1)

    def is_within_data_range(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        c = self.c.to(device=x.device, dtype=x.dtype)
        if torch.abs(c).item() == 0.0:
            return torch.ones_like(x, dtype=torch.bool)
        return (x * x + y * y) < 1.0 / (c * c + EPS)


def propagate_to_plane(
    o: torch.Tensor,
    d: torch.Tensor,
    z_plane: float | torch.Tensor,
    n: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    z = torch.as_tensor(z_plane, device=o.device, dtype=o.dtype)
    t = (z - o[:, 2]) / (d[:, 2] + EPS)
    valid = torch.isfinite(t) & torch.isfinite(o).all(dim=1) & torch.isfinite(d).all(dim=1) & (t >= 0)
    hit = torch.where(valid.unsqueeze(-1), o + t.unsqueeze(-1) * d, o)
    opl = torch.where(valid, t * float(n), torch.zeros_like(t))
    return hit, valid, opl


def propagate_to_surface(
    surface: CurvedRefractiveSurface,
    o: torch.Tensor,
    d: torch.Tensor,
    n: float = 1.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    z0 = surface.z0.to(device=o.device, dtype=o.dtype)
    t0 = (z0 - o[:, 2]) / (d[:, 2] + EPS)

    with torch.no_grad():
        t = t0.clone()
        for _ in range(NEWTONS_MAXITER):
            p = o + t.unsqueeze(-1) * d
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            residual = surface.sag(x, y) - z
            dsdx, dsdy = surface.grad(x, y)
            dfdt = dsdx * d[:, 0] + dsdy * d[:, 1] - d[:, 2]
            t = t - torch.clamp(residual / (dfdt + EPS), -NEWTONS_STEP_BOUND, NEWTONS_STEP_BOUND)
            valid_eval = surface.is_within_data_range(x, y) & torch.isfinite(p).all(dim=1)
            if valid_eval.any() and (torch.abs(residual[valid_eval]) <= NEWTONS_TOL_LOOSE).all():
                break
        t_delta = t - t0

    t = t0 + t_delta
    p = o + t.unsqueeze(-1) * d
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    residual = surface.sag(x, y) - z
    dsdx, dsdy = surface.grad(x, y)
    dfdt = dsdx * d[:, 0] + dsdy * d[:, 1] - d[:, 2]
    t = t - torch.clamp(residual / (dfdt + 1e-9), -NEWTONS_STEP_BOUND, NEWTONS_STEP_BOUND)

    hit = o + t.unsqueeze(-1) * d
    hx, hy, hz = hit[:, 0], hit[:, 1], hit[:, 2]
    with torch.no_grad():
        in_aperture = hx * hx + hy * hy <= surface.radius_mm ** 2
        valid = (
            in_aperture
            & surface.is_within_data_range(hx, hy)
            & (t >= 0)
            & (torch.abs(surface.sag(hx, hy) - hz) < NEWTONS_TOL_TIGHT)
            & torch.isfinite(hit).all(dim=1)
        )
    normal = surface.normal_vec_left(hx, hy)
    opl = torch.where(valid, t * float(n), torch.zeros_like(t))
    hit = torch.where(valid.unsqueeze(-1), hit, o)
    return hit, normal, valid, opl


def _empty_bundle(device: torch.device, dtype: torch.dtype):
    return (
        torch.zeros((0, 3), device=device, dtype=dtype),
        torch.zeros((0, 3), device=device, dtype=dtype),
        torch.zeros((0,), device=device, dtype=complex_dtype_for(dtype)),
        torch.zeros((0,), device=device, dtype=dtype),
    )


def _aperture_mask(hit: torch.Tensor, radius_mm: float) -> torch.Tensor:
    return hit[:, 0] * hit[:, 0] + hit[:, 1] * hit[:, 1] <= float(radius_mm) ** 2


def _snell(d: torch.Tensor, normal: torch.Tensor, n1: float, n2: float) -> Tuple[torch.Tensor, torch.Tensor]:
    aligned = torch.where((torch.sum(d * normal, dim=-1) < 0).unsqueeze(-1), -normal, normal)
    eta = float(n1) / float(n2)
    cosi = torch.sum(d * aligned, dim=-1)
    valid = eta * eta * (1.0 - cosi * cosi) < 1.0
    sr = torch.sqrt(torch.clamp(1.0 - eta * eta * (1.0 - cosi * cosi), min=0.0) + EPS)
    d_out = sr.unsqueeze(-1) * aligned + eta * (d - cosi.unsqueeze(-1) * aligned)
    return F.normalize(d_out, p=2, dim=-1), valid


def refract_curved(
    surface: CurvedRefractiveSurface,
    o: torch.Tensor,
    d: torch.Tensor,
    opl: torch.Tensor,
    amps: torch.Tensor,
    wavelength_mm: float,
    n1: float,
    n2: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del wavelength_mm
    hit, normal, valid_hit, opl_inc = propagate_to_surface(surface, o, d, n=n1)
    d_out, valid_snell = _snell(d, normal, n1, n2)
    valid = valid_hit & valid_snell & _aperture_mask(hit, surface.radius_mm)
    if not torch.any(valid):
        return _empty_bundle(o.device, o.dtype)
    return hit[valid], d_out[valid], amps[valid], (opl + opl_inc)[valid]


def refract_flat(
    surface: CurvedRefractiveSurface,
    o: torch.Tensor,
    d: torch.Tensor,
    opl: torch.Tensor,
    amps: torch.Tensor,
    wavelength_mm: float,
    n1: float,
    n2: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    del wavelength_mm
    hit, valid_hit, opl_inc = propagate_to_plane(o, d, surface.z0, n=n1)
    normal = torch.zeros_like(d)
    normal[:, 2] = -1.0
    d_out, valid_snell = _snell(d, normal, n1, n2)
    valid = valid_hit & valid_snell & _aperture_mask(hit, surface.radius_mm)
    if not torch.any(valid):
        return _empty_bundle(o.device, o.dtype)
    return hit[valid], d_out[valid], amps[valid], (opl + opl_inc)[valid]
