"""Curved reflective DOE geometry and local windowed Fourier transform (WFT) reflection."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from ..common import complex_dtype_for, fft2c, nan_to_num_complex
from .refractive import EPS

NEWTONS_MAXITER = 10
NEWTONS_TOL = 50e-6
NEWTONS_STEP_BOUND = 5.0


@dataclass
class CurvedDoeSurface:
    """Sagged reflective DOE surface (aspheric or two-bump conformal shape)."""
    radius_mm: float
    surface_type: str = "two_bump"
    c: float = 0.0
    k: float = 0.0
    ai: tuple[float, ...] = ()
    z_offset_mm: float = 0.0
    x_min_mm: float = -1.0
    x_max_mm: float = 1.0
    y_min_mm: float = -1.0
    y_max_mm: float = 1.0
    bump1_center: tuple[float, float] = (0.6, 0.6)
    bump1_radius: float = 0.15
    bump1_height: float = 1e-4
    bump2_center: tuple[float, float] = (0.3, 0.3)
    bump2_radius: float = 0.1
    bump2_height: float = 1e-4
    edge_width: float = 0.05

    def _normalize_coords(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.surface_type != "two_bump":
            return x, y
        x_norm = (x - float(self.x_min_mm)) / (float(self.x_max_mm) - float(self.x_min_mm) + EPS)
        y_norm = (y - float(self.y_min_mm)) / (float(self.y_max_mm) - float(self.y_min_mm) + EPS)
        return x_norm, y_norm

    def _edge_mask(self, x_norm: torch.Tensor, y_norm: torch.Tensor) -> torch.Tensor:
        dist = torch.minimum(
            torch.minimum(x_norm, 1.0 - x_norm),
            torch.minimum(y_norm, 1.0 - y_norm),
        )
        return torch.clamp(dist / (float(self.edge_width) + EPS), 0.0, 1.0).square()

    def _edge_mask_with_grad(
        self,
        x_norm: torch.Tensor,
        y_norm: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        left = x_norm
        right = 1.0 - x_norm
        bottom = y_norm
        top = 1.0 - y_norm
        dist = torch.minimum(torch.minimum(left, right), torch.minimum(bottom, top))

        edge_width = float(self.edge_width) + EPS
        ramp = torch.clamp(dist / edge_width, 0.0, 1.0)
        edge = ramp.square()
        active = ((dist > 0.0) & (dist < edge_width)).to(x_norm.dtype)
        dedist = 2.0 * ramp * active / edge_width

        is_left = (left <= right) & (left <= bottom) & (left <= top)
        is_right = (~is_left) & (right <= bottom) & (right <= top)
        is_bottom = (~is_left) & (~is_right) & (bottom <= top)
        is_top = (~is_left) & (~is_right) & (~is_bottom)

        ddist_dx = torch.zeros_like(x_norm)
        ddist_dy = torch.zeros_like(y_norm)
        ddist_dx = torch.where(is_left, torch.ones_like(ddist_dx), ddist_dx)
        ddist_dx = torch.where(is_right, -torch.ones_like(ddist_dx), ddist_dx)
        ddist_dy = torch.where(is_bottom, torch.ones_like(ddist_dy), ddist_dy)
        ddist_dy = torch.where(is_top, -torch.ones_like(ddist_dy), ddist_dy)
        return edge, dedist * ddist_dx, dedist * ddist_dy

    def sag(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        z0 = float(self.z_offset_mm)
        if self.surface_type == "aspheric":
            r2 = x * x + y * y
            c = float(self.c)
            k = float(self.k)
            base = torch.zeros_like(r2)
            if abs(c) > 0.0:
                under = torch.clamp(1.0 - (1.0 + k) * r2 * c * c, min=1e-12)
                base = r2 * c / (1.0 + torch.sqrt(under))
            for idx, coeff in enumerate(self.ai):
                base = base + float(coeff) * (r2 ** (idx + 1))
            return base + z0

        if self.surface_type != "two_bump":
            raise ValueError(f"Unknown surface_type: {self.surface_type}")

        x_norm, y_norm = self._normalize_coords(x, y)
        x1, y1 = self.bump1_center
        x2, y2 = self.bump2_center
        r1_sq = ((x_norm - x1) ** 2 + (y_norm - y1) ** 2) / (float(self.bump1_radius) ** 2 + EPS)
        r2_sq = ((x_norm - x2) ** 2 + (y_norm - y2) ** 2) / (float(self.bump2_radius) ** 2 + EPS)
        bump = float(self.bump1_height) * torch.exp(-0.5 * r1_sq)
        bump = bump + float(self.bump2_height) * torch.exp(-0.5 * r2_sq)
        return bump * self._edge_mask(x_norm, y_norm) + z0

    def grad(self, x: torch.Tensor, y: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if self.surface_type == "aspheric":
            r2 = x * x + y * y
            c = float(self.c)
            k = float(self.k)
            dsdr2 = torch.zeros_like(r2)
            if abs(c) > 0.0:
                under = torch.clamp(1.0 - (1.0 + k) * r2 * c * c, min=1e-12)
                sf = torch.sqrt(under)
                dsdr2 = (1.0 + sf + (1.0 + k) * r2 * c * c / (2.0 * sf)) * c / (1.0 + sf) ** 2
            for idx, coeff in enumerate(self.ai):
                power = idx + 1
                dsdr2 = dsdr2 + power * float(coeff) * (r2 ** (power - 1))
            return dsdr2 * 2.0 * x, dsdr2 * 2.0 * y

        if self.surface_type != "two_bump":
            raise ValueError(f"Unknown surface_type: {self.surface_type}")

        x_norm, y_norm = self._normalize_coords(x, y)
        sx = 1.0 / (float(self.x_max_mm) - float(self.x_min_mm) + EPS)
        sy = 1.0 / (float(self.y_max_mm) - float(self.y_min_mm) + EPS)

        x1, y1 = self.bump1_center
        dx1 = x_norm - x1
        dy1 = y_norm - y1
        r1_sq = (dx1.square() + dy1.square()) / (float(self.bump1_radius) ** 2 + EPS)
        b1 = float(self.bump1_height) * torch.exp(-0.5 * r1_sq)

        x2, y2 = self.bump2_center
        dx2 = x_norm - x2
        dy2 = y_norm - y2
        r2_sq = (dx2.square() + dy2.square()) / (float(self.bump2_radius) ** 2 + EPS)
        b2 = float(self.bump2_height) * torch.exp(-0.5 * r2_sq)

        edge, dedge_dx_n, dedge_dy_n = self._edge_mask_with_grad(x_norm, y_norm)
        bump = b1 + b2
        dsdx_n = -(b1 * dx1) / (float(self.bump1_radius) ** 2 + EPS)
        dsdx_n = dsdx_n - (b2 * dx2) / (float(self.bump2_radius) ** 2 + EPS)
        dsdy_n = -(b1 * dy1) / (float(self.bump1_radius) ** 2 + EPS)
        dsdy_n = dsdy_n - (b2 * dy2) / (float(self.bump2_radius) ** 2 + EPS)

        dsdx_n = dsdx_n * edge + bump * dedge_dx_n
        dsdy_n = dsdy_n * edge + bump * dedge_dy_n
        return dsdx_n * sx, dsdy_n * sy

    def normal(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        dsdx, dsdy = self.grad(x, y)
        n = torch.stack([-dsdx, -dsdy, torch.ones_like(dsdx)], dim=-1)
        return F.normalize(n, dim=-1)


def _empty_ray_bundle(device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    cplx_dtype = complex_dtype_for(dtype)
    return (
        torch.zeros((0, 3), device=device, dtype=dtype),
        torch.zeros((0, 3), device=device, dtype=dtype),
        torch.zeros((0,), device=device, dtype=cplx_dtype),
        torch.zeros((0,), device=device, dtype=dtype),
    )


def _grid_sample_fp32(img: torch.Tensor, grid: torch.Tensor, *, align_corners: bool = True) -> torch.Tensor:
    out_dtype = img.dtype
    out = F.grid_sample(
        img.to(torch.float32),
        grid.to(torch.float32),
        mode="bilinear",
        align_corners=align_corners,
        padding_mode="zeros",
    )
    return out.to(out_dtype)


def _xy_to_norm_centered(
    xy: torch.Tensor,
    H: int,
    W: int,
    dx: float,
    dy: float,
    origin_xy: Tuple[float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    x0, y0 = origin_xy
    x = xy[..., 0]
    y = xy[..., 1]
    i = (x - float(x0)) / float(dx) + (W - 1) * 0.5
    j = (y - float(y0)) / float(dy) + (H - 1) * 0.5
    inorm = (i / (W - 1) - 0.5) * 2.0
    jnorm = (j / (H - 1) - 0.5) * 2.0
    return inorm, jnorm


def _build_tangent_uv(n_hat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = F.normalize(n_hat, dim=-1)
    world_x = torch.tensor([1.0, 0.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n)
    t = world_x - torch.sum(world_x * n, dim=-1, keepdim=True) * n
    t_norm = torch.norm(t, dim=-1, keepdim=True)
    fallback = torch.tensor([0.0, 1.0, 0.0], device=n.device, dtype=n.dtype).expand_as(n)
    t = torch.where(t_norm < 1e-8, fallback, t)
    u = F.normalize(t, dim=-1)
    v = F.normalize(torch.cross(n, u, dim=-1), dim=-1)
    return u, v


def _extract_local_patches(
    complex_field: torch.Tensor,
    o_xy: torch.Tensor,
    u_hat: torch.Tensor,
    v_hat: torch.Tensor,
    patch_px: int,
    dx: float,
    dy: float,
    origin_xy: Tuple[float, float],
    pad_factor: int,
    window: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    H, W = complex_field.shape
    N = o_xy.shape[0]
    device = complex_field.device
    real_dtype = complex_field.real.dtype

    # Keep the local WFT sampling pitch locked to the global DOE pixel pitch.
    # Stretching by the tangent-basis xy projection can over-broaden patches on
    # steeper slopes and wash out wrapped metalens phase gradients.
    du = torch.full((N,), float(dx), device=device, dtype=real_dtype)
    dv = torch.full((N,), float(dy), device=device, dtype=real_dtype)

    half = (int(patch_px) - 1) * 0.5
    uu = torch.linspace(-half, half, int(patch_px), device=device, dtype=real_dtype)
    vv = torch.linspace(-half, half, int(patch_px), device=device, dtype=real_dtype)
    U, V = torch.meshgrid(uu, vv, indexing="xy")
    U = U[None] * du[:, None, None]
    V = V[None] * dv[:, None, None]

    u_xy = u_hat[:, :2].view(N, 1, 1, 2)
    v_xy = v_hat[:, :2].view(N, 1, 1, 2)
    XY = o_xy.view(N, 1, 1, 2) + U.unsqueeze(-1) * u_xy + V.unsqueeze(-1) * v_xy
    inorm, jnorm = _xy_to_norm_centered(XY, H, W, dx, dy, origin_xy)
    grid = torch.stack([inorm, jnorm], dim=-1)

    img_r = complex_field.real.view(1, 1, H, W).expand(N, -1, -1, -1)
    img_i = complex_field.imag.view(1, 1, H, W).expand(N, -1, -1, -1)
    patch_r = _grid_sample_fp32(img_r, grid, align_corners=True).squeeze(1)
    patch_i = _grid_sample_fp32(img_i, grid, align_corners=True).squeeze(1)
    patch = torch.complex(patch_r, patch_i)

    if window != "none":
        t = torch.linspace(0, 1, int(patch_px), device=device, dtype=real_dtype)
        if window == "hann":
            w = 0.5 - 0.5 * torch.cos(2.0 * math.pi * t)
        elif window == "hamming":
            w = 0.54 - 0.46 * torch.cos(2.0 * math.pi * t)
        elif window == "blackman":
            w = 0.42 - 0.5 * torch.cos(2.0 * math.pi * t) + 0.08 * torch.cos(4.0 * math.pi * t)
        else:
            w = torch.ones_like(t)
        patch = patch * (w[:, None] * w[None, :]).unsqueeze(0).to(patch.dtype)

    pad_size = int(pad_factor * max(0, int(patch_px) - 1))
    if pad_size % 2 == 0:
        pad_size += 1
    tpad = pad_size - int(patch_px)
    pad_before = tpad // 2
    pad_after = tpad - pad_before
    patches_pad = F.pad(patch, (pad_before, pad_after, pad_before, pad_after))
    return patches_pad, du, dv


def _propagate_to_surface(
    surface: CurvedDoeSurface,
    o: torch.Tensor,
    d: torch.Tensor,
    opl: torch.Tensor,
    valid: torch.Tensor,
    *,
    max_iter: int = NEWTONS_MAXITER,
    tol: float = NEWTONS_TOL,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = o.dtype
    t0 = ((float(surface.z_offset_mm) - o[:, 2]) / (d[:, 2] + 1e-9)).to(dtype)
    with torch.no_grad():
        t = t0.detach().clone()
        for _ in range(max(1, int(max_iter) - 1)):
            p = o + t.unsqueeze(-1) * d
            x, y, z = p[:, 0], p[:, 1], p[:, 2]
            f = z - surface.sag(x, y)
            dsdx, dsdy = surface.grad(x, y)
            dfdt = d[:, 2] - (dsdx * d[:, 0] + dsdy * d[:, 1])
            delta = torch.clamp(f / (dfdt + 1e-9), -NEWTONS_STEP_BOUND, NEWTONS_STEP_BOUND)
            t = t - delta
            if torch.max(torch.abs(delta)) < float(tol):
                break
        t_correction = t - t0.detach()

    t = t0 + t_correction
    p = o + t.unsqueeze(-1) * d
    x, y, z = p[:, 0], p[:, 1], p[:, 2]
    f = z - surface.sag(x, y)
    dsdx, dsdy = surface.grad(x, y)
    dfdt = d[:, 2] - (dsdx * d[:, 0] + dsdy * d[:, 1])
    delta = torch.clamp(f / (dfdt + 1e-9), -NEWTONS_STEP_BOUND, NEWTONS_STEP_BOUND)
    t = t - delta

    o_hit = o + t.unsqueeze(-1) * d
    xh, yh = o_hit[:, 0], o_hit[:, 1]
    valid_hit = valid & torch.isfinite(t) & (t >= 0) & ((xh * xh + yh * yh) <= float(surface.radius_mm) ** 2)
    opl_hit = opl + torch.norm(o_hit - o, dim=-1)
    n_hat = surface.normal(xh, yh)
    return o_hit, n_hat, opl_hit, valid_hit


def _sample_gumbel_softmax(
    logits: torch.Tensor,
    repeats: int,
    tau: float,
    straight_through: bool,
    seed: Optional[int],
) -> torch.Tensor:
    N, HW = logits.shape
    S = N * int(repeats)
    device = logits.device
    dtype = logits.dtype
    if seed is None:
        noise = torch.rand((S, HW), device=device, dtype=dtype).clamp_min(1e-9)
    else:
        with torch.random.fork_rng(devices=[device] if device.type == "cuda" else []):
            torch.manual_seed(int(seed))
            if device.type == "cuda":
                torch.cuda.manual_seed_all(int(seed))
            noise = torch.rand((S, HW), device=device, dtype=dtype).clamp_min(1e-9)
    gumbel = -torch.log(-torch.log(noise))
    y_soft = F.softmax((logits.repeat_interleave(repeats, dim=0) + gumbel) / max(float(tau), 1e-6), dim=-1)
    if not straight_through:
        return y_soft
    idx = y_soft.argmax(dim=-1)
    y_hard = torch.zeros_like(y_soft).scatter_(1, idx.unsqueeze(1), 1.0)
    return y_hard + (y_soft - y_soft.detach())


def reflect_with_wft(
    *,
    complex_field: torch.Tensor,
    surface: CurvedDoeSurface,
    o: torch.Tensor,
    d: torch.Tensor,
    opl: torch.Tensor,
    wavelength_mm: float,
    sampled_secondary_ray_count: int,
    ray_sampling: str = "multinomial",
    gumbel_tau: float = 3.0,
    gumbel_straight_through: bool = True,
    patch_px: int = 101,
    pad_factor: int = 4,
    window: str = "none",
    dx: float = 0.01,
    dy: float = 0.01,
    origin_xy: Tuple[float, float] = (0.0, 0.0),
    seed: Optional[int] = None,
    n_before: float = 1.0,
    n_after: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Hit a curved DOE, form local WFT patches, and sample reflected secondary rays."""
    device = complex_field.device
    real_dtype = complex_field.real.dtype
    cplx_dtype = complex_dtype_for(real_dtype)
    valid0 = torch.ones((o.shape[0],), device=device, dtype=torch.bool)
    o_hit, n_hat, opl_hit, valid_hit = _propagate_to_surface(surface, o, d, opl, valid0)
    if not torch.any(valid_hit):
        return _empty_ray_bundle(device, real_dtype)

    o_hit = o_hit[valid_hit]
    d_in = d[valid_hit]
    opl_hit = opl_hit[valid_hit]
    n_hat = n_hat[valid_hit]
    N = o_hit.shape[0]
    spr = max(1, int(sampled_secondary_ray_count))

    u_hat, v_hat = _build_tangent_uv(n_hat)
    patches_pad, du, dv = _extract_local_patches(
        complex_field=complex_field,
        o_xy=o_hit[:, :2],
        u_hat=u_hat,
        v_hat=v_hat,
        patch_px=int(patch_px),
        dx=float(dx),
        dy=float(dy),
        origin_xy=origin_xy,
        pad_factor=int(pad_factor),
        window=window,
    )

    _, ppad, _ = patches_pad.shape
    HW = ppad * ppad
    unit_fu = torch.fft.fftshift(torch.fft.fftfreq(ppad, d=1.0)).to(device=device, dtype=real_dtype)
    unit_fv = torch.fft.fftshift(torch.fft.fftfreq(ppad, d=1.0)).to(device=device, dtype=real_dtype)
    Fu = unit_fu.view(1, 1, ppad) / du.view(N, 1, 1)
    Fv = unit_fv.view(1, ppad, 1) / dv.view(N, 1, 1)

    k0 = 2.0 * math.pi / max(float(wavelength_mm), EPS)
    ku0 = (float(n_before) * k0 * torch.sum(d_in * u_hat, dim=-1)).view(N, 1, 1)
    kv0 = (float(n_before) * k0 * torch.sum(d_in * v_hat, dim=-1)).view(N, 1, 1)
    k_exit = float(n_after) * k0
    Kx = (ku0 + 2.0 * math.pi * Fu).expand(N, ppad, ppad)
    Ky = (kv0 + 2.0 * math.pi * Fv).expand(N, ppad, ppad)

    spec = fft2c(patches_pad)
    propagating = (Kx * Kx + Ky * Ky) < (k_exit * k_exit)
    mag = torch.nan_to_num(spec.abs() * propagating.to(real_dtype), nan=0.0, posinf=0.0, neginf=0.0)
    mag_sum = mag.sum(dim=(1, 2), keepdim=True)
    dead = mag_sum <= 0
    if dead.any():
        cy = ppad // 2
        cx = ppad // 2
        mag = mag.clone()
        mag[dead.expand_as(mag)] = 0.0
        mag[dead.squeeze(-1).squeeze(-1), cy, cx] = 1.0
        mag_sum = mag.sum(dim=(1, 2), keepdim=True)
    pdf = mag / (mag_sum + EPS)

    pdf_flat = pdf.reshape(N, HW)
    spec_flat = spec.reshape(N, HW).to(cplx_dtype)
    kx_flat = Kx.reshape(N, HW)
    ky_flat = Ky.reshape(N, HW)

    mode = str(ray_sampling).strip().lower()
    if mode in {"gumbel", "gumbel_softmax", "gumbel-softmax"}:
        logits = torch.log(pdf_flat + 1e-30)
        y = _sample_gumbel_softmax(logits, spr, gumbel_tau, bool(gumbel_straight_through), seed)
        spec_rep = spec_flat.repeat_interleave(spr, dim=0)
        pdf_rep = pdf_flat.repeat_interleave(spr, dim=0)
        kx_rep = kx_flat.repeat_interleave(spr, dim=0)
        ky_rep = ky_flat.repeat_interleave(spr, dim=0)
        kx_s = torch.sum(y * kx_rep, dim=-1)
        ky_s = torch.sum(y * ky_rep, dim=-1)
        spec_s = torch.sum(y.to(cplx_dtype) * spec_rep, dim=-1)
        pdf_s = torch.sum(y * pdf_rep, dim=-1).clamp_min(1e-20)
    elif mode in {"multinomial", "mc", "sample"}:
        pdf_flat = pdf_flat.detach()
        if seed is None:
            idx = torch.multinomial(pdf_flat, spr, replacement=True)
        else:
            generator = torch.Generator(device=device)
            generator.manual_seed(int(seed))
            idx = torch.multinomial(pdf_flat, spr, replacement=True, generator=generator)
        kx_s = torch.gather(kx_flat, 1, idx).reshape(-1)
        ky_s = torch.gather(ky_flat, 1, idx).reshape(-1)
        spec_s = torch.gather(spec_flat, 1, idx).reshape(-1).to(cplx_dtype)
        pdf_s = torch.gather(pdf_flat, 1, idx).reshape(-1).clamp_min(1e-20)
    else:
        raise ValueError(f"Unsupported ray_sampling {ray_sampling!r}.")

    du_s = kx_s / (k_exit + EPS)
    dv_s = ky_s / (k_exit + EPS)
    dw_s = torch.sqrt((1.0 - du_s * du_s - dv_s * dv_s).clamp_min(0.0))

    u_rep = u_hat.repeat_interleave(spr, dim=0)
    v_rep = v_hat.repeat_interleave(spr, dim=0)
    n_rep = n_hat.repeat_interleave(spr, dim=0)
    sign_ref = -torch.sign(torch.sum(d_in * n_hat, dim=-1)).repeat_interleave(spr, dim=0)
    sign_ref = torch.where(sign_ref == 0, torch.ones_like(sign_ref), sign_ref)
    d_out = du_s.unsqueeze(-1) * u_rep + dv_s.unsqueeze(-1) * v_rep + (sign_ref * dw_s).unsqueeze(-1) * n_rep
    d_out = F.normalize(d_out, dim=-1)

    cell_area = (du * dv).repeat_interleave(spr, dim=0)
    amps = spec_s * (cell_area.to(cplx_dtype) / pdf_s.to(cplx_dtype)) / float(spr)
    amps = nan_to_num_complex(amps)
    o_out = o_hit.repeat_interleave(spr, dim=0)
    opl_out = opl_hit.repeat_interleave(spr, dim=0)
    return o_out, d_out, amps, opl_out
