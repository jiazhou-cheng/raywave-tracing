from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
import torch

try:
    from .config import (
        HybridForwardConfig,
        RaywaveDOESurfaceConfig,
        RefractiveSurfaceConfig,
    )
    from .utils import (
        CurvedRefractiveSurface,
        FlatDoeSurface,
        complex_dtype_for,
        empty_ray_bundle,
        generate_collimated_rays,
        huygens_psf_gridded,
        phase_to_complex_field,
        propagate_to_plane,
        refract_curved,
        refract_flat,
        set_step_seed,
    )
except ImportError:
    from config import (
        HybridForwardConfig,
        RaywaveDOESurfaceConfig,
        RefractiveSurfaceConfig,
    )
    from utils import (
        CurvedRefractiveSurface,
        FlatDoeSurface,
        complex_dtype_for,
        empty_ray_bundle,
        generate_collimated_rays,
        huygens_psf_gridded,
        phase_to_complex_field,
        propagate_to_plane,
        refract_curved,
        refract_flat,
        set_step_seed,
    )


def _make_refractive_surface(
    surface: RefractiveSurfaceConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> CurvedRefractiveSurface:
    if surface.shape.lower() == "flat":
        return CurvedRefractiveSurface.flat(
            radius_mm=surface.aperture_radius_mm,
            z_offset_mm=surface.z_mm,
            device=device,
            dtype=dtype,
        )
    if surface.shape.lower() in {"spherical", "sphere", "curved"}:
        return CurvedRefractiveSurface.sphere(
            radius_mm=surface.aperture_radius_mm,
            curvature_mm_inv=surface.curvature_mm_inv,
            z_offset_mm=surface.z_mm,
            device=device,
            dtype=dtype,
        )
    raise ValueError(f"Unsupported refractive surface shape {surface.shape!r}.")


def make_doe_surface(
    surface: RaywaveDOESurfaceConfig,
    *,
    wavelength_mm: float,
    device: torch.device,
    dtype: torch.dtype,
) -> FlatDoeSurface:
    return FlatDoeSurface.plane(
        z_mm=surface.z_mm,
        pitch_mm=surface.pitch_mm,
        wavelength_mm=wavelength_mm,
        aperture_radius_mm=surface.resolved_aperture_radius_mm,
        origin_xy=surface.origin_xy_mm,
        out_primary_ray_count=surface.out_primary_ray_count,
        sampled_secondary_ray_count=surface.sampled_secondary_ray_count,
        preserve_energy=surface.preserve_energy,
        n_before=surface.n_before,
        n_after=surface.n_after,
        device=device,
        dtype=dtype,
    )


def make_doe_grid(
    config: HybridForwardConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return config.doe_grid(device, dtype)


def _trace_surface(
    surface: RaywaveDOESurfaceConfig | RefractiveSurfaceConfig,
    *,
    phase_cplx: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    amps: torch.Tensor,
    opl: torch.Tensor,
    config: HybridForwardConfig,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if isinstance(surface, RaywaveDOESurfaceConfig):
        return config.flat_doe_surface(device, dtype, doe=surface).interact(phase_cplx, o, d)

    lens = _make_refractive_surface(surface, device=device, dtype=dtype)
    refract = refract_flat if surface.shape.lower() == "flat" else refract_curved
    return refract(lens, o, d, opl, amps, config.wavelength_mm, surface.n_before, surface.n_after)


def trace_system(
    phase_cplx: torch.Tensor,
    *,
    config: HybridForwardConfig,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = phase_cplx.device
    dtype = phase_cplx.real.dtype

    X, Y, _ = config.doe_grid(device, dtype)
    o, d = generate_collimated_rays(
        X,
        Y,
        radius_mm=config.aperture_radius_mm,
        num_rays=config.num_input_rays,
        z_start_mm=config.source.z_mm,
        seed=seed,
    )
    amps = torch.ones((o.shape[0],), device=device, dtype=complex_dtype_for(dtype))
    opl = torch.zeros((o.shape[0],), device=device, dtype=dtype)

    for surface in config.surfaces:
        o, d, amps, opl = _trace_surface(
            surface,
            phase_cplx=phase_cplx,
            o=o,
            d=d,
            amps=amps,
            opl=opl,
            config=config,
            device=device,
            dtype=dtype,
        )
        if o.numel() == 0:
            return empty_ray_bundle(device, dtype)

    hit, valid, opl_inc = propagate_to_plane(o, d, config.sensor.z_mm, n=config.sensor.n_before)
    return hit[valid], d[valid], amps[valid], opl[valid] + opl_inc[valid]


def forward_from_complex_field(
    phase_cplx: torch.Tensor,
    *,
    config: HybridForwardConfig,
    seed: Optional[int] = None,
) -> torch.Tensor:
    set_step_seed(seed)
    o, d, amps, opl = trace_system(phase_cplx, config=config, seed=seed)
    return huygens_psf_gridded(
        o,
        d,
        opl,
        amps,
        config.wavelength_mm,
        config.sensor.pitch_mm,
        config.sensor.grid_size,
        oversamp=config.sensor.oversamp,
    )


def forward(
    phase_rad: torch.Tensor,
    *,
    config: HybridForwardConfig,
    seed: Optional[int] = None,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if mask is None:
        _, _, mask = config.doe_grid(phase_rad.device, phase_rad.dtype)
    return forward_from_complex_field(
        phase_to_complex_field(phase_rad, mask),
        config=config,
        seed=seed,
    )


def load_complex_phase(
    path: str,
    *,
    config: HybridForwardConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    phase = torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)
    _, _, mask = config.doe_grid(device, dtype)
    return phase_to_complex_field(phase, mask)
