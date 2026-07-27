from __future__ import annotations

import math
from typing import Optional, Tuple

import numpy as np
import torch

from .config import ConformalForwardConfig
from src_lightweight import (
    complex_dtype_for,
    empty_ray_bundle,
    generate_collimated_rays,
    huygens_psf_gridded,
    phase_to_complex_field,
    propagate_to_plane,
    set_step_seed,
)
from src_lightweight.optics.curved_doe import reflect_with_wft


def make_doe_grid(
    config: ConformalForwardConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return config.doe_grid(device, dtype)


def sensor_grid_for_huygens_output(
    config: ConformalForwardConfig,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    ks = int(config.sensor.grid_size)
    pitch = float(config.sensor.pitch_mm)
    os_size = int(math.ceil(ks * float(config.sensor.oversamp)))
    coords_os = (torch.arange(os_size, device=device, dtype=dtype) - (os_size // 2)) * pitch
    start = 0 if os_size == ks else (os_size - ks) // 2
    coords = coords_os[start : start + ks]
    return torch.meshgrid(coords, coords, indexing="xy")


def trace_system(
    complex_field: torch.Tensor,
    *,
    config: ConformalForwardConfig,
    seed: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = complex_field.device
    dtype = complex_field.real.dtype
    X, Y, _ = config.doe_grid(device, dtype)

    o, d = generate_collimated_rays(
        X,
        Y,
        radius_mm=config.aperture_radius_mm,
        num_rays=config.num_input_rays,
        z_start_mm=config.source.z_mm,
        seed=seed,
    )
    # Shoot source rays toward the curved DOE surface first.
    direction_z = 1.0 if float(config.surface.z_mm) >= float(config.source.z_mm) else -1.0
    d = torch.zeros_like(d)
    d[:, 2] = direction_z
    opl0 = torch.zeros((o.shape[0],), device=device, dtype=dtype)
    surface = config.surface.make_surface()

    o_out, d_out, amp_out, opl_out = reflect_with_wft(
        complex_field=complex_field,
        surface=surface,
        o=o,
        d=d,
        opl=opl0,
        wavelength_mm=config.wavelength_mm,
        sampled_secondary_ray_count=config.surface.sampled_secondary_ray_count,
        ray_sampling=config.surface.ray_sampling,
        gumbel_tau=config.surface.gumbel_tau,
        gumbel_straight_through=config.surface.gumbel_straight_through,
        patch_px=config.surface.patch_px,
        pad_factor=config.surface.pad_factor,
        window=config.surface.window,
        dx=config.surface.pitch_mm,
        dy=config.surface.pitch_mm,
        origin_xy=config.surface.origin_xy_mm,
        seed=seed,
        n_before=config.surface.n_before,
        n_after=config.surface.n_after,
    )
    if o_out.numel() == 0:
        return empty_ray_bundle(device, dtype)

    hit, valid, opl_inc = propagate_to_plane(
        o_out,
        d_out,
        config.sensor.z_mm,
        n=config.sensor.n_before,
    )

    return hit[valid], d_out[valid], amp_out[valid], opl_out[valid] + opl_inc[valid]


def forward_from_complex_field(
    complex_field: torch.Tensor,
    *,
    config: ConformalForwardConfig,
    seed: Optional[int] = None,
    return_grid: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Propagate a DOE transmission field to the sensor.

    ``complex_field`` may be a general complex-amplitude map, not only ``exp(i*phase)``.
    """
    set_step_seed(seed)
    o, d, amps, opl = trace_system(complex_field, config=config, seed=seed)
    field = huygens_psf_gridded(
        o,
        d,
        opl,
        amps,
        config.wavelength_mm,
        config.sensor.pitch_mm,
        config.sensor.grid_size,
        oversamp=config.sensor.oversamp,
    )
    if not return_grid:
        return field
    Xg, Yg = sensor_grid_for_huygens_output(
        config,
        device=complex_field.device,
        dtype=complex_field.real.dtype,
    )
    return Xg, Yg, field


def forward(
    phase_rad: torch.Tensor,
    *,
    config: ConformalForwardConfig,
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
    config: ConformalForwardConfig,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    phase = torch.from_numpy(np.load(path)).to(device=device, dtype=dtype)
    _, _, mask = config.doe_grid(device, dtype)
    return phase_to_complex_field(phase, mask)
