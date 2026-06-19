"""Compatibility exports for utilities moved to :mod:`src`."""

from __future__ import annotations

import sys
from pathlib import Path

try:
    from src import (
        EPS,
        CurvedRefractiveSurface,
        FlatDoeSurface,
        complex_dtype_for,
        doe_raywave_plane,
        empty_ray_bundle,
        fft2c,
        generate_collimated_rays,
        huygens_psf_gridded,
        ifft2c,
        load_phase_npy,
        load_target_image,
        loss_from_field_sum,
        nan_to_num_complex,
        phase_to_complex_field,
        propagate_to_plane,
        propagate_to_surface,
        refract_curved,
        refract_flat,
        sample_ray_origins_pixel_centers,
        save_intensity_png,
        save_loss_plot,
        save_phase_png,
        set_step_seed,
    )
except ModuleNotFoundError:
    repo_root = Path(__file__).resolve().parents[2]
    sys.path.insert(0, str(repo_root))
    from src import (
        EPS,
        CurvedRefractiveSurface,
        FlatDoeSurface,
        complex_dtype_for,
        doe_raywave_plane,
        empty_ray_bundle,
        fft2c,
        generate_collimated_rays,
        huygens_psf_gridded,
        ifft2c,
        load_phase_npy,
        load_target_image,
        loss_from_field_sum,
        nan_to_num_complex,
        phase_to_complex_field,
        propagate_to_plane,
        propagate_to_surface,
        refract_curved,
        refract_flat,
        sample_ray_origins_pixel_centers,
        save_intensity_png,
        save_loss_plot,
        save_phase_png,
        set_step_seed,
    )

# Backward-compatible names used by ``forward`` and older scripts.
DoeSurface = FlatDoeSurface
CurvedSurface = CurvedRefractiveSurface

__all__ = [
    "EPS",
    "CurvedRefractiveSurface",
    "CurvedSurface",
    "DoeSurface",
    "FlatDoeSurface",
    "complex_dtype_for",
    "doe_raywave_plane",
    "empty_ray_bundle",
    "fft2c",
    "generate_collimated_rays",
    "huygens_psf_gridded",
    "ifft2c",
    "load_phase_npy",
    "load_target_image",
    "loss_from_field_sum",
    "nan_to_num_complex",
    "phase_to_complex_field",
    "propagate_to_plane",
    "propagate_to_surface",
    "refract_curved",
    "refract_flat",
    "sample_ray_origins_pixel_centers",
    "save_intensity_png",
    "save_loss_plot",
    "save_phase_png",
    "set_step_seed",
]
