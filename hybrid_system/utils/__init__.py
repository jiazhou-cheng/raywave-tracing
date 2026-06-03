"""Utilities for the hybrid RayWave demo: optics, I/O, and optimization."""

from __future__ import annotations

from .common import (
    complex_dtype_for,
    empty_ray_bundle,
    fft2c,
    generate_collimated_rays,
    ifft2c,
    nan_to_num_complex,
    sample_ray_origins_pixel_centers,
    set_step_seed,
)
from .diffractive import FlatDoeSurface, doe_raywave_plane
from .diffractive import _phase_to_complex_field as phase_to_complex_field
from .io import (
    load_phase_npy,
    load_target_image,
    save_intensity_png,
    save_loss_plot,
    save_phase_png,
)
from .loss import loss_from_field_sum
from .psf import huygens_psf_gridded
from .refractive import (
    EPS,
    CurvedRefractiveSurface,
    propagate_to_plane,
    propagate_to_surface,
    refract_curved,
    refract_flat,
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
