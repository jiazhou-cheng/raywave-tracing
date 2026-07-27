"""Reusable optics primitives for RayWave demos."""

from __future__ import annotations

from ..common import (
    complex_dtype_for,
    empty_ray_bundle,
    fft2c,
    generate_collimated_rays,
    ifft2c,
    nan_to_num_complex,
    sample_ray_origins_pixel_centers,
    set_step_seed,
)
from .asm import FT, asm, iFT
from .curved_doe import CurvedDoeSurface, reflect_with_wft
from .diffractive import FlatDoeSurface, doe_raywave_plane
from .diffractive import _phase_to_complex_field as phase_to_complex_field
from .psf import huygens_psf_gridded
from .refractive import (
    EPS,
    CurvedRefractiveSurface,
    propagate_to_plane,
    propagate_to_surface,
    refract_curved,
    refract_flat,
)

__all__ = [
    "EPS",
    "FT",
    "CurvedDoeSurface",
    "CurvedRefractiveSurface",
    "FlatDoeSurface",
    "asm",
    "complex_dtype_for",
    "doe_raywave_plane",
    "empty_ray_bundle",
    "fft2c",
    "generate_collimated_rays",
    "huygens_psf_gridded",
    "ifft2c",
    "iFT",
    "nan_to_num_complex",
    "phase_to_complex_field",
    "propagate_to_plane",
    "propagate_to_surface",
    "reflect_with_wft",
    "refract_curved",
    "refract_flat",
    "sample_ray_origins_pixel_centers",
    "set_step_seed",
]
