"""Reusable optics primitives."""

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
from .diffractive import FlatDoeSurface, doe_raywave_plane
from .diffractive import _phase_to_complex_field as phase_to_complex_field
from .psf import huygens_psf_gridded
from .asm import FT, asm, iFT
from .refractive import (
    EPS,
    CurvedRefractiveSurface,
    propagate_to_plane,
    propagate_to_surface,
    refract_curved,
    refract_flat,
)

DoeSurface = FlatDoeSurface
CurvedSurface = CurvedRefractiveSurface

__all__ = [
    "EPS",
    "FT",
    "CurvedRefractiveSurface",
    "CurvedSurface",
    "DoeSurface",
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
    "refract_curved",
    "refract_flat",
    "sample_ray_origins_pixel_centers",
    "set_step_seed",
]
