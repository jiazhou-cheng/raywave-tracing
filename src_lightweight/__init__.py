"""Lightweight RayWave optics and utilities for the paper demos."""

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
from .io import (
    load_phase_npy,
    load_target_image,
    save_intensity_png,
    save_loss_plot,
    save_phase_png,
)
from .loss import loss_from_field_sum
from .optics import (
    EPS,
    CurvedDoeSurface,
    CurvedRefractiveSurface,
    FlatDoeSurface,
    asm,
    doe_raywave_plane,
    huygens_psf_gridded,
    phase_to_complex_field,
    propagate_to_plane,
    propagate_to_surface,
    reflect_with_wft,
    refract_curved,
    refract_flat,
)
from .phase_retrieval import retrieve_phase
from .runtime_metrics import RuntimeMetricsRecorder

__all__ = [
    "EPS",
    "CurvedDoeSurface",
    "CurvedRefractiveSurface",
    "FlatDoeSurface",
    "RuntimeMetricsRecorder",
    "asm",
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
    "reflect_with_wft",
    "refract_curved",
    "refract_flat",
    "retrieve_phase",
    "sample_ray_origins_pixel_centers",
    "save_intensity_png",
    "save_loss_plot",
    "save_phase_png",
    "set_step_seed",
]
