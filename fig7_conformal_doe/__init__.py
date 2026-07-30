"""Conformal curved reflective DOE demos (Fig. 7)."""

from .config import (
    DEFAULT_EXPERIMENT_VARIANT,
    CurvedDoeSurfaceConfig,
    ConformalForwardConfig,
    ConformalExperimentConfig,
    OptimizationConfig,
    SensorConfig,
    SourceConfig,
    experiment_config_paths_from_dir,
    load_combined_experiment_config,
    load_experiment_config,
    load_optical_config,
    load_optimization_config,
    optical_config_from_dict,
    optimization_config_from_dict,
)
from .forward import forward, forward_from_complex_field, make_doe_grid, trace_system

__all__ = [
    "DEFAULT_EXPERIMENT_VARIANT",
    "CurvedDoeSurfaceConfig",
    "ConformalForwardConfig",
    "ConformalExperimentConfig",
    "OptimizationConfig",
    "SensorConfig",
    "SourceConfig",
    "experiment_config_paths_from_dir",
    "forward",
    "forward_from_complex_field",
    "load_combined_experiment_config",
    "load_experiment_config",
    "load_optical_config",
    "load_optimization_config",
    "make_doe_grid",
    "optical_config_from_dict",
    "optimization_config_from_dict",
    "trace_system",
]
