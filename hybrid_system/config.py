"""Load optical-system and optimization settings from YAML/JSON files."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

try:
    from .utils import FlatDoeSurface
except ImportError:
    from utils import FlatDoeSurface

_CONFIG_DIR = Path(__file__).resolve().parent / "configs"


@dataclass
class SourceConfig:
    z_mm: float
    num_input_rays: int


@dataclass
class RaywaveDOESurfaceConfig:
    z_mm: float
    grid_size: int
    pitch_mm: float
    out_primary_ray_count: int
    sampled_secondary_ray_count: int
    n_before: float
    n_after: float
    phase_profile: Optional[str]
    name: str
    aperture_radius_mm: Optional[float]
    origin_xy_mm: tuple[float, float]
    preserve_energy: bool
    type: str = "raywave_doe"

    @property
    def resolved_aperture_radius_mm(self) -> float:
        if self.aperture_radius_mm is not None:
            return float(self.aperture_radius_mm)
        return 0.5 * int(self.grid_size) * float(self.pitch_mm)


@dataclass
class RefractiveSurfaceConfig:
    name: str
    z_mm: float
    aperture_radius_mm: float
    n_before: float
    n_after: float
    shape: str
    curvature_mm_inv: float
    type: str = "refractive"


@dataclass
class SensorConfig:
    z_mm: float
    grid_size: int
    pitch_mm: float
    oversamp: float
    n_before: float


@dataclass
class HybridForwardConfig:
    wavelength_um: float
    source: SourceConfig
    surfaces: list[RaywaveDOESurfaceConfig | RefractiveSurfaceConfig]
    sensor: SensorConfig

    def __post_init__(self) -> None:
        self.source = _coerce_dataclass(SourceConfig, self.source)
        self.sensor = _coerce_dataclass(SensorConfig, self.sensor)
        self.surfaces = [_surface_from_mapping(s) for s in self.surfaces]

        if not self.surfaces:
            raise ValueError("HybridForwardConfig.surfaces must not be empty.")
        if not any(isinstance(s, RaywaveDOESurfaceConfig) for s in self.surfaces):
            raise ValueError("HybridForwardConfig requires one raywave_doe surface.")

    @property
    def wavelength_mm(self) -> float:
        return self.wavelength_um * 1e-3

    @property
    def doe(self) -> RaywaveDOESurfaceConfig:
        for surface in self.surfaces:
            if isinstance(surface, RaywaveDOESurfaceConfig):
                return surface
        raise ValueError("HybridForwardConfig requires one raywave_doe surface.")

    @property
    def doe_grid_size(self) -> int:
        return self.doe.grid_size

    @property
    def doe_pitch_mm(self) -> float:
        return self.doe.pitch_mm

    @property
    def doe_z_mm(self) -> float:
        return self.doe.z_mm

    @property
    def source_z_mm(self) -> float:
        return self.source.z_mm

    @property
    def aperture_radius_mm(self) -> float:
        return self.doe.resolved_aperture_radius_mm

    @property
    def sensor_z_mm(self) -> float:
        return self.sensor.z_mm

    @property
    def sensor_grid_size(self) -> int:
        return self.sensor.grid_size

    @property
    def sensor_pitch_mm(self) -> float:
        return self.sensor.pitch_mm

    @property
    def oversamp(self) -> float:
        return self.sensor.oversamp

    @property
    def num_input_rays(self) -> int:
        return self.source.num_input_rays

    @property
    def out_primary_ray_count(self) -> int:
        return self.doe.out_primary_ray_count

    @property
    def sampled_secondary_ray_count(self) -> int:
        return self.doe.sampled_secondary_ray_count

    @property
    def preserve_energy(self) -> bool:
        return self.doe.preserve_energy

    def flat_doe_surface(
        self,
        device: torch.device,
        dtype: torch.dtype,
        *,
        doe: Optional[RaywaveDOESurfaceConfig] = None,
    ) -> FlatDoeSurface:
        spec = self.doe if doe is None else doe
        return FlatDoeSurface.plane(
            z_mm=spec.z_mm,
            pitch_mm=spec.pitch_mm,
            wavelength_mm=self.wavelength_mm,
            aperture_radius_mm=spec.resolved_aperture_radius_mm,
            origin_xy=spec.origin_xy_mm,
            out_primary_ray_count=spec.out_primary_ray_count,
            sampled_secondary_ray_count=spec.sampled_secondary_ray_count,
            preserve_energy=spec.preserve_energy,
            n_before=spec.n_before,
            n_after=spec.n_after,
            device=device,
            dtype=dtype,
        )

    def doe_grid(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return self.flat_doe_surface(device, dtype).grid(self.doe_grid_size)

    def with_overrides(self, **kwargs: Any) -> HybridForwardConfig:
        source_updates: dict[str, Any] = {}
        doe_updates: dict[str, Any] = {}
        sensor_updates: dict[str, Any] = {}

        key_to_target = {
            "num_input_rays": (source_updates, "num_input_rays"),
            "sensor_grid_size": (sensor_updates, "grid_size"),
            "sensor_pitch_mm": (sensor_updates, "pitch_mm"),
            "oversamp": (sensor_updates, "oversamp"),
            "out_primary_ray_count": (doe_updates, "out_primary_ray_count"),
            "sampled_secondary_ray_count": (doe_updates, "sampled_secondary_ray_count"),
        }

        for key, value in kwargs.items():
            if value is None:
                continue
            if key not in key_to_target:
                raise ValueError(f"Unsupported config override: {key}")
            target_dict, field_name = key_to_target[key]
            target_dict[field_name] = value

        surfaces = [
            replace(surface, **doe_updates)
            if isinstance(surface, RaywaveDOESurfaceConfig)
            else surface
            for surface in self.surfaces
        ]

        return replace(
            self,
            source=replace(self.source, **source_updates),
            surfaces=surfaces,
            sensor=replace(self.sensor, **sensor_updates),
        )


@dataclass
class OptimizationConfig:
    target: str
    results_dir: str
    epochs: int
    steps_per_epoch: int
    lr: float
    seed: int
    two_pass: bool
    device: str
    dtype: str
    init_phase: Optional[str] = None


@dataclass
class HybridExperimentConfig:
    optical_system: HybridForwardConfig
    optimization: OptimizationConfig


def _coerce_dataclass(cls: type, value: Any):
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"Expected {cls.__name__} or mapping, got {type(value).__name__}.")


def _surface_from_mapping(value: Any) -> RaywaveDOESurfaceConfig | RefractiveSurfaceConfig:
    if isinstance(value, (RaywaveDOESurfaceConfig, RefractiveSurfaceConfig)):
        return value
    if not isinstance(value, dict):
        raise TypeError(f"Surface config must be a mapping, got {type(value).__name__}.")
    data = dict(value)
    surface_type = str(data.get("type", "")).lower()
    if surface_type in {"raywave_doe", "doe"}:
        return RaywaveDOESurfaceConfig(**data)
    if surface_type in {"refractive", "refractive_surface"}:
        return RefractiveSurfaceConfig(**data)
    raise ValueError(f"Unsupported surface type {surface_type!r}. Expected 'raywave_doe' or 'refractive'.")


def _load_mapping(path: Path) -> dict[str, Any]:
    text = path.read_text()
    if path.suffix.lower() in {".yaml", ".yml"}:
        try:
            import yaml
        except Exception as exc:
            raise RuntimeError("PyYAML is required to load YAML config files.") from exc
        data = yaml.safe_load(text)
    else:
        data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError(f"Config file {path} must contain a mapping.")
    return data


def _strip_legacy_optimization_keys(data: dict[str, Any]) -> dict[str, Any]:
    payload = dict(data)
    payload.pop("phase_retrieval", None)
    payload.pop("roi_size", None)
    return payload


def optical_config_from_dict(data: dict[str, Any]) -> HybridForwardConfig:
    payload = dict(data)
    if "source" in payload:
        payload["source"] = _coerce_dataclass(SourceConfig, payload["source"])
    if "sensor" in payload:
        payload["sensor"] = _coerce_dataclass(SensorConfig, payload["sensor"])
    if "surfaces" in payload:
        payload["surfaces"] = [_surface_from_mapping(s) for s in payload["surfaces"]]
    return HybridForwardConfig(**payload)


def optimization_config_from_dict(data: dict[str, Any]) -> OptimizationConfig:
    return OptimizationConfig(**_strip_legacy_optimization_keys(data))


def load_optical_config(path: str | Path) -> HybridForwardConfig:
    """Load optical-system settings from a YAML/JSON file (no built-in default path)."""
    config_path = Path(path)
    data = _load_mapping(config_path)
    if "optical_system" in data:
        data = data["optical_system"]
    return optical_config_from_dict(data)


def load_optimization_config(path: str | Path) -> OptimizationConfig:
    """Load optimization settings from a YAML/JSON file (no built-in default path)."""
    config_path = Path(path)
    data = _load_mapping(config_path)
    if "optimization" in data:
        data = data["optimization"]
    return optimization_config_from_dict(data)


def load_experiment_config(
    optical_path: str | Path,
    optimization_path: str | Path,
) -> HybridExperimentConfig:
    """Load optical and optimization settings from two separate config files."""
    return HybridExperimentConfig(
        optical_system=load_optical_config(optical_path),
        optimization=load_optimization_config(optimization_path),
    )


def load_combined_experiment_config(path: str | Path) -> HybridExperimentConfig:
    """Load a single file with top-level ``optical_system`` and ``optimization`` sections."""
    config_path = Path(path)
    data = _load_mapping(config_path)
    if "optical_system" not in data or "optimization" not in data:
        raise ValueError(
            f"{config_path} must contain top-level 'optical_system' and 'optimization' mappings."
        )
    return HybridExperimentConfig(
        optical_system=optical_config_from_dict(dict(data["optical_system"])),
        optimization=optimization_config_from_dict(dict(data["optimization"])),
    )
