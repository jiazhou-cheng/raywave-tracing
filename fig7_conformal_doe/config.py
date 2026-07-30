"""Config loading for the conformal (curved reflective) DOE demos."""

from __future__ import annotations

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Optional, Tuple

import torch

from src_lightweight.optics.curved_doe import CurvedDoeSurface

_CONFIG_DIR = Path(__file__).resolve().parents[1] / "configs"
DEFAULT_EXPERIMENT_VARIANT = "fig7_beamsplitter"
_OPTICS_CONFIG_NAMES = ("optics.yaml", "optical_system.yaml")
_OPTIMIZATION_CONFIG_NAME = "optimization.yaml"


def _looks_like_config_dir(path: Path) -> bool:
    return path.is_dir() and any((path / name).is_file() for name in _OPTICS_CONFIG_NAMES)


def resolve_experiment_config_dir(path: str | Path) -> Path:
    candidate = Path(path)
    under_configs = _CONFIG_DIR / candidate
    # Prefer configs/<name> so package dirs with the same name are not mistaken for configs.
    if _looks_like_config_dir(under_configs):
        return under_configs.resolve()
    if _looks_like_config_dir(candidate):
        return candidate.resolve()
    raise FileNotFoundError(f"Config folder {path!r} not found.")


def experiment_config_paths_from_dir(path: str | Path) -> tuple[Path, Path]:
    config_dir = resolve_experiment_config_dir(path)
    optics_path = next((config_dir / name for name in _OPTICS_CONFIG_NAMES if (config_dir / name).is_file()), None)
    optimization_path = config_dir / _OPTIMIZATION_CONFIG_NAME
    if optics_path is None or not optimization_path.is_file():
        raise FileNotFoundError(f"Missing optics/optimization files under {config_dir}.")
    return optics_path, optimization_path


def _normalize_ray_sampling(mode: str) -> str:
    key = str(mode).strip().lower()
    if key in {"gumbel", "gumbel_softmax", "gumbel-softmax"}:
        return "gumbel"
    if key in {"multinomial", "mc", "sample"}:
        return "multinomial"
    raise ValueError(f"Unsupported ray_sampling {mode!r}.")


@dataclass
class SourceConfig:
    z_mm: float
    num_input_rays: int


@dataclass
class SensorConfig:
    z_mm: float
    grid_size: int
    pitch_mm: float
    oversamp: float
    n_before: float


@dataclass
class CurvedDoeSurfaceConfig:
    name: str
    z_mm: float
    grid_size: int
    pitch_mm: float
    phase_profile: Optional[str]
    aperture_radius_mm: Optional[float]
    origin_xy_mm: tuple[float, float]
    sampled_secondary_ray_count: int
    n_before: float
    n_after: float
    preserve_energy: bool
    ray_sampling: str = "multinomial"
    gumbel_tau: float = 3.0
    gumbel_straight_through: bool = True
    patch_px: int = 101
    pad_factor: int = 4
    window: str = "none"
    surface_type: str = "two_bump"
    c: float = 0.0
    k: float = 0.0
    ai: tuple[float, ...] = ()
    z_offset_mm: float = 0.0
    x_min_mm: Optional[float] = None
    x_max_mm: Optional[float] = None
    y_min_mm: Optional[float] = None
    y_max_mm: Optional[float] = None
    bump1_center: tuple[float, float] = (0.6, 0.6)
    bump1_radius: float = 0.15
    bump1_height: float = 1e-4
    bump2_center: tuple[float, float] = (0.3, 0.3)
    bump2_radius: float = 0.1
    bump2_height: float = 1e-4
    edge_width: float = 0.05

    def __post_init__(self) -> None:
        self.ray_sampling = _normalize_ray_sampling(self.ray_sampling)
        self.gumbel_tau = float(self.gumbel_tau)
        self.gumbel_straight_through = bool(self.gumbel_straight_through)

    @property
    def resolved_aperture_radius_mm(self) -> float:
        if self.aperture_radius_mm is not None:
            return float(self.aperture_radius_mm)
        return 0.5 * int(self.grid_size) * float(self.pitch_mm)

    def make_surface(self) -> CurvedDoeSurface:
        r = self.resolved_aperture_radius_mm
        return CurvedDoeSurface(
            radius_mm=r,
            surface_type=self.surface_type,
            c=self.c,
            k=self.k,
            ai=tuple(self.ai),
            z_offset_mm=self.z_mm + self.z_offset_mm,
            x_min_mm=-r if self.x_min_mm is None else float(self.x_min_mm),
            x_max_mm=r if self.x_max_mm is None else float(self.x_max_mm),
            y_min_mm=-r if self.y_min_mm is None else float(self.y_min_mm),
            y_max_mm=r if self.y_max_mm is None else float(self.y_max_mm),
            bump1_center=tuple(self.bump1_center),
            bump1_radius=float(self.bump1_radius),
            bump1_height=float(self.bump1_height),
            bump2_center=tuple(self.bump2_center),
            bump2_radius=float(self.bump2_radius),
            bump2_height=float(self.bump2_height),
            edge_width=float(self.edge_width),
        )


@dataclass
class ConformalForwardConfig:
    wavelength_um: float
    source: SourceConfig
    surface: CurvedDoeSurfaceConfig
    sensor: SensorConfig

    def __post_init__(self) -> None:
        self.source = _coerce_dataclass(SourceConfig, self.source)
        self.surface = _coerce_dataclass(CurvedDoeSurfaceConfig, self.surface)
        self.sensor = _coerce_dataclass(SensorConfig, self.sensor)

    @property
    def wavelength_mm(self) -> float:
        return self.wavelength_um * 1e-3

    @property
    def doe_grid_size(self) -> int:
        return self.surface.grid_size

    @property
    def doe_pitch_mm(self) -> float:
        return self.surface.pitch_mm

    @property
    def aperture_radius_mm(self) -> float:
        return self.surface.resolved_aperture_radius_mm

    @property
    def num_input_rays(self) -> int:
        return self.source.num_input_rays

    @property
    def sampled_secondary_ray_count(self) -> int:
        return self.surface.sampled_secondary_ray_count

    @property
    def ray_sampling(self) -> str:
        return self.surface.ray_sampling

    @property
    def gumbel_tau(self) -> float:
        return self.surface.gumbel_tau

    @property
    def gumbel_straight_through(self) -> bool:
        return self.surface.gumbel_straight_through

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

    def doe_grid(self, device: torch.device, dtype: torch.dtype) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        ks = int(self.surface.grid_size)
        pitch = float(self.surface.pitch_mm)
        x0, y0 = self.surface.origin_xy_mm
        xs = x0 + (torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2) * pitch
        ys = y0 + (torch.arange(ks, device=device, dtype=dtype) - (ks - 1) / 2) * pitch
        X, Y = torch.meshgrid(xs, ys, indexing="xy")
        mask = (X * X + Y * Y) <= (self.aperture_radius_mm ** 2)
        return X, Y, mask

    def with_overrides(self, **kwargs: Any) -> "ConformalForwardConfig":
        source_updates: dict[str, Any] = {}
        surface_updates: dict[str, Any] = {}
        sensor_updates: dict[str, Any] = {}
        table = {
            "num_input_rays": (source_updates, "num_input_rays"),
            "sensor_grid_size": (sensor_updates, "grid_size"),
            "sensor_pitch_mm": (sensor_updates, "pitch_mm"),
            "oversamp": (sensor_updates, "oversamp"),
            "sampled_secondary_ray_count": (surface_updates, "sampled_secondary_ray_count"),
            "ray_sampling": (surface_updates, "ray_sampling"),
            "gumbel_tau": (surface_updates, "gumbel_tau"),
            "gumbel_straight_through": (surface_updates, "gumbel_straight_through"),
        }
        for key, value in kwargs.items():
            if value is None:
                continue
            dst, field = table[key]
            if field == "ray_sampling":
                value = _normalize_ray_sampling(value)
            dst[field] = value
        return replace(
            self,
            source=replace(self.source, **source_updates),
            surface=replace(self.surface, **surface_updates),
            sensor=replace(self.sensor, **sensor_updates),
        )


@dataclass
class OptimizationConfig:
    target: str
    results_dir: str
    epochs: int
    steps_per_epoch: int
    lr: float
    lr_min: float
    seed: int
    two_pass: bool
    device: str
    dtype: str
    init_phase: Optional[str] = None
    record_runtime: bool = False


@dataclass
class ConformalExperimentConfig:
    optical_system: ConformalForwardConfig
    optimization: OptimizationConfig


def _coerce_dataclass(cls: type, value: Any):
    if isinstance(value, cls):
        return value
    if isinstance(value, dict):
        return cls(**value)
    raise TypeError(f"Expected {cls.__name__} or mapping, got {type(value).__name__}.")


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


def optical_config_from_dict(data: dict[str, Any]) -> ConformalForwardConfig:
    payload = dict(data)
    if "source" in payload:
        payload["source"] = _coerce_dataclass(SourceConfig, payload["source"])
    if "sensor" in payload:
        payload["sensor"] = _coerce_dataclass(SensorConfig, payload["sensor"])
    if "surface" not in payload and "surfaces" in payload:
        surfaces = payload.get("surfaces") or []
        curved = None
        for item in surfaces:
            if str(item.get("type", "")).lower() in {"curved_doe", "curved"}:
                curved = item
                break
        if curved is None:
            raise ValueError("No curved_doe surface found in 'surfaces'.")
        payload["surface"] = curved
    payload["surface"] = _coerce_dataclass(CurvedDoeSurfaceConfig, payload["surface"])
    return ConformalForwardConfig(**payload)


def optimization_config_from_dict(data: dict[str, Any]) -> OptimizationConfig:
    return OptimizationConfig(**_strip_legacy_optimization_keys(data))


def load_optical_config(path: str | Path) -> ConformalForwardConfig:
    config_path = Path(path)
    data = _load_mapping(config_path)
    if "optical_system" in data:
        data = data["optical_system"]
    elif "optics" in data:
        data = data["optics"]
    return optical_config_from_dict(data)


def load_optimization_config(path: str | Path) -> OptimizationConfig:
    config_path = Path(path)
    data = _load_mapping(config_path)
    if "optimization" in data:
        data = data["optimization"]
    return optimization_config_from_dict(data)


def load_experiment_config(optical_path: str | Path, optimization_path: str | Path) -> ConformalExperimentConfig:
    return ConformalExperimentConfig(
        optical_system=load_optical_config(optical_path),
        optimization=load_optimization_config(optimization_path),
    )


def load_combined_experiment_config(path: str | Path) -> ConformalExperimentConfig:
    config_path = Path(path)
    data = _load_mapping(config_path)
    if "optical_system" not in data or "optimization" not in data:
        raise ValueError(f"{config_path} must contain 'optical_system' and 'optimization'.")
    return ConformalExperimentConfig(
        optical_system=optical_config_from_dict(dict(data["optical_system"])),
        optimization=optimization_config_from_dict(dict(data["optimization"])),
    )
