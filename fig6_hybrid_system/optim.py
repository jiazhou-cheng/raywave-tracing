"""Optimize the hybrid flat-DOE + lens system (Fig. 6).

HOW TO RUN:
  python -m fig6_hybrid_system.optim
  python -m fig6_hybrid_system.optim --config-dir configs/fig6_hybrid_system
"""
import argparse
import gc
import os
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .config import (
    DEFAULT_EXPERIMENT_VARIANT,
    HybridExperimentConfig,
    HybridForwardConfig,
    experiment_config_paths_from_dir,
    load_combined_experiment_config,
    load_experiment_config,
)
from .forward import forward, make_doe_grid
from src_lightweight import (
    RuntimeMetricsRecorder,
    load_phase_npy,
    load_target_image,
    loss_from_field_sum,
    save_intensity_png,
    save_loss_plot,
    save_phase_png,
    set_step_seed,
)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {name!r}. Expected 'float32' or 'float64'.")


def make_output_dirs(results_dir: str) -> dict[str, str]:
    dirs = {
        "base": results_dir,
        "field": os.path.join(results_dir, "field"),
        "intensity": os.path.join(results_dir, "intensity"),
        "phase": os.path.join(results_dir, "phase"),
        "history": os.path.join(results_dir, "history"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def initialize_phase(
    init_phase: Optional[torch.Tensor],
    *,
    config: HybridForwardConfig,
    mask: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if init_phase is None:
        phase = torch.rand((config.doe_grid_size, config.doe_grid_size), device=device, dtype=dtype)
        phase = phase * (2.0 * torch.pi) - torch.pi
    else:
        phase = init_phase.detach().to(device=device, dtype=dtype).clone()
    if phase.shape != mask.shape:
        raise ValueError(f"Initial phase shape {tuple(phase.shape)} does not match DOE grid {tuple(mask.shape)}.")
    with torch.no_grad():
        phase.mul_(mask)
    phase.requires_grad_(True)
    return phase


def optimize_phase(
    *,
    target: torch.Tensor,
    init_phase: Optional[torch.Tensor],
    config: HybridForwardConfig,
    results_dir: str,
    num_epochs: int,
    num_steps_per_epoch: int,
    lr: float,
    base_seed: int,
    two_pass: bool = True,
    record_runtime: bool = False,
    runtime_metrics_path: Optional[str] = None,
) -> torch.Tensor:
    out = make_output_dirs(results_dir)
    device = target.device
    dtype = target.dtype
    recorder = RuntimeMetricsRecorder(
        output_path=runtime_metrics_path or os.path.join(results_dir, "runtime_metrics.json"),
        device=device,
        enabled=record_runtime,
        metadata={
            "results_dir": results_dir,
            "num_epochs": int(num_epochs),
            "num_steps_per_epoch": int(num_steps_per_epoch),
            "two_pass": bool(two_pass),
            "num_input_rays": int(config.num_input_rays),
            "out_primary_ray_count": int(config.out_primary_ray_count),
            "sampled_secondary_ray_count": int(config.sampled_secondary_ray_count),
            "ray_sampling": config.ray_sampling,
            "device": str(device),
            "dtype": str(dtype),
        },
    )
    recorder.start()
    _, _, mask = make_doe_grid(config, device=device, dtype=dtype)
    phase = initialize_phase(init_phase, config=config, mask=mask, device=device, dtype=dtype)

    optimizer = torch.optim.Adam([phase], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(num_epochs, 1), eta_min=lr)
    loss_history: list[float] = []
    lr_history: list[float] = []

    for epoch in range(num_epochs):
        recorder.begin_iteration(epoch=int(epoch))
        epoch_seed = int(base_seed + epoch * 100000)
        optimizer.zero_grad(set_to_none=True)

        if two_pass:
            field_sum = None
            with torch.no_grad():
                for step in range(num_steps_per_epoch):
                    seed = epoch_seed + step
                    set_step_seed(seed)
                    field_step = forward(phase, config=config, seed=seed, mask=mask)
                    field_sum = field_step if field_sum is None else field_sum + field_step

            field_var = field_sum.detach().requires_grad_(True)
            loss = loss_from_field_sum(field_var, target)
            field_grad = torch.autograd.grad(loss, field_var, retain_graph=False, create_graph=False)[0].detach()

            for step in range(num_steps_per_epoch):
                seed = epoch_seed + step
                set_step_seed(seed)
                field_step = forward(phase, config=config, seed=seed, mask=mask)
                phase_grad = torch.autograd.grad(
                    outputs=field_step,
                    inputs=phase,
                    grad_outputs=field_grad,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )[0]
                if phase.grad is None:
                    phase.grad = phase_grad / float(num_steps_per_epoch)
                else:
                    phase.grad.add_(phase_grad / float(num_steps_per_epoch))
        else:
            field_sum = None
            for step in range(num_steps_per_epoch):
                seed = epoch_seed + step
                set_step_seed(seed)
                field_step = forward(phase, config=config, seed=seed, mask=mask)
                field_sum = field_step if field_sum is None else field_sum + field_step
            loss = loss_from_field_sum(field_sum, target)
            loss.backward()

        if phase.grad is not None:
            phase.grad.mul_(mask.to(dtype=phase.grad.dtype))
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            phase.mul_(mask)

        loss_value = float(loss.detach().cpu())
        loss_history.append(loss_value)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))
        print(f"[epoch {epoch:04d}] loss={loss_value:.6e}, lr={lr_history[-1]:.3e}")

        with torch.no_grad():
            torch.save(field_sum.detach().cpu(), os.path.join(out["field"], f"field_sum_{epoch:04d}.pt"))
            np.save(os.path.join(out["phase"], f"phase_{epoch:04d}.npy"), phase.detach().cpu().numpy())
            save_phase_png(phase, os.path.join(out["phase"], f"phase_{epoch:04d}.png"))
            save_intensity_png(field_sum, os.path.join(out["intensity"], f"intensity_{epoch:04d}.png"))
            np.save(os.path.join(out["history"], "loss.npy"), np.asarray(loss_history, dtype=np.float32))
            np.save(os.path.join(out["history"], "lr.npy"), np.asarray(lr_history, dtype=np.float32))

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        recorder.end_iteration(
            epoch=int(epoch),
            loss=loss_value,
            lr=lr_history[-1],
        )

    save_loss_plot(loss_history, lr_history, os.path.join(out["history"], "loss.png"))
    recorder.finish(
        final_loss=loss_history[-1] if loss_history else None,
        final_lr=lr_history[-1] if lr_history else None,
    )
    return phase.detach()


def resolve_config_sources(args: argparse.Namespace) -> argparse.Namespace:
    if args.config is not None:
        return args

    if args.config_dir is not None:
        if args.optical_config is not None or args.optimization_config is not None:
            raise ValueError(
                "Use either --config-dir or (--optical-config and --optimization-config), not both."
            )
        optical_path, optimization_path = experiment_config_paths_from_dir(args.config_dir)
        args.optical_config = str(optical_path)
        args.optimization_config = str(optimization_path)
        return args

    if args.optical_config is not None or args.optimization_config is not None:
        if args.optical_config is None or args.optimization_config is None:
            raise ValueError("Provide both --optical-config and --optimization-config.")
        return args

    optical_path, optimization_path = experiment_config_paths_from_dir(DEFAULT_EXPERIMENT_VARIANT)
    args.config_dir = DEFAULT_EXPERIMENT_VARIANT
    args.optical_config = str(optical_path)
    args.optimization_config = str(optimization_path)
    return args


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Optimize the hybrid flat-DOE + lens system (Fig. 6).")
    parser.add_argument(
        "--config-dir",
        type=str,
        default=None,
        help=(
            "Folder containing optics.yaml and optimization.yaml. "
            "Accepts a variant name (e.g. fig6_hybrid_system) or an absolute path."
        ),
    )
    parser.add_argument(
        "--optical-config",
        type=str,
        default=None,
        help="YAML/JSON optics configuration (e.g. configs/fig6_hybrid_system/optics.yaml).",
    )
    parser.add_argument(
        "--optimization-config",
        type=str,
        default=None,
        help="YAML/JSON optimization configuration.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Combined YAML with optical_system and optimization sections.",
    )
    parser.add_argument(
        "--record-runtime",
        action="store_true",
        default=None,
        help="Record per-epoch runtime and GPU memory metrics to runtime_metrics.json.",
    )
    parser.add_argument(
        "--no-record-runtime",
        action="store_false",
        dest="record_runtime",
        help="Disable runtime recording even if enabled in the optimization config.",
    )
    args = parser.parse_args()
    if args.config is not None:
        if args.config_dir is not None or args.optical_config is not None or args.optimization_config is not None:
            parser.error(
                "Use either --config, --config-dir, or (--optical-config and --optimization-config), not a mix."
            )
        return args
    try:
        return resolve_config_sources(args)
    except ValueError as exc:
        parser.error(str(exc))


def main() -> None:
    args = parse_args()
    if args.config is not None:
        experiment: HybridExperimentConfig = load_combined_experiment_config(args.config)
    else:
        experiment = load_experiment_config(args.optical_config, args.optimization_config)

    config = experiment.optical_system
    optim_config = experiment.optimization
    record_runtime = (
        optim_config.record_runtime
        if args.record_runtime is None
        else bool(args.record_runtime)
    )
    device = resolve_device(optim_config.device)
    dtype = resolve_dtype(optim_config.dtype)

    target = load_target_image(
        optim_config.target,
        size=config.sensor_grid_size,
        device=device,
        dtype=dtype,
    )
    init_phase = (
        None
        if optim_config.init_phase is None
        else load_phase_npy(optim_config.init_phase, device=device, dtype=dtype)
    )

    print(f"device={device}")
    if args.config is not None:
        print(f"config={args.config} (combined)")
    elif args.config_dir is not None:
        print(f"config_dir={Path(args.optical_config).resolve().parent}")
        print(f"optical_config={Path(args.optical_config).resolve()}")
        print(f"optimization_config={Path(args.optimization_config).resolve()}")
    else:
        print(f"optical_config={Path(args.optical_config).resolve()}")
        print(f"optimization_config={Path(args.optimization_config).resolve()}")
    print(f"wavelength={config.wavelength_um:.3f} um, sensor_z={config.sensor_z_mm:.3f} mm")
    print(f"DOE={config.doe_grid_size} x {config.doe_grid_size}, pitch={config.doe_pitch_mm * 1e3:.3f} um")
    print(f"rays={config.num_input_rays} -> {config.out_primary_ray_count} x {config.sampled_secondary_ray_count}")
    print(f"record_runtime={record_runtime}")

    optimize_phase(
        target=target,
        init_phase=init_phase,
        config=config,
        results_dir=optim_config.results_dir,
        num_epochs=optim_config.epochs,
        num_steps_per_epoch=optim_config.steps_per_epoch,
        lr=optim_config.lr,
        base_seed=optim_config.seed,
        two_pass=optim_config.two_pass,
        record_runtime=record_runtime,
    )


if __name__ == "__main__":
    main()
