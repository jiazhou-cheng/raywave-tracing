"""Two-focus beamsplitter on a conformal reflective DOE (Fig. 7).

The DOE field is parameterized as:

    U(x, y) = exp(i * phi_1(x, y)) + exp(i * phi_2(x, y))

HOW TO RUN:
  python -m conformal_doe_fig7.optim_beamsplitter
  python -m conformal_doe_fig7.optim_beamsplitter --config-dir configs/conformal_doe_bs_fig7
"""

from __future__ import annotations

import argparse
import gc
import math
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .config import (
    experiment_config_paths_from_dir,
    load_experiment_config,
)
from .forward import forward_from_complex_field, make_doe_grid
from src_lightweight import (
    RuntimeMetricsRecorder,
    save_intensity_png,
    save_loss_plot,
    save_phase_png,
    set_step_seed,
)


# -----------------------------------------------------------------------------
# Basic utilities
# -----------------------------------------------------------------------------

def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_dtype(name: str) -> torch.dtype:
    if name == "float32":
        return torch.float32
    if name == "float64":
        return torch.float64
    raise ValueError(f"Unsupported dtype {name!r}.")


def make_output_dirs(results_dir: str) -> dict[str, str]:
    dirs = {
        "base": results_dir,
        "field": os.path.join(results_dir, "field"),
        "intensity": os.path.join(results_dir, "intensity"),
        "phase": os.path.join(results_dir, "phase"),
        "history": os.path.join(results_dir, "history"),
        "target": os.path.join(results_dir, "target"),
    }
    for path in dirs.values():
        os.makedirs(path, exist_ok=True)
    return dirs


def wrap_phase(phi: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(phi), torch.cos(phi))


def nearest_pixel_from_mm(
    Xg: torch.Tensor,
    Yg: torch.Tensor,
    x_mm: float,
    y_mm: float,
) -> tuple[int, int]:
    d2 = (Xg - x_mm).square() + (Yg - y_mm).square()
    flat_idx = torch.argmin(d2)
    iy = (flat_idx // d2.shape[1]).item()
    ix = (flat_idx % d2.shape[1]).item()
    return iy, ix


# -----------------------------------------------------------------------------
# Target construction
# -----------------------------------------------------------------------------

def make_two_point_target(
    Xg: torch.Tensor,
    Yg: torch.Tensor,
    focus_points_mm: tuple[tuple[float, float], tuple[float, float]],
    *,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, list[tuple[int, int]]]:
    """
    Create a unit-sum two-point intensity target.

    Each focus gets half of the target energy.
    """
    target = torch.zeros_like(Xg, dtype=dtype, device=Xg.device)
    pixels: list[tuple[int, int]] = []

    weight = 1.0 / float(len(focus_points_mm))
    for x_mm, y_mm in focus_points_mm:
        iy, ix = nearest_pixel_from_mm(Xg, Yg, x_mm, y_mm)
        target[iy, ix] += weight
        pixels.append((iy, ix))

    return target, pixels


# -----------------------------------------------------------------------------
# DOE field parameterization
# -----------------------------------------------------------------------------

def complex_beamsplitter_field(
    phi_1: torch.Tensor,
    phi_2: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """
    Complex-amplitude reflective DOE:

        U = exp(i phi_1) + exp(i phi_2).

    This intentionally follows the manuscript parameterization without
    normalizing by sqrt(2).
    """
    U = torch.exp(1j * phi_1) + torch.exp(1j * phi_2)
    return U * mask.to(U.dtype)


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------

def loss_from_field_sum(
    field_sum: torch.Tensor,
    target: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    intensity = field_sum.abs().to(target.dtype).square()
    intensity = intensity / intensity.sum().clamp_min(1e-12)

    loss = F.mse_loss(intensity, target, reduction="sum")
    return loss, intensity


# -----------------------------------------------------------------------------
# Saving
# -----------------------------------------------------------------------------

def save_amplitude_png(amplitude: torch.Tensor, path: str) -> None:
    import matplotlib.pyplot as plt

    image = amplitude.detach().float().cpu().numpy()

    plt.figure(figsize=(5, 5))
    plt.imshow(image, cmap="viridis")
    plt.axis("off")
    plt.tight_layout(pad=0)
    plt.savefig(path, dpi=150)
    plt.close()


def save_beamsplitter_doe(
    phi_1: torch.Tensor,
    phi_2: torch.Tensor,
    mask: torch.Tensor,
    out_dir: str,
    *,
    epoch: int,
) -> None:
    tag = f"{epoch:04d}"

    U_phys = complex_beamsplitter_field(phi_1, phi_2, mask)
    amp_phys = U_phys.abs()
    phase_phys = torch.angle(U_phys)

    np.save(
        os.path.join(out_dir, f"U_phys_{tag}.npy"),
        U_phys.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(out_dir, f"phi_1_{tag}.npy"),
        phi_1.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(out_dir, f"phi_2_{tag}.npy"),
        phi_2.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(out_dir, f"amp_phys_{tag}.npy"),
        amp_phys.detach().cpu().numpy(),
    )
    np.save(
        os.path.join(out_dir, f"phase_phys_{tag}.npy"),
        phase_phys.detach().cpu().numpy(),
    )

    save_phase_png(phi_1, os.path.join(out_dir, f"phi_1_{tag}.png"))
    save_phase_png(phi_2, os.path.join(out_dir, f"phi_2_{tag}.png"))
    save_amplitude_png(amp_phys, os.path.join(out_dir, f"amp_phys_{tag}.png"))
    save_phase_png(phase_phys, os.path.join(out_dir, f"phase_phys_{tag}.png"))


# -----------------------------------------------------------------------------
# Optimization
# -----------------------------------------------------------------------------

def optimize_beamsplitter(
    *,
    config,
    lr: float,
    lr_min: float,
    num_epochs: int,
    num_steps_per_epoch: int,
    base_seed: int,
    results_dir: str,
    two_pass: bool,
    record_runtime: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    out = make_output_dirs(results_dir)

    device = resolve_device(config.optimization.device)
    dtype = resolve_dtype(config.optimization.dtype)
    forward_cfg = config.optical_system

    # Two foci separated by 0.1 mm along x.
    focus_points_mm = (
        (-0.05, 0.0),
        (+0.05, 0.0),
    )

    recorder = RuntimeMetricsRecorder(
        output_path=os.path.join(results_dir, "runtime_metrics.json"),
        device=device,
        enabled=record_runtime,
        metadata={
            "mode": "two_focus_reflective_beamsplitter",
            "field_parameterization": "exp(i phi_1) + exp(i phi_2)",
            "focus_points_mm": focus_points_mm,
            "num_epochs": int(num_epochs),
            "num_steps_per_epoch": int(num_steps_per_epoch),
            "lr": float(lr),
        },
    )
    recorder.start()

    _, _, mask = make_doe_grid(forward_cfg, device=device, dtype=dtype)

    phi_1 = ((2.0 * math.pi * torch.rand(mask.shape, device=device, dtype=dtype,) - math.pi) * mask).detach().requires_grad_(True)
    phi_2 = ((2.0 * math.pi * torch.rand(mask.shape, device=device, dtype=dtype,) - math.pi) * mask).detach().requires_grad_(True)
    with torch.no_grad():
        U0 = complex_beamsplitter_field(phi_1, phi_2, mask)
        Xg, Yg, _ = forward_from_complex_field(
            U0,
            config=forward_cfg,
            seed=base_seed,
            return_grid=True,
        )

    target, pixels = make_two_point_target(
        Xg,
        Yg,
        focus_points_mm,
        dtype=dtype,
    )

    np.save(
        os.path.join(out["target"], "target.npy"),
        target.detach().cpu().numpy(),
    )
    save_intensity_png(
        target.to(torch.complex64),
        os.path.join(out["target"], "target.png"),
    )

    print("Two-focus target")
    for idx, ((x_req, y_req), (iy, ix)) in enumerate(
        zip(focus_points_mm, pixels),
        start=1,
    ):
        print(
            f"  focus {idx}: requested=({x_req:.6g}, {y_req:.6g}) mm, "
            f"pixel=({ix}, {iy}), "
            f"actual=({Xg[iy, ix].item():.6g}, {Yg[iy, ix].item():.6g}) mm"
        )

    print(
        "Focus separation [um]:",
        1e3 * abs(focus_points_mm[1][0] - focus_points_mm[0][0]),
    )

    optimizer = torch.optim.Adam([phi_1, phi_2], lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(num_epochs, 1),
        eta_min=float(lr_min),
    )

    loss_history: list[float] = []
    lr_history: list[float] = []

    for epoch in range(num_epochs):
        recorder.begin_iteration(epoch=int(epoch))
        epoch_seed = int(base_seed + epoch * 100000)

        optimizer.zero_grad(set_to_none=True)

        # ---------------------------------------------------------------------
        # Two-pass adjoint mode
        # ---------------------------------------------------------------------
        if two_pass:
            field_sum = None

            with torch.no_grad():
                for step in range(num_steps_per_epoch):
                    seed = epoch_seed + step
                    set_step_seed(seed)

                    U = complex_beamsplitter_field(phi_1, phi_2, mask)
                    field_step = forward_from_complex_field(
                        U,
                        config=forward_cfg,
                        seed=seed,
                    )
                    field_sum = (
                        field_step
                        if field_sum is None
                        else field_sum + field_step
                    )

            field_var = field_sum.detach().requires_grad_(True)
            loss_main, I_pred = loss_from_field_sum(field_var, target)
            field_grad = torch.autograd.grad(
                loss_main,
                field_var,
                retain_graph=False,
                create_graph=False,
            )[0].detach()

            for step in range(num_steps_per_epoch):
                seed = epoch_seed + step
                set_step_seed(seed)

                U = complex_beamsplitter_field(phi_1, phi_2, mask)
                field_step = forward_from_complex_field(
                    U,
                    config=forward_cfg,
                    seed=seed,
                )

                grad_phi_1, grad_phi_2 = torch.autograd.grad(
                    outputs=field_step,
                    inputs=(phi_1, phi_2),
                    grad_outputs=field_grad,
                    retain_graph=False,
                    create_graph=False,
                    allow_unused=False,
                )

                scale = 1.0 / float(num_steps_per_epoch)

                if phi_1.grad is None:
                    phi_1.grad = grad_phi_1 * scale
                else:
                    phi_1.grad.add_(grad_phi_1 * scale)

                if phi_2.grad is None:
                    phi_2.grad = grad_phi_2 * scale
                else:
                    phi_2.grad.add_(grad_phi_2 * scale)

            loss_value = float(loss_main.detach().cpu())

        # ---------------------------------------------------------------------
        # Standard direct-autograd mode
        # ---------------------------------------------------------------------
        else:
            field_sum = None

            for step in range(num_steps_per_epoch):
                seed = epoch_seed + step
                set_step_seed(seed)

                U = complex_beamsplitter_field(phi_1, phi_2, mask)
                field_step = forward_from_complex_field(
                    U,
                    config=forward_cfg,
                    seed=seed,
                )

                field_sum = (
                    field_step
                    if field_sum is None
                    else field_sum + field_step
                )

            loss_main, I_pred = loss_from_field_sum(field_sum, target)
            loss_main.backward()
            loss_value = float(loss_main.detach().cpu())

        # Mask gradients outside the physical DOE.
        if phi_1.grad is not None:
            phi_1.grad.mul_(mask.to(dtype=phi_1.grad.dtype))

        if phi_2.grad is not None:
            phi_2.grad.mul_(mask.to(dtype=phi_2.grad.dtype))

        optimizer.step()
        scheduler.step()

        with torch.no_grad():
            phi_1.mul_(mask)
            phi_2.mul_(mask)

            phi_1.copy_(wrap_phase(phi_1))
            phi_2.copy_(wrap_phase(phi_2))

        loss_history.append(loss_value)
        lr_history.append(float(optimizer.param_groups[0]["lr"]))

        print(
            f"[epoch {epoch:04d}] "
            f"loss={loss_value:.6e}, "
            f"mse={float(loss_main.detach().cpu()):.6e}, "
            f"lr={lr_history[-1]:.3e}"
        )

        with torch.no_grad():
            torch.save(
                field_sum.detach().cpu(),
                os.path.join(out["field"], f"field_sum_{epoch:04d}.pt"),
            )
            save_intensity_png(
                field_sum,
                os.path.join(out["intensity"], f"intensity_{epoch:04d}.png"),
            )
            save_beamsplitter_doe(
                phi_1,
                phi_2,
                mask,
                out["phase"],
                epoch=epoch,
            )

            np.save(
                os.path.join(out["history"], "loss.npy"),
                np.asarray(loss_history, dtype=np.float32),
            )
            np.save(
                os.path.join(out["history"], "lr.npy"),
                np.asarray(lr_history, dtype=np.float32),
            )
            np.save(
                os.path.join(out["intensity"], f"pred_norm_{epoch:04d}.npy"),
                I_pred.detach().cpu().numpy(),
            )

        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        recorder.end_iteration(
            epoch=int(epoch),
            loss=loss_value,
            lr=lr_history[-1],
        )

    save_loss_plot(
        loss_history,
        lr_history,
        os.path.join(out["history"], "loss.png"),
    )

    recorder.finish(
        final_loss=loss_history[-1] if loss_history else None,
        final_lr=lr_history[-1] if lr_history else None,
    )

    return phi_1.detach(), phi_2.detach()


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize a conformal reflective DOE for two focused spots (Fig. 7)."
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default="configs/conformal_doe_bs_fig7",
    )
    parser.add_argument("--optical-config", type=str, default=None)
    parser.add_argument("--optimization-config", type=str, default=None)
    parser.add_argument("--record-runtime", action="store_true")
    parser.add_argument("--no-record-runtime", action="store_true")

    args = parser.parse_args()

    if args.record_runtime and args.no_record_runtime:
        raise ValueError(
            "Cannot use both --record-runtime and --no-record-runtime."
        )

    return args


def main() -> None:
    args = parse_args()

    if args.optical_config is None and args.optimization_config is None:
        optical_path, optimization_path = experiment_config_paths_from_dir(
            args.config_dir
        )
    elif (
        args.optical_config is not None
        and args.optimization_config is not None
    ):
        optical_path = Path(args.optical_config)
        optimization_path = Path(args.optimization_config)
    else:
        raise ValueError(
            "Provide both --optical-config and --optimization-config, "
            "or use --config-dir only."
        )

    experiment = load_experiment_config(optical_path, optimization_path)
    cfg = experiment.optical_system
    opt = experiment.optimization

    record_runtime = bool(opt.record_runtime)
    if args.record_runtime:
        record_runtime = True
    if args.no_record_runtime:
        record_runtime = False

    print(f"optical_config={Path(optical_path).resolve()}")
    print(f"optimization_config={Path(optimization_path).resolve()}")
    print(
        f"wavelength={cfg.wavelength_um:.3f} um, "
        f"sensor_z={cfg.sensor_z_mm:.4f} mm"
    )
    print(
        f"DOE={cfg.doe_grid_size} x {cfg.doe_grid_size}, "
        f"pitch={cfg.doe_pitch_mm * 1e3:.3f} um"
    )
    print(
        f"rays={cfg.num_input_rays} x "
        f"{cfg.sampled_secondary_ray_count}"
    )

    optimize_beamsplitter(
        config=experiment,
        lr=float(opt.lr),
        lr_min=float(opt.lr_min),
        num_epochs=int(opt.epochs),
        num_steps_per_epoch=int(opt.steps_per_epoch),
        base_seed=int(opt.seed),
        results_dir=str(opt.results_dir),
        two_pass=bool(opt.two_pass),
        record_runtime=record_runtime,
    )


if __name__ == "__main__":
    main()