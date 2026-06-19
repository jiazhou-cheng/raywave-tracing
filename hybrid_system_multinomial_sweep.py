from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import IO, Iterable, Optional

import torch

from hybrid_system.config import (
    DEFAULT_EXPERIMENT_VARIANT,
    HybridExperimentConfig,
    experiment_config_paths_from_dir,
    load_experiment_config,
)


DEFAULT_CONFIG_DIR = "hybrid_multinomial"
DEFAULT_OUTPUT_ROOT = Path("outputs") / "hybrid_multinomial_sweep"
DEFAULT_OUT_PRIMARY_RAY_COUNTS = (128, 256)
DEFAULT_SAMPLED_SECONDARY_RAY_COUNTS = (8192, 16384, 32768)
DEFAULT_STEPS_PER_EPOCH_VALUES = (64, 128)


def parse_int_list(value: str) -> list[int]:
    values = [item.strip() for item in value.split(",")]
    parsed = [int(item) for item in values if item]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer.")
    if any(item <= 0 for item in parsed):
        raise argparse.ArgumentTypeError("Ray counts must be positive integers.")
    return parsed


def parse_gpu_list(value: str) -> list[str]:
    key = value.strip().lower()
    if key == "auto":
        if not torch.cuda.is_available():
            return []
        return [str(index) for index in range(torch.cuda.device_count())]
    values = [item.strip() for item in value.split(",")]
    parsed = [item for item in values if item]
    for item in parsed:
        if not item.isdigit():
            raise argparse.ArgumentTypeError("GPU ids must be comma-separated integers, or 'auto'.")
    return parsed


def combo_results_dir(
    output_root: Path,
    out_primary_ray_count: int,
    sampled_secondary_ray_count: int,
    steps_per_epoch: int,
) -> Path:
    return (
        output_root
        / f"out_primary_ray_count_{out_primary_ray_count}"
        / f"sampled_secondary_ray_count_{sampled_secondary_ray_count}"
        / f"steps_per_epoch_{steps_per_epoch}"
    )


def load_hybrid_multinomial_experiment(config_dir: str) -> HybridExperimentConfig:
    optical_path, optimization_path = experiment_config_paths_from_dir(config_dir)
    return load_experiment_config(optical_path, optimization_path)


def write_run_config(
    path: Path,
    *,
    experiment: HybridExperimentConfig,
    out_primary_ray_count: int,
    sampled_secondary_ray_count: int,
    steps_per_epoch: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "sweep_parameters": {
            "out_primary_ray_count": out_primary_ray_count,
            "sampled_secondary_ray_count": sampled_secondary_ray_count,
            "steps_per_epoch": steps_per_epoch,
        },
        "optical_system": asdict(experiment.optical_system),
        "optimization": asdict(experiment.optimization),
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_sweep_summary(path: Path, records: list[dict[str, object]]) -> None:
    path.write_text(json.dumps({"jobs": records}, indent=2), encoding="utf-8")


def sweep_combinations(
    out_primary_ray_counts: Iterable[int],
    sampled_secondary_ray_counts: Iterable[int],
    steps_per_epoch_values: Iterable[int],
) -> list[tuple[int, int, int]]:
    return [
        (int(out_primary), int(sampled_secondary), int(steps_per_epoch))
        for out_primary in out_primary_ray_counts
        for sampled_secondary in sampled_secondary_ray_counts
        for steps_per_epoch in steps_per_epoch_values
    ]


def make_job_command(config_path: Path, record_runtime: Optional[bool]) -> list[str]:
    cmd = [
        sys.executable,
        "-m",
        "hybrid_system.optim",
        "--config",
        str(config_path),
    ]
    if record_runtime is True:
        cmd.append("--record-runtime")
    elif record_runtime is False:
        cmd.append("--no-record-runtime")
    return cmd


def launch_job(
    *,
    record: dict[str, object],
    config_path: Path,
    run_dir: Path,
    gpu_id: Optional[str],
    record_runtime: Optional[bool],
) -> tuple[subprocess.Popen[bytes], IO[bytes]]:
    env = os.environ.copy()
    if gpu_id is not None:
        env["CUDA_VISIBLE_DEVICES"] = gpu_id
    log_path = run_dir / "job.log"
    log_file = log_path.open("ab")
    process = subprocess.Popen(
        make_job_command(config_path, record_runtime),
        cwd=Path.cwd(),
        env=env,
        stdout=log_file,
        stderr=subprocess.STDOUT,
    )
    record.update(
        {
            "status": "running",
            "pid": int(process.pid),
            "gpu_id": gpu_id,
            "log_path": str(log_path),
            "start_wall_time": time.time(),
        }
    )
    return process, log_file


def run_subprocess_sweep(
    *,
    job_records: list[dict[str, object]],
    gpu_ids: list[str],
    record_runtime: Optional[bool],
    summary_path: Path,
) -> None:
    pending = list(range(len(job_records)))
    active: dict[int, tuple[subprocess.Popen[bytes], IO[bytes], Optional[str]]] = {}
    available_devices: list[Optional[str]] = list(gpu_ids) if gpu_ids else [None]
    failures: list[dict[str, object]] = []

    try:
        while pending or active:
            while pending and available_devices:
                gpu_id = available_devices.pop(0)
                job_index = pending.pop(0)
                record = job_records[job_index]
                run_dir = Path(str(record["results_dir"]))
                config_path = run_dir / "run_config.json"
                process, log_file = launch_job(
                    record=record,
                    config_path=config_path,
                    run_dir=run_dir,
                    gpu_id=gpu_id,
                    record_runtime=record_runtime,
                )
                active[job_index] = (process, log_file, gpu_id)
                print(
                    f"launch job={job_index + 1}/{len(job_records)} "
                    f"gpu={gpu_id if gpu_id is not None else 'cpu/default'} "
                    f"pid={process.pid} results_dir={run_dir}"
                )

            write_sweep_summary(summary_path, job_records)
            time.sleep(2.0)

            finished: list[int] = []
            for job_index, (process, log_file, gpu_id) in active.items():
                return_code = process.poll()
                if return_code is None:
                    continue
                log_file.close()
                record = job_records[job_index]
                record.update(
                    {
                        "status": "completed" if return_code == 0 else "failed",
                        "return_code": int(return_code),
                        "end_wall_time": time.time(),
                    }
                )
                if return_code != 0:
                    failures.append(record)
                available_devices.append(gpu_id)
                finished.append(job_index)
                print(
                    f"finish job={job_index + 1}/{len(job_records)} "
                    f"gpu={gpu_id if gpu_id is not None else 'cpu/default'} "
                    f"return_code={return_code}"
                )

            for job_index in finished:
                del active[job_index]
    except KeyboardInterrupt:
        for job_index, (process, log_file, _gpu_id) in active.items():
            process.terminate()
            log_file.close()
            job_records[job_index].update({"status": "terminated", "end_wall_time": time.time()})
        write_sweep_summary(summary_path, job_records)
        raise

    write_sweep_summary(summary_path, job_records)
    if failures:
        failed_dirs = ", ".join(str(item["results_dir"]) for item in failures)
        raise RuntimeError(f"{len(failures)} sweep job(s) failed. Check job.log under: {failed_dirs}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Sweep out_primary_ray_count, sampled_secondary_ray_count, and steps_per_epoch "
            "for the hybrid_multinomial configuration."
        )
    )
    parser.add_argument(
        "--config-dir",
        type=str,
        default=DEFAULT_CONFIG_DIR,
        help=(
            "Config folder containing optics.yaml and optimization.yaml. "
            f"Defaults to {DEFAULT_CONFIG_DIR!r}; {DEFAULT_EXPERIMENT_VARIANT!r} is the package default."
        ),
    )
    parser.add_argument(
        "--out-primary-ray-counts",
        type=parse_int_list,
        default=list(DEFAULT_OUT_PRIMARY_RAY_COUNTS),
        help="Comma-separated values for DOE out_primary_ray_count.",
    )
    parser.add_argument(
        "--sampled-secondary-ray-counts",
        type=parse_int_list,
        default=list(DEFAULT_SAMPLED_SECONDARY_RAY_COUNTS),
        help="Comma-separated values for DOE sampled_secondary_ray_count.",
    )
    parser.add_argument(
        "--steps-per-epoch-values",
        type=parse_int_list,
        default=list(DEFAULT_STEPS_PER_EPOCH_VALUES),
        help="Comma-separated values for optimization steps_per_epoch.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=DEFAULT_OUTPUT_ROOT,
        help="Root folder for all sweep outputs. Defaults under ./outputs.",
    )
    parser.add_argument(
        "--gpus",
        type=str,
        default="auto",
        help="Comma-separated GPU ids to use, or 'auto'. Defaults to all visible CUDA devices.",
    )
    parser.add_argument(
        "--record-runtime",
        action="store_true",
        default=None,
        help="Enable runtime metrics for every subprocess job.",
    )
    parser.add_argument(
        "--no-record-runtime",
        action="store_false",
        dest="record_runtime",
        help="Disable runtime metrics for every subprocess job.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.gpus = parse_gpu_list(args.gpus)
    base_experiment = load_hybrid_multinomial_experiment(args.config_dir)
    base_config = base_experiment.optical_system
    base_optim = base_experiment.optimization

    combos = sweep_combinations(
        args.out_primary_ray_counts,
        args.sampled_secondary_ray_counts,
        args.steps_per_epoch_values,
    )
    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)

    print(f"config_dir={Path(experiment_config_paths_from_dir(args.config_dir)[0]).resolve().parent}")
    print(f"gpus={args.gpus if args.gpus else 'none; running one job at a time'}")
    print(f"output_root={output_root.resolve()}")
    print(f"fixed num_input_rays={base_config.num_input_rays}")
    print(f"fixed base_steps_per_epoch={base_optim.steps_per_epoch}")
    print(f"sweep combinations={len(combos)}")

    job_records: list[dict[str, object]] = []
    for index, (out_primary_ray_count, sampled_secondary_ray_count, steps_per_epoch) in enumerate(combos, start=1):
        run_dir = combo_results_dir(
            output_root,
            out_primary_ray_count,
            sampled_secondary_ray_count,
            steps_per_epoch,
        )
        config = base_config.with_overrides(
            out_primary_ray_count=out_primary_ray_count,
            sampled_secondary_ray_count=sampled_secondary_ray_count,
        )
        optim = replace(
            base_optim,
            results_dir=str(run_dir),
            device="auto",
            steps_per_epoch=steps_per_epoch,
        )
        experiment = replace(base_experiment, optical_system=config, optimization=optim)

        write_run_config(
            run_dir / "run_config.json",
            experiment=experiment,
            out_primary_ray_count=out_primary_ray_count,
            sampled_secondary_ray_count=sampled_secondary_ray_count,
            steps_per_epoch=steps_per_epoch,
        )
        job_records.append(
            {
                "index": index,
                "out_primary_ray_count": out_primary_ray_count,
                "sampled_secondary_ray_count": sampled_secondary_ray_count,
                "steps_per_epoch": steps_per_epoch,
                "results_dir": str(run_dir),
                "config_path": str(run_dir / "run_config.json"),
                "status": "pending",
            }
        )

    run_subprocess_sweep(
        job_records=job_records,
        gpu_ids=args.gpus,
        record_runtime=args.record_runtime,
        summary_path=output_root / "sweep_summary.json",
    )


if __name__ == "__main__":
    main()
