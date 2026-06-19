from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import torch


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _bytes_to_mib(value: Optional[int]) -> Optional[float]:
    if value is None:
        return None
    return float(value) / (1024.0 * 1024.0)


def _cuda_memory_snapshot(device: torch.device) -> dict[str, int] | None:
    if device.type != "cuda" or not torch.cuda.is_available():
        return None
    return {
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "max_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "max_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def _with_mib(memory: dict[str, int] | None) -> dict[str, int | float] | None:
    if memory is None:
        return None
    payload: dict[str, int | float] = dict(memory)
    payload.update(
        {
            "allocated_mib": _bytes_to_mib(memory["allocated_bytes"]),
            "reserved_mib": _bytes_to_mib(memory["reserved_bytes"]),
            "max_allocated_mib": _bytes_to_mib(memory["max_allocated_bytes"]),
            "max_reserved_mib": _bytes_to_mib(memory["max_reserved_bytes"]),
        }
    )
    return payload


@dataclass
class RuntimeMetricsRecorder:
    output_path: str | Path
    device: torch.device
    metadata: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def __post_init__(self) -> None:
        self.output_path = Path(self.output_path)
        self.iterations: list[dict[str, Any]] = []
        self._run_start_wall: Optional[float] = None
        self._run_start_process: Optional[float] = None
        self._iteration_start_wall: Optional[float] = None
        self._iteration_start_process: Optional[float] = None
        self._iteration_metadata: dict[str, Any] = {}
        self._start_time_utc: Optional[str] = None

    @property
    def cuda_enabled(self) -> bool:
        return self.device.type == "cuda" and torch.cuda.is_available()

    def _sync(self) -> None:
        if self.cuda_enabled:
            torch.cuda.synchronize(self.device)

    def start(self) -> None:
        if not self.enabled:
            return
        self._sync()
        if self.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._start_time_utc = _utc_now_iso()
        self._run_start_wall = time.perf_counter()
        self._run_start_process = time.process_time()

    def begin_iteration(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        self._sync()
        if self.cuda_enabled:
            torch.cuda.reset_peak_memory_stats(self.device)
        self._iteration_metadata = dict(metadata)
        self._iteration_start_wall = time.perf_counter()
        self._iteration_start_process = time.process_time()

    def end_iteration(self, **metadata: Any) -> None:
        if not self.enabled:
            return
        if self._iteration_start_wall is None or self._iteration_start_process is None:
            raise RuntimeError("RuntimeMetricsRecorder.end_iteration called before begin_iteration.")
        self._sync()
        end_wall = time.perf_counter()
        end_process = time.process_time()
        memory = _cuda_memory_snapshot(self.device)
        record = {
            **self._iteration_metadata,
            **metadata,
            "wall_time_sec": end_wall - self._iteration_start_wall,
            "process_time_sec": end_process - self._iteration_start_process,
            "gpu_memory": _with_mib(memory),
        }
        self.iterations.append(record)
        self._iteration_start_wall = None
        self._iteration_start_process = None
        self._iteration_metadata = {}

    def finish(self, **metadata: Any) -> dict[str, Any] | None:
        if not self.enabled:
            return None
        if self._run_start_wall is None or self._run_start_process is None:
            raise RuntimeError("RuntimeMetricsRecorder.finish called before start.")
        self._sync()
        end_wall = time.perf_counter()
        end_process = time.process_time()
        memory = _cuda_memory_snapshot(self.device)
        payload = {
            "metadata": {**self.metadata, **metadata},
            "start_time_utc": self._start_time_utc,
            "end_time_utc": _utc_now_iso(),
            "summary": self._summary(
                total_wall_time_sec=end_wall - self._run_start_wall,
                total_process_time_sec=end_process - self._run_start_process,
                final_gpu_memory=memory,
            ),
            "iterations": self.iterations,
        }
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    def _summary(
        self,
        *,
        total_wall_time_sec: float,
        total_process_time_sec: float,
        final_gpu_memory: dict[str, int] | None,
    ) -> dict[str, Any]:
        longest_iteration = None
        if self.iterations:
            longest_iteration = max(self.iterations, key=lambda item: item["wall_time_sec"])

        peak_allocated_bytes = None
        peak_reserved_bytes = None
        memory_records = [
            item["gpu_memory"]
            for item in self.iterations
            if item.get("gpu_memory") is not None
        ]
        if memory_records:
            peak_allocated_bytes = max(int(item["max_allocated_bytes"]) for item in memory_records)
            peak_reserved_bytes = max(int(item["max_reserved_bytes"]) for item in memory_records)

        return {
            "iteration_count": len(self.iterations),
            "total_wall_time_sec": total_wall_time_sec,
            "total_process_time_sec": total_process_time_sec,
            "longest_iteration": longest_iteration,
            "peak_gpu_allocated_bytes": peak_allocated_bytes,
            "peak_gpu_allocated_mib": _bytes_to_mib(peak_allocated_bytes),
            "peak_gpu_reserved_bytes": peak_reserved_bytes,
            "peak_gpu_reserved_mib": _bytes_to_mib(peak_reserved_bytes),
            "final_gpu_memory": _with_mib(final_gpu_memory),
        }
