from __future__ import annotations

import resource
import time
from dataclasses import asdict, dataclass

import torch


@dataclass(frozen=True)
class StepMetrics:
    step: int
    loss: float
    seconds: float
    tokens_per_second_per_device: float
    model_flops_utilization: float | None
    peak_rss_bytes: int
    cuda_peak_allocated_bytes: int | None
    cuda_peak_reserved_bytes: int | None

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)


def _synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def measure_step(
    step: int,
    loss: float,
    started_at: float,
    *,
    local_tokens: int,
    logical_parameters: int,
    peak_flops_per_device: float | None = None,
) -> StepMetrics:
    _synchronize()
    seconds = time.perf_counter() - started_at
    if seconds <= 0:
        raise ValueError("step duration must be positive")
    flops = 6 * logical_parameters * local_tokens
    mfu = flops / seconds / peak_flops_per_device if peak_flops_per_device else None
    cuda = torch.cuda.is_available()
    return StepMetrics(
        step=step,
        loss=loss,
        seconds=seconds,
        tokens_per_second_per_device=local_tokens / seconds,
        model_flops_utilization=mfu,
        peak_rss_bytes=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss * 1024,
        cuda_peak_allocated_bytes=torch.cuda.max_memory_allocated() if cuda else None,
        cuda_peak_reserved_bytes=torch.cuda.max_memory_reserved() if cuda else None,
    )


def start_step() -> float:
    _synchronize()
    return time.perf_counter()
