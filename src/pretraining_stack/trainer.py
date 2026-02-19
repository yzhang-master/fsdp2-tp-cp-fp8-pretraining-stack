from __future__ import annotations

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor

from .data import context_parallel_batch
from .model import causal_loss
from .parallel import ParallelContext


def synchronize_cp_gradients(
    model: torch.nn.Module, parallel: ParallelContext
) -> None:
    if parallel.config.cp_size == 1:
        return
    for parameter in model.parameters():
        if parameter.grad is None:
            continue
        gradient = parameter.grad
        if isinstance(gradient, DTensor):
            gradient = gradient.to_local()
        dist.all_reduce(gradient, group=parallel.cp_group)


def training_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    packed_rows: torch.Tensor,
    parallel: ParallelContext,
) -> float:
    inputs, labels, target_count = context_parallel_batch(
        packed_rows, parallel.cp_rank, parallel.config.cp_size
    )
    optimizer.zero_grad(set_to_none=True)
    loss = causal_loss(model(inputs), labels, target_count)
    loss.backward()
    synchronize_cp_gradients(model, parallel)
    optimizer.step()
    value = loss.detach().clone()
    if parallel.cp_group is not None:
        dist.all_reduce(value, group=parallel.cp_group)
    if parallel.mesh is not None and parallel.config.dp_size > 1:
        dist.all_reduce(value, group=parallel.group("dp"))
        value /= parallel.config.dp_size
    return float(value)
