from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch import nn
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    get_optimizer_state_dict,
    set_model_state_dict,
    set_optimizer_state_dict,
)
from torch.distributed.checkpoint.stateful import Stateful

from .parallel import ParallelContext

METADATA_FILE = "trainer.json"


class CheckpointState(Stateful):
    def __init__(self, model: nn.Module, optimizer: torch.optim.Optimizer) -> None:
        self.model = model
        self.optimizer = optimizer

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": get_model_state_dict(self.model),
            "optimizer": get_optimizer_state_dict(self.model, self.optimizer),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        set_model_state_dict(self.model, state_dict["model"])
        set_optimizer_state_dict(
            self.model, self.optimizer, optim_state_dict=state_dict["optimizer"]
        )


def _checkpoint_directories(
    directory: str | Path, parallel: ParallelContext | None
) -> tuple[Path, Path, dist.ProcessGroup | None]:
    root = Path(directory)
    if parallel is None or parallel.mesh is None:
        return root, root, None
    return (
        root,
        root / f"tp_{parallel.tp_rank:02d}",
        parallel.checkpoint_group,
    )


def save_checkpoint(
    directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    step: int,
    data_cursor: int,
    parallel: ParallelContext | None = None,
) -> None:
    root, shard_directory, process_group = _checkpoint_directories(directory, parallel)
    root.mkdir(parents=True, exist_ok=True)
    shard_directory.mkdir(parents=True, exist_ok=True)
    state = {"app": CheckpointState(model, optimizer)}
    dcp.save(
        state,
        checkpoint_id=shard_directory,
        process_group=process_group,
        no_dist=not dist.is_initialized(),
    )
    if not dist.is_initialized() or dist.get_rank() == 0:
        metadata = {"format_version": 1, "step": step, "data_cursor": data_cursor}
        (root / METADATA_FILE).write_text(json.dumps(metadata, indent=2) + "\n")
    if dist.is_initialized():
        dist.barrier()


def load_checkpoint(
    directory: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    *,
    parallel: ParallelContext | None = None,
) -> dict[str, int]:
    root, shard_directory, process_group = _checkpoint_directories(directory, parallel)
    metadata = json.loads((root / METADATA_FILE).read_text())
    state = {"app": CheckpointState(model, optimizer)}
    dcp.load(
        state,
        checkpoint_id=shard_directory,
        process_group=process_group,
        no_dist=not dist.is_initialized(),
    )
    return {"step": int(metadata["step"]), "data_cursor": int(metadata["data_cursor"])}
