from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.device_mesh import DeviceMesh, init_device_mesh

from .config import ParallelConfig
from .fp8 import linear


class _ForwardAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        output = tensor.clone()
        dist.all_reduce(output, group=group)
        return output

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return grad_output, None


class _BackwardAllReduce(torch.autograd.Function):
    @staticmethod
    def forward(ctx: object, tensor: torch.Tensor, group: dist.ProcessGroup) -> torch.Tensor:
        ctx.group = group
        return tensor

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        gradient = grad_output.contiguous().clone()
        dist.all_reduce(gradient, group=ctx.group)
        return gradient, None


def group_size(group: dist.ProcessGroup | None) -> int:
    return dist.get_world_size(group) if group is not None else 1


def group_rank(group: dist.ProcessGroup | None) -> int:
    return dist.get_rank(group) if group is not None else 0


@dataclass
class ParallelContext:
    config: ParallelConfig
    mesh: DeviceMesh | None = None
    checkpoint_group: dist.ProcessGroup | None = None

    @classmethod
    def initialize(
        cls, config: ParallelConfig, device: str = "cpu"
    ) -> ParallelContext:
        if not dist.is_initialized():
            raise RuntimeError("initialize torch.distributed before the parallel context")
        if dist.get_world_size() != config.world_size:
            raise ValueError(
                f"world size {dist.get_world_size()} does not match "
                f"DP×TP×CP={config.world_size}"
            )
        mesh = init_device_mesh(
            device,
            (config.dp_size, config.tp_size, config.cp_size),
            mesh_dim_names=("dp", "tp", "cp"),
        )
        checkpoint_groups = []
        for tp_rank in range(config.tp_size):
            ranks = [
                dp_rank * config.tp_size * config.cp_size
                + tp_rank * config.cp_size
                + cp_rank
                for dp_rank in range(config.dp_size)
                for cp_rank in range(config.cp_size)
            ]
            checkpoint_groups.append(dist.new_group(ranks))
        return cls(config, mesh, checkpoint_groups[group_rank(mesh.get_group("tp"))])

    @classmethod
    def single(cls) -> ParallelContext:
        return cls(ParallelConfig())

    def group(self, dimension: str) -> dist.ProcessGroup | None:
        if self.mesh is None:
            return None
        return self.mesh.get_group(dimension)

    @property
    def tp_group(self) -> dist.ProcessGroup | None:
        return self.group("tp")

    @property
    def cp_group(self) -> dist.ProcessGroup | None:
        return self.group("cp")

    @property
    def dp_mesh(self) -> DeviceMesh | None:
        return self.mesh["dp"] if self.mesh is not None else None

    @property
    def tp_rank(self) -> int:
        return group_rank(self.tp_group)

    @property
    def cp_rank(self) -> int:
        return group_rank(self.cp_group)

    @property
    def dp_rank(self) -> int:
        return group_rank(self.group("dp"))


class ColumnParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: dist.ProcessGroup | None = None,
        fp8: bool = False,
    ) -> None:
        super().__init__()
        self.world_size = group_size(group)
        self.rank = group_rank(group)
        self.group = group
        if out_features % self.world_size:
            raise ValueError("column-parallel output features must be divisible by TP size")
        self.in_features = in_features
        self.out_features = out_features
        self.local_out_features = out_features // self.world_size
        self.fp8 = fp8
        self.weight = nn.Parameter(torch.empty(self.local_out_features, in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.group is not None:
            x = _BackwardAllReduce.apply(x, self.group)
        return linear(x, self.weight, self.fp8)


class RowParallelLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        *,
        group: dist.ProcessGroup | None = None,
        fp8: bool = False,
    ) -> None:
        super().__init__()
        self.world_size = group_size(group)
        self.rank = group_rank(group)
        self.group = group
        if in_features % self.world_size:
            raise ValueError("row-parallel input features must be divisible by TP size")
        self.in_features = in_features
        self.out_features = out_features
        self.local_in_features = in_features // self.world_size
        self.fp8 = fp8
        self.weight = nn.Parameter(torch.empty(out_features, self.local_in_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output = linear(x, self.weight, self.fp8)
        if self.group is not None:
            output = _ForwardAllReduce.apply(output, self.group)
        return output


def copy_reference_parameters(
    reference: nn.Module, parallel: nn.Module, tp_rank: int = 0
) -> None:
    """Shard a replicated tiny reference into explicit TP modules for parity checks."""
    reference_modules = dict(reference.named_modules())
    with torch.no_grad():
        for name, module in parallel.named_modules():
            if isinstance(module, ColumnParallelLinear):
                source = reference_modules[name].weight
                local = source.chunk(module.world_size, dim=0)[tp_rank]
                module.weight.copy_(local)
            elif isinstance(module, RowParallelLinear):
                source = reference_modules[name].weight
                local = source.chunk(module.world_size, dim=1)[tp_rank]
                module.weight.copy_(local)
        parallel_state = parallel.state_dict()
        for name, parameter in parallel_state.items():
            if ".weight" not in name:
                continue
            module_name = name.rsplit(".", 1)[0]
            module = dict(parallel.named_modules())[module_name]
            if not isinstance(module, (ColumnParallelLinear, RowParallelLinear)):
                parameter.copy_(reference.state_dict()[name])


def synchronize_replicated_parameters(
    model: nn.Module, tp_group: dist.ProcessGroup | None
) -> None:
    if tp_group is None:
        return
    source_rank = dist.get_global_rank(tp_group, 0)
    modules = dict(model.named_modules())
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            module_name = name.rsplit(".", 1)[0]
            if not isinstance(
                modules[module_name], (ColumnParallelLinear, RowParallelLinear)
            ):
                dist.broadcast(parameter, src=source_rank, group=tp_group)
