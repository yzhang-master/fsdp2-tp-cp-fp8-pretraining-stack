from __future__ import annotations

import torch
import torch.distributed as dist
from torch import nn
from torch.distributed.fsdp import fully_shard
from torch.nn import functional as F

from .config import ModelConfig
from .fp8 import convert_linears
from .parallel import (
    ColumnParallelLinear,
    ParallelContext,
    RowParallelLinear,
    group_rank,
    group_size,
    synchronize_replicated_parameters,
)


class _GatherSequence(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: object, tensor: torch.Tensor, group: dist.ProcessGroup
    ) -> torch.Tensor:
        ctx.group = group
        ctx.rank = dist.get_rank(group)
        ctx.world_size = dist.get_world_size(group)
        shards = [torch.empty_like(tensor) for _ in range(ctx.world_size)]
        dist.all_gather(shards, tensor.contiguous(), group=group)
        return torch.cat(shards, dim=2)

    @staticmethod
    def backward(ctx: object, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        grad = grad_output.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        return grad.chunk(ctx.world_size, dim=2)[ctx.rank].contiguous(), None


def gather_sequence(
    tensor: torch.Tensor, group: dist.ProcessGroup | None
) -> torch.Tensor:
    if group is None:
        return tensor
    return _GatherSequence.apply(tensor, group)


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        variance = x.float().pow(2).mean(-1, keepdim=True)
        return (x.float() * torch.rsqrt(variance + self.eps)).to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


def rope_tables(
    positions: torch.Tensor, head_dim: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    half = head_dim // 2
    frequencies = 1.0 / (10000 ** (torch.arange(half, device=positions.device).float() / half))
    angles = torch.outer(positions.float(), frequencies)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1).to(dtype)
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1).to(dtype)
    return cos[None, None], sin[None, None]


def apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    return x * cos + rotate_half(x) * sin


class ContextParallelAttention(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        tp_group: dist.ProcessGroup | None = None,
        cp_group: dist.ProcessGroup | None = None,
        fp8: bool = False,
    ) -> None:
        super().__init__()
        self.heads = config.num_attention_heads
        self.kv_heads = config.num_key_value_heads
        self.head_dim = config.hidden_size // config.num_attention_heads
        self.tp_size = group_size(tp_group)
        self.cp_size = group_size(cp_group)
        self.cp_rank = group_rank(cp_group)
        if self.heads % self.tp_size or self.kv_heads % self.tp_size:
            raise ValueError("attention and key-value heads must be divisible by TP size")
        self.local_heads = self.heads // self.tp_size
        self.local_kv_heads = self.kv_heads // self.tp_size
        self.cp_group = cp_group
        hidden = config.hidden_size
        self.q_proj = ColumnParallelLinear(
            hidden, hidden, group=tp_group, fp8=fp8
        )
        self.k_proj = ColumnParallelLinear(
            hidden, self.kv_heads * self.head_dim, group=tp_group, fp8=fp8
        )
        self.v_proj = ColumnParallelLinear(
            hidden, self.kv_heads * self.head_dim, group=tp_group, fp8=fp8
        )
        self.o_proj = RowParallelLinear(hidden, hidden, group=tp_group, fp8=fp8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, local_length, _ = x.shape
        position_offset = self.cp_rank * local_length
        q = self.q_proj(x).view(
            batch, local_length, self.local_heads, self.head_dim
        ).transpose(1, 2)
        k = self.k_proj(x).view(
            batch, local_length, self.local_kv_heads, self.head_dim
        ).transpose(1, 2)
        v = self.v_proj(x).view(
            batch, local_length, self.local_kv_heads, self.head_dim
        ).transpose(1, 2)
        q_cos, q_sin = rope_tables(
            torch.arange(position_offset, position_offset + local_length, device=x.device),
            self.head_dim,
            x.dtype,
        )
        k_cos, k_sin = rope_tables(
            torch.arange(local_length, device=x.device), self.head_dim, x.dtype
        )
        q, k = apply_rope(q, q_cos, q_sin), apply_rope(k, k_cos, k_sin)
        k, v = gather_sequence(k, self.cp_group), gather_sequence(v, self.cp_group)
        k = k.repeat_interleave(self.local_heads // self.local_kv_heads, dim=1)
        v = v.repeat_interleave(self.local_heads // self.local_kv_heads, dim=1)
        if self.cp_size == 1:
            y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        else:
            query_positions = position_offset + torch.arange(local_length, device=x.device)
            key_positions = torch.arange(k.shape[-2], device=x.device)
            allowed = key_positions[None, :] <= query_positions[:, None]
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed)
        y = y.transpose(1, 2).contiguous().view(batch, local_length, -1)
        return self.o_proj(y)


class TensorParallelMLP(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        tp_group: dist.ProcessGroup | None = None,
        fp8: bool = False,
    ) -> None:
        super().__init__()
        self.gate_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, group=tp_group, fp8=fp8
        )
        self.up_proj = ColumnParallelLinear(
            config.hidden_size, config.intermediate_size, group=tp_group, fp8=fp8
        )
        self.down_proj = RowParallelLinear(
            config.intermediate_size, config.hidden_size, group=tp_group, fp8=fp8
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class LlamaBlock(nn.Module):
    def __init__(
        self,
        config: ModelConfig,
        tp_group: dist.ProcessGroup | None = None,
        cp_group: dist.ProcessGroup | None = None,
        fp8: bool = False,
    ) -> None:
        super().__init__()
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.self_attn = ContextParallelAttention(config, tp_group, cp_group, fp8)
        self.mlp = TensorParallelMLP(config, tp_group, fp8)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attn(self.input_layernorm(x))
        return x + self.mlp(self.post_attention_layernorm(x))


class LlamaForPretraining(nn.Module):
    def __init__(
        self,
        config: ModelConfig | None = None,
        *,
        tp_group: dist.ProcessGroup | None = None,
        cp_group: dist.ProcessGroup | None = None,
        fp8: bool = False,
    ) -> None:
        super().__init__()
        config = config or ModelConfig()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            LlamaBlock(config, tp_group, cp_group, fp8)
            for _ in range(config.num_hidden_layers)
        )
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if fp8:
            convert_linears(self)
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, ColumnParallelLinear, RowParallelLinear)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        if input_ids.shape[1] > self.config.max_position_embeddings:
            raise ValueError("local sequence exceeds maximum position embeddings")
        x = self.embed_tokens(input_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.norm(x))


def build_model(
    config: ModelConfig,
    parallel: ParallelContext,
    *,
    fp8: bool = False,
    seed: int = 123,
) -> LlamaForPretraining:
    torch.manual_seed(seed + parallel.tp_rank)
    model = LlamaForPretraining(
        config, tp_group=parallel.tp_group, cp_group=parallel.cp_group, fp8=fp8
    )
    synchronize_replicated_parameters(model, parallel.tp_group)
    return model


def apply_fsdp2(
    model: LlamaForPretraining, parallel: ParallelContext
) -> LlamaForPretraining:
    for layer in model.layers:
        fully_shard(layer, mesh=parallel.dp_mesh)
    fully_shard(model, mesh=parallel.dp_mesh)
    return model


def causal_loss(
    logits: torch.Tensor, labels: torch.Tensor, target_count: int
) -> torch.Tensor:
    if target_count < 1:
        raise ValueError("target count must be positive")
    return F.cross_entropy(
        logits.float().flatten(0, 1),
        labels.flatten(),
        ignore_index=-100,
        reduction="sum",
    ) / target_count
