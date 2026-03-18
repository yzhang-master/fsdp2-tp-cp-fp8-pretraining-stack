from __future__ import annotations

import socket
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pretraining_stack.checkpoint import load_checkpoint, save_checkpoint
from pretraining_stack.config import ModelConfig, ParallelConfig
from pretraining_stack.data import context_parallel_batch
from pretraining_stack.model import LlamaForPretraining, apply_fsdp2, causal_loss
from pretraining_stack.parallel import ParallelContext, copy_reference_parameters
from pretraining_stack.trainer import training_step


def _config() -> ModelConfig:
    return ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _models(parallel: ParallelContext) -> tuple[LlamaForPretraining, LlamaForPretraining]:
    torch.manual_seed(29)
    reference = LlamaForPretraining(_config())
    sharded = LlamaForPretraining(
        _config(), tp_group=parallel.tp_group, cp_group=parallel.cp_group
    )
    copy_reference_parameters(reference, sharded, parallel.tp_rank)
    apply_fsdp2(sharded, parallel)
    return reference, sharded


def _global_loss(
    model: LlamaForPretraining, rows: torch.Tensor, parallel: ParallelContext
) -> torch.Tensor:
    inputs, labels, count = context_parallel_batch(
        rows, parallel.cp_rank, parallel.config.cp_size
    )
    value = causal_loss(model(inputs), labels, count)
    if parallel.cp_group is not None:
        dist.all_reduce(value, group=parallel.cp_group)
    if parallel.config.dp_size > 1:
        dist.all_reduce(value, group=parallel.group("dp"))
        value /= parallel.config.dp_size
    return value


def _reference_loss(
    model: LlamaForPretraining, rows: torch.Tensor, cp_size: int = 1
) -> torch.Tensor:
    labels = rows[:, 1:].clone()
    labels[:, -1] = -100
    return causal_loss(model(rows[:, :-1]), labels, rows.shape[0] * (rows.shape[1] - 2))


def _tp_cp_worker(rank: int, port: int, cp_size: int) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        parallel = ParallelContext.initialize(
            ParallelConfig(dp_size=1, tp_size=3 - cp_size, cp_size=cp_size)
        )
        reference, sharded = _models(parallel)
        rows = torch.arange(18, dtype=torch.long).reshape(2, 9) % 64
        optimizer = torch.optim.AdamW(sharded.parameters(), lr=2e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-3)
        actual = _global_loss(sharded, rows, parallel)
        expected = _reference_loss(reference, rows, cp_size)
        torch.testing.assert_close(actual, expected, atol=2e-6, rtol=2e-6)
        reference_optimizer.zero_grad(set_to_none=True)
        _reference_loss(reference, rows, cp_size).backward()
        reference_optimizer.step()
        before = training_step(sharded, optimizer, rows, parallel)
        next_rows = (rows + 7) % 64
        actual_next = _global_loss(sharded, next_rows, parallel)
        expected_next = _reference_loss(reference, next_rows, cp_size)
        torch.testing.assert_close(actual_next, expected_next, atol=3e-5, rtol=3e-5)
        assert before == float(actual)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("cp_size", [1, 2])
def test_two_rank_tp_or_cp_matches_reference_through_update(cp_size: int) -> None:
    mp.spawn(_tp_cp_worker, args=(_free_port(), cp_size), nprocs=2, join=True)


def _fsdp_worker(rank: int, port: int) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        parallel = ParallelContext.initialize(ParallelConfig(dp_size=2))
        reference, sharded = _models(parallel)
        rows = (torch.arange(18).reshape(2, 9) + rank * 11) % 64
        optimizer = torch.optim.AdamW(sharded.parameters(), lr=2e-3)
        reference_optimizer = torch.optim.AdamW(reference.parameters(), lr=2e-3)
        actual = training_step(sharded, optimizer, rows, parallel)
        reference_optimizer.zero_grad(set_to_none=True)
        reference_loss = sum(
            _reference_loss(reference, (torch.arange(18).reshape(2, 9) + r * 11) % 64)
            for r in range(2)
        ) / 2
        reference_loss.backward()
        reference_optimizer.step()
        assert actual == float(reference_loss)
        next_rows = (rows + 3) % 64
        expected_next = sum(
            _reference_loss(
                reference, (torch.arange(18).reshape(2, 9) + r * 11 + 3) % 64
            )
            for r in range(2)
        ) / 2
        torch.testing.assert_close(
            _global_loss(sharded, next_rows, parallel),
            expected_next,
            atol=3e-5,
            rtol=3e-5,
        )
    finally:
        dist.destroy_process_group()


def test_two_rank_fsdp2_averages_real_data_parallel_gradients() -> None:
    mp.spawn(_fsdp_worker, args=(_free_port(),), nprocs=2, join=True)


def _checkpoint_worker(rank: int, port: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=2
    )
    try:
        parallel = ParallelContext.initialize(ParallelConfig(tp_size=2))
        reference, sharded = _models(parallel)
        optimizer = torch.optim.AdamW(sharded.parameters(), lr=2e-3)
        rows = torch.arange(18, dtype=torch.long).reshape(2, 9) % 64
        training_step(sharded, optimizer, rows, parallel)
        save_checkpoint(
            directory, sharded, optimizer, step=1, data_cursor=2, parallel=parallel
        )
        next_rows = (rows + 5) % 64
        expected_start = _global_loss(sharded, next_rows, parallel)
        expected = training_step(sharded, optimizer, next_rows, parallel)

        _, restored = _models(parallel)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
        metadata = load_checkpoint(
            directory, restored, restored_optimizer, parallel=parallel
        )
        assert metadata == {"step": 1, "data_cursor": 2}
        actual_start = _global_loss(restored, next_rows, parallel)
        torch.testing.assert_close(actual_start, expected_start, atol=0, rtol=0)
        actual = training_step(restored, restored_optimizer, next_rows, parallel)
        assert actual == expected
    finally:
        dist.destroy_process_group()


def test_dcp_resume_restores_sharded_optimizer(tmp_path: Path) -> None:
    mp.spawn(
        _checkpoint_worker,
        args=(_free_port(), str(tmp_path / "dcp")),
        nprocs=2,
        join=True,
    )
