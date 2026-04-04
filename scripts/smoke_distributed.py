from __future__ import annotations

import argparse
import json
import os
import socket
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from pretraining_stack.checkpoint import load_checkpoint, save_checkpoint
from pretraining_stack.config import ModelConfig, ParallelConfig
from pretraining_stack.model import apply_fsdp2, build_model
from pretraining_stack.parallel import ParallelContext
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


def run(rank: int, config: ParallelConfig, port: int, directory: str) -> None:
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo", init_method=f"tcp://127.0.0.1:{port}", rank=rank, world_size=config.world_size
    )
    try:
        parallel = ParallelContext.initialize(config)
        model = build_model(_config(), parallel, fp8=True, seed=31)
        model = apply_fsdp2(model, parallel)
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3)
        rows = (torch.arange(18).reshape(2, 9) + parallel.dp_rank * 11) % 64
        initial_loss = training_step(model, optimizer, rows, parallel)
        save_checkpoint(
            directory, model, optimizer, step=1, data_cursor=2, parallel=parallel
        )
        expected = training_step(model, optimizer, (rows + 5) % 64, parallel)

        restored = build_model(_config(), parallel, fp8=True, seed=31)
        restored = apply_fsdp2(restored, parallel)
        restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=0.5)
        metadata = load_checkpoint(
            directory, restored, restored_optimizer, parallel=parallel
        )
        assert metadata == {"step": 1, "data_cursor": 2}
        actual = training_step(restored, restored_optimizer, (rows + 5) % 64, parallel)
        if actual != expected:
            raise AssertionError(f"resume loss mismatch: expected={expected}, actual={actual}")
        if rank == 0:
            print(
                json.dumps(
                    {
                        "status": "passed",
                        "world_size": config.world_size,
                        "dp": config.dp_size,
                        "tp": config.tp_size,
                        "cp": config.cp_size,
                        "precision": "fp8_e4m3_ste",
                        "initial_loss": initial_loss,
                        "resume_loss": actual,
                    }
                ),
                flush=True,
            )
    finally:
        dist.destroy_process_group()


def main() -> None:
    parser = argparse.ArgumentParser(description="All-dimensions CPU/Gloo recovery smoke")
    parser.add_argument("--dp-size", type=int, default=2)
    parser.add_argument("--tp-size", type=int, default=2)
    parser.add_argument("--cp-size", type=int, default=2)
    parser.add_argument("--directory", type=Path, default=Path("checkpoints/smoke"))
    args = parser.parse_args()
    config = ParallelConfig(args.dp_size, args.tp_size, args.cp_size)
    if "RANK" in os.environ:
        run(int(os.environ["RANK"]), config, int(os.environ["MASTER_PORT"]), str(args.directory))
        return
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    with tempfile.TemporaryDirectory(prefix=".smoke_", dir=Path.cwd()) as directory:
        mp.start_processes(
            run,
            args=(config, port, directory),
            nprocs=config.world_size,
            start_method="spawn",
            join=True,
        )


if __name__ == "__main__":
    main()
