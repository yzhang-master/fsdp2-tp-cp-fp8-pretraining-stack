from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path

import torch
import torch.distributed as dist

from pretraining_stack.checkpoint import load_checkpoint, save_checkpoint
from pretraining_stack.config import (
    LLAMA_1B_CONFIG,
    TINY_CONFIG,
    ParallelConfig,
    logical_parameter_count,
)
from pretraining_stack.data import PackedShards, pack_documents, synthetic_documents
from pretraining_stack.metrics import measure_step, start_step
from pretraining_stack.model import apply_fsdp2, build_model
from pretraining_stack.parallel import ParallelContext
from pretraining_stack.trainer import training_step


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="FSDP2 + TP + CP + FP8 pretraining")
    parser.add_argument("--model", choices=("tiny", "1b"), default="tiny")
    parser.add_argument("--precision", choices=("fp32", "bf16", "fp8"), default="fp8")
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint-dir", type=Path)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--peak-flops", type=float)
    parser.add_argument("--output", type=Path)
    return parser


def _dataset(args: argparse.Namespace, config: object) -> PackedShards | torch.Tensor:
    if args.data is not None:
        dataset = PackedShards(args.data)
        if dataset.sequence_length != args.seq_len:
            raise ValueError("CLI sequence length must match the packed data manifest")
        return dataset
    documents = synthetic_documents(
        document_count=max(8, args.steps * args.dp_size * args.batch_size + 1),
        document_length=args.seq_len,
        vocab_size=config.vocab_size,
        seed=args.seed + 1000,
    )
    return pack_documents(documents, args.seq_len, eos_token_id=0)


def _batch(
    dataset: PackedShards | torch.Tensor,
    step: int,
    args: argparse.Namespace,
    parallel: ParallelContext,
) -> torch.Tensor:
    if isinstance(dataset, PackedShards):
        return dataset.batch(step, args.batch_size, parallel.dp_rank, args.dp_size)
    start = (step * args.dp_size + parallel.dp_rank) * args.batch_size
    indices = (start + torch.arange(args.batch_size)) % len(dataset)
    return dataset[indices].clone()


def main() -> None:
    args = _parser().parse_args()
    if args.steps < 1 or args.batch_size < 1 or args.seq_len < 2:
        raise ValueError("steps and batch size must be positive; sequence length must exceed 1")
    parallel_config = ParallelConfig(args.dp_size, args.tp_size, args.cp_size)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", "0")))
    if world_size > 1:
        dist.init_process_group("nccl" if use_cuda else "gloo")
        parallel = ParallelContext.initialize(parallel_config, "cuda" if use_cuda else "cpu")
    else:
        if parallel_config.world_size != 1:
            raise ValueError("multi-dimensional parallelism requires torchrun")
        parallel = ParallelContext.single()
    try:
        config = TINY_CONFIG if args.model == "tiny" else LLAMA_1B_CONFIG
        dataset = _dataset(args, config)
        device = torch.device("cuda" if use_cuda else "cpu")
        model = build_model(config, parallel, fp8=args.precision == "fp8", seed=args.seed)
        if args.precision == "bf16":
            model = model.to(torch.bfloat16)
        model = model.to(device)
        if parallel.mesh is not None:
            model = apply_fsdp2(model, parallel)
        optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
        first_step = 0
        if args.resume is not None:
            metadata = load_checkpoint(args.resume, model, optimizer, parallel=parallel)
            first_step = metadata["step"]
        if use_cuda:
            torch.cuda.reset_peak_memory_stats()
        records = []
        last_checkpoint_step = first_step
        for step in range(first_step, args.steps):
            rows = _batch(dataset, step, args, parallel).to(device)
            started = start_step()
            loss = training_step(model, optimizer, rows, parallel)
            records.append(
                measure_step(
                    step,
                    loss,
                    started,
                    local_tokens=args.batch_size * args.seq_len // args.cp_size,
                    logical_parameters=logical_parameter_count(config),
                    peak_flops_per_device=args.peak_flops,
                ).to_dict()
            )
            completed = step + 1
            if args.checkpoint_dir is not None and (
                args.checkpoint_every and completed % args.checkpoint_every == 0
            ):
                save_checkpoint(
                    args.checkpoint_dir,
                    model,
                    optimizer,
                    step=completed,
                    data_cursor=completed * args.dp_size * args.batch_size,
                    parallel=parallel,
                )
                last_checkpoint_step = completed
        if args.checkpoint_dir is not None and last_checkpoint_step != args.steps:
            save_checkpoint(
                args.checkpoint_dir,
                model,
                optimizer,
                step=args.steps,
                data_cursor=args.steps * args.dp_size * args.batch_size,
                parallel=parallel,
            )
        result = {
            "measured_at": datetime.now(UTC).isoformat(),
            "torch_version": torch.__version__,
            "device": "cuda" if use_cuda else "cpu",
            "backend": "nccl" if use_cuda and world_size > 1 else "gloo",
            "model": args.model,
            "precision": args.precision,
            "parallelism": {"dp": args.dp_size, "tp": args.tp_size, "cp": args.cp_size},
            "logical_parameters": logical_parameter_count(config),
            "first_step": first_step,
            "steps_completed": args.steps,
            "hardware_fp8_gemm": False,
            "rank_metrics": records if rank == 0 else [],
        }
        if rank == 0:
            print(json.dumps(result), flush=True)
            if args.output is not None:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(result, indent=2) + "\n")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
