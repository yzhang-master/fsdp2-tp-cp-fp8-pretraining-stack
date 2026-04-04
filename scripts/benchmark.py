from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from pathlib import Path

import torch

from pretraining_stack.config import ParallelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reproducible BF16/FP8 benchmark commands")
    parser.add_argument("--model", choices=("tiny", "1b"), default="tiny")
    parser.add_argument(
        "--precisions",
        nargs="+",
        choices=("fp32", "bf16", "fp8"),
        default=["bf16", "fp8"],
    )
    parser.add_argument("--dp-size", type=int, default=1)
    parser.add_argument("--tp-size", type=int, default=1)
    parser.add_argument("--cp-size", type=int, default=1)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--peak-flops", type=float)
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    args = parser.parse_args()
    if args.model == "1b" and not torch.cuda.is_available():
        raise RuntimeError("the 1B benchmark requires a suitable CUDA host; no result was produced")
    bundled_torchrun = Path(sys.executable).with_name("torchrun")
    torchrun = (
        str(bundled_torchrun) if bundled_torchrun.is_file() else shutil.which("torchrun")
    )
    if torchrun is None:
        raise RuntimeError("torchrun is unavailable")
    world_size = ParallelConfig(args.dp_size, args.tp_size, args.cp_size).world_size
    root = Path(__file__).parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(root / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    for precision in args.precisions:
        output = args.output_dir / f"{args.model}_{precision}.json"
        command = [
            torchrun,
            "--nproc-per-node",
            str(world_size),
            str(root / "scripts" / "pretrain.py"),
            "--model",
            args.model,
            "--precision",
            precision,
            "--dp-size",
            str(args.dp_size),
            "--tp-size",
            str(args.tp_size),
            "--cp-size",
            str(args.cp_size),
            "--steps",
            str(args.steps),
            "--batch-size",
            str(args.batch_size),
            "--seq-len",
            str(args.seq_len),
            "--output",
            str(output),
        ]
        if args.peak_flops is not None:
            command.extend(("--peak-flops", str(args.peak_flops)))
        print(shlex.join(command), flush=True)
        subprocess.run(command, cwd=root, env=environment, check=True)


if __name__ == "__main__":
    main()
