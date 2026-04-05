# fsdp2-tp-cp-fp8-pretraining-stack

> **Era:** 2025+ · Staff AI Engineer training systems. Techniques from 2022–2025, reimplemented in 2026 on current PyTorch.

## Problem

Large-model pretraining needs data, tensor and context parallelism to compose correctly, not merely coexist in a configuration file. This repository is a small but executable Llama pretraining stack with real CPU/Gloo correctness and recovery checks. The tiny proxy is not presented as a measured 1B training run.

## Method

```text
deterministic packed rows
        │ data-parallel batch
        ▼
FSDP2 fully_shard over DP mesh
        │ contiguous sequence shard over CP
        ▼
local Q + gathered K/V, absolute RoPE and global causal labels
        │ column/row TP shards over heads and MLP intermediate features
        ▼
E4M3-scaled linear inputs/weights → causal loss → AdamW → DCP resume
```

- **FSDP2:** `fully_shard` is applied per block and at the root over the DP dimension of a 3-D `DeviceMesh`. AdamW is constructed after sharding and before the first forward pass.
- **Tensor parallelism:** column-parallel Q/K/V and SwiGLU projections own disjoint output shards; row-parallel output/down projections own disjoint input shards and sum outputs with Gloo/NCCL collectives. Column inputs use identity forward and all-reduced backward; row outputs use all-reduced forward and identity backward. Tests compare every local gradient with a replicated reference.
- **Context parallelism:** each rank owns a contiguous token/query shard. K/V are gathered with an autograd operation whose backward sums query-side contributions and returns the source shard. Q and K use absolute RoPE positions. Shifted labels include the boundary between CP shards and ignore only the final sequence target. Each local loss is normalized by the global target count, then CP parameter gradients are summed.
- **FP8:** activations and weights use per-tensor absmax scaling `scale = max(abs(x))/448`, actual `torch.float8_e4m3fn` storage, dequantization, and straight-through gradients. The portable matmul runs after dequantization. **No hardware FP8 GEMM or FP8 speedup is claimed.**
- **DCP:** model and AdamW state use `torch.distributed.checkpoint`. Explicit TP-local tensors are saved under `tp_00`, `tp_01`, etc. with DP×CP checkpoint process groups, preventing DCP from incorrectly deduplicating distinct TP shards. Trainer step and data cursor are stored in `trainer.json`. Resume requires the same DP/TP/CP topology.
- **Data:** documents are packed contiguously with EOS markers into `sequence_length + 1` rows. Shard files have SHA-256 hashes in a JSON manifest. DP batch order is deterministic and wraps only when steps exhaust the rows.
- **Metrics:** each rank measures synchronized step time, local token throughput, Linux peak RSS, and CUDA peak allocated/reserved memory when available. With a supplied per-device peak, MFU is `6 * logical_parameters * local_tokens / seconds / peak_flops`; otherwise it is `null`, not invented.

## Results

Verified locally with Python 3.12.3, PyTorch 2.15.0+cpu and Gloo:

- **18 tests pass**, covering FP8 storage/gradients, causality, deterministic packing and checksum failures, CP boundary labels, local and sharded DCP/AdamW resume, FSDP2 DP gradient averaging, and TP/CP local-logit, per-parameter gradient and post-update parity.
- The final **8-process DP=2, TP=2, CP=2 FP8 recovery smoke passed**. Rank 0 reported `initial_loss=4.153505325317383` and exact post-resume continuation `resume_loss=4.120505332946777`.
- Mutation checks were run by deliberately changing the implementation and requiring failure: constant FP8 scale, EOS+1 packing, skipped DCP load, and missing TP input-gradient reduction were all caught. The sharding guard checks parameter gradients directly rather than relying only on a loss decrease.
- A real benchmark-driver run of two tiny FP8 CPU steps produced per-step throughputs **37.74829023422** and **27.002190553622153 tokens/s/device**, peak RSS **406,028,288 bytes**, and `mfu=null` because no peak FLOPS were supplied. These are tiny CPU systems measurements, not model-quality or GPU-performance claims.

| Model | Devices | Precision | tok/s/device | Peak memory | MFU | Recovery |
|---|---|---|---|---|---|---|
| Tiny proxy | 1×CPU | portable FP8 | measured above | measured CPU RSS above | n/a | passed |
| 1B configuration | 8×GPU | BF16 | **pending** | **pending** | **pending** | pending |
| 1B configuration | 8×GPU | portable FP8 | **pending** | **pending** | **pending** | pending |

Produce the pending measurements on a suitable eight-GPU host:

```bash
torchrun --nproc-per-node=8 scripts/pretrain.py --model 1b --precision bf16 \
  --dp-size 2 --tp-size 2 --cp-size 2 --steps 20 --batch-size 1 --seq-len 2048 \
  --peak-flops 9.89e14 --output results/1b_bf16.json

torchrun --nproc-per-node=8 scripts/pretrain.py --model 1b --precision fp8 \
  --dp-size 2 --tp-size 2 --cp-size 2 --steps 20 --batch-size 1 --seq-len 2048 \
  --peak-flops 1.979e15 --output results/1b_fp8.json
```

The FP8 command still uses dequantized floating-point matmuls. Its MFU is an effective-model-FLOPS measurement, not proof of hardware FP8 execution.

## Reproduce

```bash
python -m pip install -e ".[dev]"
ruff check .
pytest -q

python scripts/prepare_data.py --output data/packed --documents 32 \
  --document-length 64 --sequence-length 64 --vocab-size 32000 --shards 4

python scripts/smoke_distributed.py

torchrun --nproc-per-node=4 scripts/pretrain.py --model tiny --precision fp8 \
  --dp-size 2 --tp-size 2 --cp-size 1 --steps 2 --batch-size 2 --seq-len 64 \
  --checkpoint-dir checkpoints/tiny --output results/tiny.json

torchrun --nproc-per-node=4 scripts/pretrain.py --model tiny --precision fp8 \
  --dp-size 2 --tp-size 2 --cp-size 1 --steps 3 --batch-size 2 --seq-len 64 \
  --resume checkpoints/tiny --output results/tiny_resumed.json

python scripts/benchmark.py --model tiny --precisions bf16 fp8 \
  --steps 20 --batch-size 2 --seq-len 64 --output-dir results
```

The smoke command spawns eight CPU processes by default and creates a temporary checkpoint inside the repository. For smaller checks, use `--dp-size 1 --tp-size 1 --cp-size 2` or another factorization. All distributed commands require the product `DP × TP × CP` to equal the process count.

## Limitations

- Only tiny CPU models have run. **No 1B pretraining, GPU execution, hardware FP8 kernel, convergence study, or language-quality evaluation occurred.**
- Portable E4M3 scaling adds CPU overhead. There is no torchao amax history, block scaling, or hardware `scaled_mm` backend.
- CP gathers full K/V per rank; it is correct but is not ring attention and does not provide ring attention's activation-memory scaling. Packed rows currently allow cross-document attention, with no document-boundary attention mask.
- DCP resumes the same topology; it does not reshard a checkpoint to a different DP/TP/CP layout. Checkpoints overwrite the configured directory.
- CPU peak RSS is process-wide and not a substitute for per-rank CUDA allocator traces. MFU uses the standard `6NT` approximation and requires an accurate user-supplied device peak.

## References

- A. Dubey et al. *The Llama 3 Herd of Models.* 2024. https://arxiv.org/abs/2407.21783
- PyTorch. *Fully Sharded Data Parallel (FSDP2).* https://docs.pytorch.org/docs/stable/distributed.fsdp.fully_shard.html
- PyTorch. *Distributed Checkpoint.* https://docs.pytorch.org/docs/stable/distributed.checkpoint.html
- J. Shoeybi et al. *Megatron-LM: Training Multi-Billion Parameter Language Models Using Model Parallelism.* 2019. https://arxiv.org/abs/1909.08053
- P. Micikevicius et al. *FP8 Formats for Deep Learning.* 2022. https://arxiv.org/abs/2209.05433

## License

MIT
