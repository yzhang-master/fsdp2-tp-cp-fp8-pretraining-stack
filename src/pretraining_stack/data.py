from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import torch

MANIFEST = "manifest.json"


def pack_documents(
    documents: Iterable[Sequence[int]], sequence_length: int, eos_token_id: int
) -> torch.Tensor:
    """Pack complete documents contiguously into deterministic fixed-length rows."""
    if sequence_length < 1:
        raise ValueError("sequence length must be positive")
    pending: list[int] = []
    rows: list[torch.Tensor] = []
    for document in documents:
        if not document:
            raise ValueError("cannot pack an empty document")
        pending.extend(int(token) for token in document)
        pending.append(eos_token_id)
        while len(pending) >= sequence_length + 1:
            rows.append(torch.tensor(pending[: sequence_length + 1], dtype=torch.long))
            pending = pending[sequence_length + 1 :]
    if not rows:
        raise ValueError("documents do not fill one complete packed row")
    return torch.stack(rows)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_packed_shards(
    documents: Iterable[Sequence[int]],
    directory: str | Path,
    *,
    sequence_length: int,
    eos_token_id: int,
    shard_count: int = 1,
) -> dict[str, object]:
    if shard_count < 1:
        raise ValueError("shard count must be positive")
    rows = pack_documents(documents, sequence_length, eos_token_id)
    if len(rows) < shard_count:
        raise ValueError("shard count cannot exceed packed row count")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    files = []
    for shard_index in range(shard_count):
        shard = rows[shard_index::shard_count].contiguous()
        path = directory / f"shard_{shard_index:05d}.pt"
        torch.save(shard, path)
        files.append(
            {"path": path.name, "rows": len(shard), "sha256": _sha256(path)}
        )
    manifest: dict[str, object] = {
        "format_version": 1,
        "sequence_length": sequence_length,
        "eos_token_id": eos_token_id,
        "row_count": len(rows),
        "shards": files,
    }
    (directory / MANIFEST).write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


class PackedShards:
    def __init__(self, directory: str | Path, verify: bool = True) -> None:
        directory = Path(directory)
        self.manifest = json.loads((directory / MANIFEST).read_text())
        self.sequence_length = int(self.manifest["sequence_length"])
        shards = []
        for description in self.manifest["shards"]:
            path = directory / description["path"]
            if verify and _sha256(path) != description["sha256"]:
                raise ValueError(f"packed shard checksum mismatch: {path}")
            shard = torch.load(path, weights_only=True)
            if shard.shape != (description["rows"], self.sequence_length + 1):
                raise ValueError(f"packed shard shape mismatch: {path}")
            shards.append(shard)
        self.rows = torch.cat(shards)

    def batch(
        self, step: int, batch_size: int, dp_rank: int = 0, dp_size: int = 1
    ) -> torch.Tensor:
        if step < 0 or batch_size < 1 or not 0 <= dp_rank < dp_size:
            raise ValueError("invalid step, batch size, or data-parallel rank")
        start = (step * dp_size + dp_rank) * batch_size
        indices = (start + torch.arange(batch_size)) % len(self.rows)
        return self.rows[indices].clone()


def context_parallel_batch(
    packed_rows: torch.Tensor, cp_rank: int = 0, cp_size: int = 1
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return local inputs, globally shifted local labels, and global target count."""
    if packed_rows.ndim != 2 or packed_rows.shape[1] < 2:
        raise ValueError("expected packed rows [batch, sequence + 1]")
    if cp_size < 1 or not 0 <= cp_rank < cp_size:
        raise ValueError("invalid context-parallel rank")
    sequence_length = packed_rows.shape[1] - 1
    if sequence_length % cp_size:
        raise ValueError("sequence length must be divisible by CP size")
    local_length = sequence_length // cp_size
    start = cp_rank * local_length
    inputs = packed_rows[:, start : start + local_length]
    labels = packed_rows[:, start + 1 : start + local_length + 1].clone()
    if cp_rank == cp_size - 1:
        labels[:, -1] = -100
    target_count = packed_rows.shape[0] * (sequence_length - 1)
    return inputs.contiguous(), labels, target_count


def synthetic_documents(
    document_count: int = 16,
    document_length: int = 19,
    vocab_size: int = 256,
    seed: int = 1234,
) -> list[list[int]]:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(
        1, vocab_size, (document_count, document_length), generator=generator
    ).tolist()
