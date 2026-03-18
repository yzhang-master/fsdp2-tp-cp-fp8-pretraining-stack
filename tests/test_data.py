import json
from pathlib import Path

import pytest
import torch

from pretraining_stack.data import (
    PackedShards,
    context_parallel_batch,
    pack_documents,
    write_packed_shards,
)


def test_packing_carries_documents_and_eos_across_rows() -> None:
    rows = pack_documents([[1, 2], [3], [4, 5, 6], [7, 8]], 3, eos_token_id=0)
    assert rows.tolist() == [[1, 2, 0, 3], [0, 4, 5, 6], [0, 7, 8, 0]]


def test_packed_shards_are_deterministic_and_verified(tmp_path: Path) -> None:
    documents = [[1, 2, 3], [4, 5], [6, 7, 8], [9, 10], [11, 12]]
    first = write_packed_shards(
        documents, tmp_path / "first", sequence_length=3, eos_token_id=0, shard_count=2
    )
    second = write_packed_shards(
        documents, tmp_path / "second", sequence_length=3, eos_token_id=0, shard_count=2
    )
    assert first == second
    dataset = PackedShards(tmp_path / "first")
    assert dataset.batch(0, 2, dp_rank=1, dp_size=2).tolist() == dataset.rows[2:4].tolist()
    shard_path = tmp_path / "first" / first["shards"][0]["path"]
    payload = bytearray(shard_path.read_bytes())
    payload[-1] ^= 1
    shard_path.write_bytes(payload)
    with pytest.raises(ValueError, match="checksum"):
        PackedShards(tmp_path / "first")


def test_manifest_records_real_rows_and_hashes(tmp_path: Path) -> None:
    manifest = write_packed_shards(
        [[1, 2, 3, 4]], tmp_path, sequence_length=2, eos_token_id=0
    )
    assert manifest["row_count"] == 1
    assert json.loads((tmp_path / "manifest.json").read_text()) == manifest


def test_context_parallel_labels_include_boundaries_and_mask_only_final_target() -> None:
    rows = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9]])
    first_inputs, first_labels, count = context_parallel_batch(rows, 0, 2)
    second_inputs, second_labels, _ = context_parallel_batch(rows, 1, 2)
    assert first_inputs.tolist() == [[1, 2, 3, 4]]
    assert first_labels.tolist() == [[2, 3, 4, 5]]
    assert second_inputs.tolist() == [[5, 6, 7, 8]]
    assert second_labels.tolist() == [[6, 7, 8, -100]]
    assert count == 7
