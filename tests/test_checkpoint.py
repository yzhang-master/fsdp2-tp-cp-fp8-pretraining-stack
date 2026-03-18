from pathlib import Path

import torch

from pretraining_stack.checkpoint import load_checkpoint, save_checkpoint
from pretraining_stack.config import ModelConfig
from pretraining_stack.model import LlamaForPretraining
from pretraining_stack.parallel import ParallelContext
from pretraining_stack.trainer import training_step


def _model() -> LlamaForPretraining:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    torch.manual_seed(17)
    return LlamaForPretraining(config)


def test_dcp_checkpoint_restores_model_optimizer_and_next_step(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint"
    parallel = ParallelContext.single()
    model = _model()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    rows = torch.randint(0, 32, (2, 9))
    training_step(model, optimizer, rows, parallel)
    save_checkpoint(checkpoint, model, optimizer, step=1, data_cursor=2)
    expected = training_step(model, optimizer, rows, parallel)
    assert list(checkpoint.rglob("*.distcp"))

    restored = _model()
    with torch.no_grad():
        for parameter in restored.parameters():
            parameter.add_(1.0)
    restored_optimizer = torch.optim.AdamW(restored.parameters(), lr=9e-1)
    metadata = load_checkpoint(checkpoint, restored, restored_optimizer)
    assert metadata == {"step": 1, "data_cursor": 2}
    assert restored_optimizer.param_groups[0]["lr"] == 3e-3
    actual = training_step(restored, restored_optimizer, rows, parallel)
    assert actual == expected
    for expected_parameter, actual_parameter in zip(
        model.parameters(), restored.parameters(), strict=True
    ):
        assert torch.equal(expected_parameter, actual_parameter)
