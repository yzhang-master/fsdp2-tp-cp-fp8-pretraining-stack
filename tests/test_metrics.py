import torch

from pretraining_stack.config import ModelConfig, logical_parameter_count
from pretraining_stack.metrics import measure_step, start_step
from pretraining_stack.model import LlamaForPretraining


def test_logical_parameter_count_matches_unsharded_model() -> None:
    config = ModelConfig(
        vocab_size=32,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
    )
    model = LlamaForPretraining(config)
    assert logical_parameter_count(config) == sum(p.numel() for p in model.parameters())


def test_metrics_compute_real_duration_throughput_mfu_and_cpu_memory() -> None:
    started = start_step()
    torch.randn(64, 64).square().mean()
    metrics = measure_step(
        3,
        1.25,
        started,
        local_tokens=32,
        logical_parameters=1000,
        peak_flops_per_device=1e6,
    )
    assert metrics.seconds > 0
    assert metrics.tokens_per_second_per_device == 32 / metrics.seconds
    assert metrics.model_flops_utilization == 192000 / metrics.seconds / 1e6
    assert metrics.peak_rss_bytes > 0
    assert metrics.cuda_peak_allocated_bytes is None
