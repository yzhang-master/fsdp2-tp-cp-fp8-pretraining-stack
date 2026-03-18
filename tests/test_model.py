import torch

from pretraining_stack.config import ModelConfig
from pretraining_stack.model import LlamaForPretraining


def tiny_config(fp8: bool = False) -> LlamaForPretraining:
    config = ModelConfig(
        vocab_size=64,
        hidden_size=16,
        intermediate_size=32,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=16,
    )
    torch.manual_seed(7)
    return LlamaForPretraining(config, fp8=fp8)


def test_causal_attention_does_not_read_future_tokens() -> None:
    model = tiny_config().eval()
    tokens = torch.randint(0, 64, (1, 8))
    changed = tokens.clone()
    changed[:, 5:] = torch.randint(0, 64, (1, 3))
    with torch.no_grad():
        first, second = model(tokens), model(changed)
    torch.testing.assert_close(first[:, :5], second[:, :5], atol=0, rtol=0)


def test_fp8_model_has_real_gradients() -> None:
    model = tiny_config(fp8=True)
    tokens = torch.randint(0, 64, (2, 8))
    loss = model(tokens).square().mean()
    loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients)


def test_invalid_gqa_configuration_is_rejected() -> None:
    try:
        ModelConfig(hidden_size=16, num_attention_heads=4, num_key_value_heads=3)
    except ValueError as error:
        assert "divisible" in str(error)
    else:
        raise AssertionError("invalid GQA configuration was accepted")
