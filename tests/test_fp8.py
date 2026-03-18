import torch

from pretraining_stack.fp8 import dequantize_e4m3, fp8_ste, quantize_e4m3


def test_fp8_uses_e4m3_storage_and_absmax_scale() -> None:
    x = torch.tensor([-896.0, -1.0, 0.0, 0.25, 896.0])
    quantized, scale = quantize_e4m3(x)
    assert quantized.dtype == torch.float8_e4m3fn
    assert scale == torch.tensor(2.0)
    restored = dequantize_e4m3(quantized, scale)
    assert restored.abs().max() <= 896.0
    assert torch.quantile((restored - x).abs(), 0.5) <= 1.0


def test_fp8_straight_through_path_changes_values_and_preserves_gradients() -> None:
    x = torch.randn(4, 8, requires_grad=True)
    weight = torch.randn(8, 8, requires_grad=True)
    output = torch.nn.functional.linear(fp8_ste(x), fp8_ste(weight))
    output.square().mean().backward()
    assert not torch.equal(output, torch.nn.functional.linear(x, weight))
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert weight.grad is not None and torch.isfinite(weight.grad).all()


def test_fp8_rejects_integer_input() -> None:
    try:
        quantize_e4m3(torch.ones(2, dtype=torch.long))
    except TypeError as error:
        assert "floating-point" in str(error)
    else:
        raise AssertionError("integer FP8 quantization was accepted")
