import torch
from torch import nn
from torch.nn import functional as F

FP8_MAX = 448.0


def quantize_e4m3(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Absmax-scale a tensor into real E4M3 storage and return its float32 scale."""
    if not x.is_floating_point():
        raise TypeError("FP8 quantization requires a floating-point tensor")
    scale = x.detach().float().abs().amax().clamp_min(1e-12) / FP8_MAX
    quantized = (x.detach().float() / scale).clamp(-FP8_MAX, FP8_MAX).to(
        torch.float8_e4m3fn
    )
    return quantized, scale


def dequantize_e4m3(quantized: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    if quantized.dtype != torch.float8_e4m3fn:
        raise TypeError("expected torch.float8_e4m3fn storage")
    return quantized.float() * scale


def fp8_ste(x: torch.Tensor) -> torch.Tensor:
    quantized, scale = quantize_e4m3(x)
    dequantized = dequantize_e4m3(quantized, scale).to(x.dtype)
    return x + (dequantized - x).detach()


def linear(x: torch.Tensor, weight: torch.Tensor, use_fp8: bool) -> torch.Tensor:
    if use_fp8:
        x, weight = fp8_ste(x), fp8_ste(weight)
    return F.linear(x, weight)


class FP8Linear(nn.Linear):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, use_fp8=True)


def convert_linears(module: nn.Module) -> nn.Module:
    for name, child in list(module.named_children()):
        if type(child) is nn.Linear:
            replacement = FP8Linear(
                child.in_features, child.out_features, child.bias is not None
            )
            replacement.to(device=child.weight.device, dtype=child.weight.dtype)
            replacement.load_state_dict(child.state_dict())
            setattr(module, name, replacement)
        else:
            convert_linears(child)
    return module
