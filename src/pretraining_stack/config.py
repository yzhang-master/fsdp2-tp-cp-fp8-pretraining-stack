from dataclasses import dataclass


@dataclass(frozen=True)
class ModelConfig:
    vocab_size: int = 128
    hidden_size: int = 64
    intermediate_size: int = 128
    num_hidden_layers: int = 2
    num_attention_heads: int = 4
    num_key_value_heads: int = 2
    max_position_embeddings: int = 128
    rms_norm_eps: float = 1e-6

    def __post_init__(self) -> None:
        if min(
            self.vocab_size,
            self.hidden_size,
            self.intermediate_size,
            self.num_hidden_layers,
            self.num_attention_heads,
            self.num_key_value_heads,
            self.max_position_embeddings,
        ) < 1:
            raise ValueError("model dimensions must be positive")
        if self.hidden_size % self.num_attention_heads:
            raise ValueError("hidden size must be divisible by attention heads")
        if self.num_attention_heads % self.num_key_value_heads:
            raise ValueError("attention heads must be divisible by key-value heads")
        if (self.hidden_size // self.num_attention_heads) % 2:
            raise ValueError("attention head dimension must be even for RoPE")


@dataclass(frozen=True)
class ParallelConfig:
    dp_size: int = 1
    tp_size: int = 1
    cp_size: int = 1

    def __post_init__(self) -> None:
        if min(self.dp_size, self.tp_size, self.cp_size) < 1:
            raise ValueError("parallel dimensions must be positive")

    @property
    def world_size(self) -> int:
        return self.dp_size * self.tp_size * self.cp_size


TINY_CONFIG = ModelConfig(
    vocab_size=256,
    hidden_size=64,
    intermediate_size=128,
    num_hidden_layers=2,
    num_attention_heads=4,
    num_key_value_heads=2,
)

LLAMA_1B_CONFIG = ModelConfig(
    vocab_size=32000,
    hidden_size=2048,
    intermediate_size=5632,
    num_hidden_layers=22,
    num_attention_heads=16,
    num_key_value_heads=4,
    max_position_embeddings=4096,
)
