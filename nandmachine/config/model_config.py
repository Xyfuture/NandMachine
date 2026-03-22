
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfigBase:
    attention_type: str = field(kw_only=True)


@dataclass
class Qwen3ModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    intermediate_size: int
    hidden_act: str
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    rope_theta: float = 10000.0
    num_hidden_layers: int | None = None
    attention_type: str = field(default="gqa", kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Qwen3ModelConfig":
        return cls(
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            max_position_embeddings=data["max_position_embeddings"],
            intermediate_size=data["intermediate_size"],
            hidden_act=data["hidden_act"],
            head_dim=data.get("head_dim"),
            rms_norm_eps=data.get("rms_norm_eps", 1e-6),
            attention_bias=data.get("attention_bias", False),
            rope_theta=data.get("rope_theta", 10000.0),
            num_hidden_layers=data.get("num_hidden_layers"),
            attention_type=data.get("attention_type", "gqa"),
        )


__all__ = ["ModelConfigBase", "Qwen3ModelConfig"]
