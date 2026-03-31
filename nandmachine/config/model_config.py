
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class ModelConfigBase:
    attention_type: str = field(kw_only=True)


def _require_config_attr(config: object, attr_name: str) -> Any:
    if not hasattr(config, attr_name):
        raise ValueError(f"MoE config missing required attribute: {attr_name}")
    return getattr(config, attr_name)


def _get_optional_config_attr(
    config: object,
    attr_name: str,
    default: Any = None,
) -> Any:
    return getattr(config, attr_name, default)


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


@dataclass
class Qwen3MoEModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    intermediate_size: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    num_hidden_layers: int
    decoder_sparse_step: int
    mlp_only_layers: list[int]
    hidden_act: str
    ffn_type: str
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    rope_theta: float = 10000.0
    shared_expert_intermediate_size: int | None = None
    attention_type: str = field(default="gqa", kw_only=True)

    @classmethod
    def from_config(cls, config: object) -> "Qwen3MoEModelConfig":
        model_type = _require_config_attr(config, "model_type")
        if model_type != "qwen3_moe":
            raise ValueError(f"Unsupported model_type: {model_type}")

        architectures = _get_optional_config_attr(config, "architectures")
        if architectures is not None and "Qwen3MoeForCausalLM" not in architectures:
            raise ValueError(f"Unsupported architectures: {architectures}")

        hidden_act = _require_config_attr(config, "hidden_act")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")

        intermediate_size = _require_config_attr(config, "intermediate_size")
        moe_intermediate_size = _require_config_attr(config, "moe_intermediate_size")
        num_experts = _require_config_attr(config, "num_experts")
        num_experts_per_tok = _require_config_attr(config, "num_experts_per_tok")
        num_hidden_layers = _require_config_attr(config, "num_hidden_layers")
        decoder_sparse_step = _require_config_attr(config, "decoder_sparse_step")
        mlp_only_layers = _require_config_attr(config, "mlp_only_layers")

        ffn_type = _get_optional_config_attr(config, "ffn_type", "moe")
        if ffn_type != "moe":
            raise ValueError(f"Unsupported ffn_type: {ffn_type}")

        shared_expert_intermediate_size = _get_optional_config_attr(
            config,
            "shared_expert_intermediate_size",
            None,
        )
        if shared_expert_intermediate_size is not None:
            raise NotImplementedError("shared expert is not implemented")

        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be > 0")
        if moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be > 0")
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if num_experts_per_tok <= 0:
            raise ValueError("num_experts_per_tok must be > 0")
        if num_experts_per_tok > num_experts:
            raise ValueError("num_experts_per_tok must be <= num_experts")
        if num_hidden_layers <= 0:
            raise ValueError("num_hidden_layers must be > 0")
        if decoder_sparse_step <= 0:
            raise ValueError("decoder_sparse_step must be > 0")
        if not isinstance(mlp_only_layers, list):
            raise TypeError("mlp_only_layers must be a list")
        seen_mlp_only_layers: set[int] = set()
        for layer_idx in mlp_only_layers:
            if not isinstance(layer_idx, int):
                raise TypeError("mlp_only_layers entries must be ints")
            if layer_idx in seen_mlp_only_layers:
                raise ValueError("mlp_only_layers must not contain duplicates")
            seen_mlp_only_layers.add(layer_idx)
            if layer_idx < 0 or layer_idx >= num_hidden_layers:
                raise ValueError(
                    "mlp_only_layers entries must be within [0, num_hidden_layers), "
                    f"got {layer_idx} for num_hidden_layers={num_hidden_layers}"
                )

        return cls(
            hidden_size=_require_config_attr(config, "hidden_size"),
            num_attention_heads=_require_config_attr(config, "num_attention_heads"),
            num_key_value_heads=_require_config_attr(config, "num_key_value_heads"),
            max_position_embeddings=_require_config_attr(
                config,
                "max_position_embeddings",
            ),
            intermediate_size=intermediate_size,
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            num_hidden_layers=num_hidden_layers,
            decoder_sparse_step=decoder_sparse_step,
            mlp_only_layers=list(mlp_only_layers),
            hidden_act=hidden_act,
            ffn_type=ffn_type,
            head_dim=_require_config_attr(config, "head_dim"),
            rms_norm_eps=_require_config_attr(config, "rms_norm_eps"),
            attention_bias=_require_config_attr(config, "attention_bias"),
            rope_theta=_require_config_attr(config, "rope_theta"),
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )


@dataclass
class LlamaModelConfig(ModelConfigBase):
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    intermediate_size: int
    hidden_act: str
    head_dim: int | None = None
    rms_norm_eps: float = 1e-5
    attention_bias: bool = False
    mlp_bias: bool = False
    rope_theta: float = 500000.0
    rope_scaling: dict[str, Any] | None = None
    num_hidden_layers: int | None = None
    attention_type: str = field(default="gqa", kw_only=True)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "LlamaModelConfig":
        return cls(
            hidden_size=data["hidden_size"],
            num_attention_heads=data["num_attention_heads"],
            num_key_value_heads=data["num_key_value_heads"],
            max_position_embeddings=data["max_position_embeddings"],
            intermediate_size=data["intermediate_size"],
            hidden_act=data["hidden_act"],
            head_dim=data.get("head_dim"),
            rms_norm_eps=data.get("rms_norm_eps", 1e-5),
            attention_bias=data.get("attention_bias", False),
            mlp_bias=data.get("mlp_bias", False),
            rope_theta=data.get("rope_theta", 500000.0),
            rope_scaling=data.get("rope_scaling"),
            num_hidden_layers=data.get("num_hidden_layers"),
            attention_type=data.get("attention_type", "gqa"),
        )

__all__ = [
    "ModelConfigBase",
    "Qwen3ModelConfig",
    "Qwen3MoEModelConfig",
    "LlamaModelConfig",
]
