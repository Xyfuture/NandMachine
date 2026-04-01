from __future__ import annotations

import torch
from torch import nn

from nandmachine.config.inference_config import MoEParallelConfig
from nandmachine.config.model_config import Qwen3MoEModelConfig
from nandmachine.frontend.modules.modules import (
    Attention,
    FusedMoE,
    RMSNorm,
)
from nandmachine.frontend.network.qwen3 import Qwen3Attention


class Qwen3MoEAttention(Qwen3Attention):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        dp_size: int,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-6,
        qkv_bias: bool = False,
        rope_theta: float = 10000.0,
    ) -> None:
        if dp_size <= 0:
            raise ValueError(f"dp_size must be > 0, got {dp_size}")
        super().__init__(
            hidden_size=hidden_size,
            num_heads=num_heads,
            num_kv_heads=num_kv_heads,
            tp_size=tp_size,
            max_position=max_position,
            head_dim=head_dim,
            rms_norm_eps=rms_norm_eps,
            qkv_bias=qkv_bias,
            rope_theta=rope_theta,
        )
        self.dp_size = dp_size
        self.attn = Attention(
            num_heads=num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=num_kv_heads,
            tp_size=tp_size,
            dp_size=dp_size,
        )


class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: object,
        parallel_config: MoEParallelConfig,
    ) -> None:
        super().__init__()
        moe_config = Qwen3MoEModelConfig.from_config(config)
        self.self_attn = Qwen3MoEAttention(
            hidden_size=moe_config.hidden_size,
            num_heads=moe_config.num_attention_heads,
            num_kv_heads=moe_config.num_key_value_heads,
            tp_size=parallel_config.attn_tp_size,
            dp_size=parallel_config.attn_dp_size,
            max_position=moe_config.max_position_embeddings,
            head_dim=moe_config.head_dim,
            rms_norm_eps=moe_config.rms_norm_eps,
            qkv_bias=moe_config.attention_bias,
            rope_theta=moe_config.rope_theta,
        )
        self.mlp = FusedMoE(
            hidden_size=moe_config.hidden_size,
            intermediate_size=moe_config.moe_intermediate_size,
            num_experts=moe_config.num_experts,
            top_k=moe_config.num_experts_per_tok,
            ffn_ep_size=parallel_config.ffn_ep_size,
            ffn_tp_size=parallel_config.ffn_tp_size,
            shared_expert_intermediate_size=moe_config.shared_expert_intermediate_size,
        )
        self.input_layernorm = RMSNorm(
            moe_config.hidden_size,
            eps=moe_config.rms_norm_eps,
        )
        self.post_attention_layernorm = RMSNorm(
            moe_config.hidden_size,
            eps=moe_config.rms_norm_eps,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(hidden_states)
        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = hidden_states + residual
        return hidden_states


__all__ = [
    "Qwen3MoEAttention",
    "Qwen3MoEDecoderLayer",
]
