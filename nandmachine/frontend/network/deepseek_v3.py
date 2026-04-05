from __future__ import annotations

import torch
from torch import nn

from nandmachine.config.inference_config import MoEParallelConfig
from nandmachine.config.model_config import DeepseekV3ModelConfig
from nandmachine.frontend.modules.modules import (
    ColumnParallelLinear,
    FusedMoE,
    MLAAttention,
    RMSNorm,
    RowParallelLinear,
)


class DeepseekV3Attention(nn.Module):
    def __init__(
        self,
        config: DeepseekV3ModelConfig,
        parallel_config: MoEParallelConfig,
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.q_lora_rank = config.q_lora_rank
        self.kv_lora_rank = config.kv_lora_rank
        self.qk_nope_head_dim = config.qk_nope_head_dim
        self.qk_rope_head_dim = config.qk_rope_head_dim
        self.v_head_dim = config.v_head_dim
        self.tp_size = parallel_config.attn_tp_size
        self.dp_size = parallel_config.attn_dp_size
        self.local_num_heads = self.num_heads // self.tp_size

        self.q_a_proj = ColumnParallelLinear(
            self.hidden_size,
            self.q_lora_rank,
            tp_size=1,
            bias=False,
        )
        self.q_a_layernorm = RMSNorm(self.q_lora_rank, eps=config.rms_norm_eps)
        self.q_b_proj = ColumnParallelLinear(
            self.q_lora_rank,
            self.num_heads * (self.qk_nope_head_dim + self.qk_rope_head_dim),
            tp_size=self.tp_size,
            bias=config.attention_bias,
        )
        self.kv_a_proj = ColumnParallelLinear(
            self.hidden_size,
            self.kv_lora_rank + self.qk_rope_head_dim,
            tp_size=1,
            bias=config.attention_bias,
        )
        self.kv_a_layernorm = RMSNorm(self.kv_lora_rank, eps=config.rms_norm_eps)
        self.attn = MLAAttention(
            num_heads=self.num_heads,
            q_lora_rank=self.q_lora_rank,
            kv_lora_rank=self.kv_lora_rank,
            qk_nope_head_dim=self.qk_nope_head_dim,
            qk_rope_head_dim=self.qk_rope_head_dim,
            v_head_dim=self.v_head_dim,
            tp_size=self.tp_size,
            dp_size=self.dp_size,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.v_head_dim,
            self.hidden_size,
            tp_size=self.tp_size,
            bias=False,
        )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        q_low_rank = self.q_a_layernorm(self.q_a_proj(hidden_states))
        q = self.q_b_proj(q_low_rank)
        q_nope_size = self.local_num_heads * self.qk_nope_head_dim
        q_rope_size = self.local_num_heads * self.qk_rope_head_dim
        q_nope, q_rope = q.split((q_nope_size, q_rope_size), dim=-1)
        q_nope = q_nope.unflatten(-1, (self.local_num_heads, self.qk_nope_head_dim))
        q_rope = q_rope.unflatten(-1, (self.local_num_heads, self.qk_rope_head_dim))

        kv = self.kv_a_proj(hidden_states)
        c_kv_cache, k_rope_cache = kv.split(
            (self.kv_lora_rank, self.qk_rope_head_dim),
            dim=-1,
        )
        c_kv_cache = self.kv_a_layernorm(c_kv_cache)

        attn_output = self.attn(q_nope, q_rope, c_kv_cache, k_rope_cache)
        attn_output = attn_output.flatten(-2, -1)
        return self.o_proj(attn_output)

class DeepseekV3MoE(FusedMoE):
    pass


class DeepseekV3DecoderLayer(nn.Module):
    def __init__(
        self,
        layer_idx: int,
        config: DeepseekV3ModelConfig,
        parallel_config: MoEParallelConfig,
    ) -> None:
        super().__init__()
        if layer_idx < 0 or layer_idx >= config.num_hidden_layers:
            raise ValueError(
                "layer_idx must be within [0, num_hidden_layers), "
                f"got layer_idx={layer_idx}, num_hidden_layers={config.num_hidden_layers}"
            )

        self.self_attn = DeepseekV3Attention(config, parallel_config)
        self.mlp = DeepseekV3MoE(
            hidden_size=config.hidden_size,
            intermediate_size=config.moe_intermediate_size,
            num_experts=config.n_routed_experts,
            top_k=config.num_experts_per_tok,
            ffn_ep_size=parallel_config.ffn_ep_size,
            ffn_tp_size=parallel_config.ffn_tp_size,
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size,
            eps=config.rms_norm_eps,
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
    "DeepseekV3Attention",
    "DeepseekV3MoE",
    "DeepseekV3DecoderLayer",
]
