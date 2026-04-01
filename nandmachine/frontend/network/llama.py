from __future__ import annotations

import torch
from torch import nn

from nandmachine.config.model_config import LlamaModelConfig
from nandmachine.frontend.modules.modules import (
    Attention,
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RMSNorm,
    RotaryEmbedding,
    RowParallelLinear,
    SiluAndMul,
)


class LlamaAttention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        tp_size: int,
        max_position: int,
        head_dim: int | None = None,
        attention_bias: bool = False,
        rope_theta: float = 500000.0,
    ) -> None:
        super().__init__()
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {tp_size}")
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads must be divisible by tp_size, got {num_heads} and {tp_size}"
            )
        if num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_kv_heads must be divisible by tp_size, got {num_kv_heads} and {tp_size}"
            )

        self.tp_size = tp_size
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim or hidden_size // self.num_heads
        self.scaling = self.head_dim ** -0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.num_heads,
            self.num_kv_heads,
            tp_size=tp_size,
            bias=attention_bias,
        )
        self.o_proj = RowParallelLinear(
            self.num_heads * self.head_dim,
            hidden_size,
            tp_size=tp_size,
            bias=attention_bias,
        )
        self.rotary_emb = RotaryEmbedding(
            head_size=self.head_dim,
            rotary_dim=self.head_dim,
            max_position_embeddings=max_position,
            base=rope_theta,
        )
        self.attn = Attention(
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            tp_size=tp_size,
            dp_size=1,
        )
        self.register_buffer(
            "_dummy_positions",
            torch.empty(0, dtype=torch.long),
            persistent=False,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        local_num_heads = self.qkv_proj.num_heads
        local_num_kv_heads = self.qkv_proj.num_kv_heads
        q_size = local_num_heads * self.head_dim
        kv_size = local_num_kv_heads * self.head_dim

        qkv = self.qkv_proj(hidden_states)
        q, k, v = qkv.split((q_size, kv_size, kv_size), dim=-1)
        q = q.unflatten(-1, (local_num_heads, self.head_dim))
        k = k.unflatten(-1, (local_num_kv_heads, self.head_dim))
        v = v.unflatten(-1, (local_num_kv_heads, self.head_dim))

        positions_to_use = self._dummy_positions if positions is None else positions
        q, k = self.rotary_emb(positions_to_use, q, k)
        output = self.attn(q, k, v)
        output = output.flatten(-2, -1)
        return self.o_proj(output)


class LlamaMLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        tp_size: int,
        mlp_bias: bool = False,
    ) -> None:
        super().__init__()
        if hidden_act != "silu":
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {tp_size}")
        if intermediate_size % tp_size != 0:
            raise ValueError(
                "intermediate_size must be divisible by tp_size, "
                f"got {intermediate_size} and {tp_size}"
            )

        self.intermediate_size = intermediate_size
        self.local_intermediate_size = intermediate_size // tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            intermediate_size * 2,
            tp_size=tp_size,
            bias=mlp_bias,
        )
        self.act_fn = SiluAndMul(hidden_dim=self.local_intermediate_size)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_size=tp_size,
            bias=mlp_bias,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up = self.gate_up_proj(x)
        act_input = gate_up.unsqueeze(1).expand(
            -1,
            self.local_intermediate_size * 2,
            -1,
            -1,
        )
        activated = self.act_fn(act_input)[..., : self.local_intermediate_size]
        return self.down_proj(activated)


class LlamaDecoderLayer(nn.Module):
    def __init__(self, config: LlamaModelConfig, tp_size: int) -> None:
        super().__init__()
        self.self_attn = LlamaAttention(
            hidden_size=config.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            tp_size=tp_size,
            max_position=config.max_position_embeddings,
            head_dim=config.head_dim,
            attention_bias=config.attention_bias,
            rope_theta=config.rope_theta,
        )
        self.mlp = LlamaMLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            tp_size=tp_size,
            mlp_bias=config.mlp_bias,
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
    "LlamaAttention",
    "LlamaMLP",
    "LlamaDecoderLayer",
]
