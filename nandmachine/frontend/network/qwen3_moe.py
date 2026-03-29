from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nandmachine.commands.macro import All2AllOp, MacroOp, VectorOp
from nandmachine.config.inference_config import MoEParallelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta
from nandmachine.frontend.modules.modules import (
    Attention,
    ColumnParallelLinear,
    HookModuleBase,
    MergedColumnParallelLinear,
    RMSNorm,
    RowParallelLinear,
    SiluAndMul,
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
            num_heads=self.num_heads,
            head_dim=self.head_dim,
            scale=self.scaling,
            num_kv_heads=self.num_kv_heads,
            tp_size=tp_size,
            dp_size=dp_size,
        )

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


@dataclass(frozen=True)
class Qwen3MoEConfigView:
    hidden_size: int
    num_attention_heads: int
    num_key_value_heads: int
    max_position_embeddings: int
    moe_intermediate_size: int
    num_experts: int
    num_experts_per_tok: int
    hidden_act: str
    ffn_type: str
    head_dim: int | None = None
    rms_norm_eps: float = 1e-6
    attention_bias: bool = False
    rope_theta: float = 10000.0
    shared_expert_intermediate_size: int | None = None

    @classmethod
    def from_config(cls, config: object) -> "Qwen3MoEConfigView":
        model_type = _require_config_attr(config, "model_type")
        if model_type != "qwen3_moe":
            raise ValueError(f"Unsupported model_type: {model_type}")

        architectures = _get_optional_config_attr(config, "architectures")
        if architectures is not None and "Qwen3MoeForCausalLM" not in architectures:
            raise ValueError(f"Unsupported architectures: {architectures}")

        hidden_act = _require_config_attr(config, "hidden_act")
        if hidden_act != "silu":
            raise ValueError(f"Unsupported hidden_act: {hidden_act}")

        moe_intermediate_size = _require_config_attr(config, "moe_intermediate_size")
        num_experts = _require_config_attr(config, "num_experts")
        num_experts_per_tok = _require_config_attr(config, "num_experts_per_tok")

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

        if moe_intermediate_size <= 0:
            raise ValueError("moe_intermediate_size must be > 0")
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if num_experts_per_tok <= 0:
            raise ValueError("num_experts_per_tok must be > 0")
        if num_experts_per_tok > num_experts:
            raise ValueError("num_experts_per_tok must be <= num_experts")

        return cls(
            hidden_size=_require_config_attr(config, "hidden_size"),
            num_attention_heads=_require_config_attr(config, "num_attention_heads"),
            num_key_value_heads=_require_config_attr(config, "num_key_value_heads"),
            max_position_embeddings=_require_config_attr(
                config,
                "max_position_embeddings",
            ),
            moe_intermediate_size=moe_intermediate_size,
            num_experts=num_experts,
            num_experts_per_tok=num_experts_per_tok,
            hidden_act=hidden_act,
            ffn_type=ffn_type,
            head_dim=_require_config_attr(config, "head_dim"),
            rms_norm_eps=_require_config_attr(config, "rms_norm_eps"),
            attention_bias=_require_config_attr(config, "attention_bias"),
            rope_theta=_require_config_attr(config, "rope_theta"),
            shared_expert_intermediate_size=shared_expert_intermediate_size,
        )


class RouterLinear(ColumnParallelLinear):
    def __init__(self, hidden_size: int, num_experts: int) -> None:
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        super().__init__(hidden_size, num_experts, tp_size=1, bias=False)
        self.hidden_size = hidden_size
        self.num_experts = num_experts


class TopKRouter(HookModuleBase):
    def __init__(self, num_experts: int, top_k: int) -> None:
        super().__init__()
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")
        self.num_experts = num_experts
        self.top_k = top_k

    def forward(self, logits: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if logits.shape[-1] != self.num_experts:
            raise ValueError(
                f"Router logits last dim must be {self.num_experts}, got {logits.shape[-1]}"
            )
        probs = torch.softmax(logits, dim=-1)
        weights, indices = torch.topk(probs, k=self.top_k, dim=-1)
        return weights, indices

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        return [
            VectorOp(
                vector_op_type="moe_topk_router",
                vector_shape=[
                    graph_meta.batch_size,
                    self.num_experts,
                    self.top_k,
                ],
            )
        ]


class ExpertMLP(HookModuleBase):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        tp_size: int,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be > 0")
        if tp_size <= 0:
            raise ValueError("tp_size must be > 0")
        if intermediate_size % tp_size != 0:
            raise ValueError("intermediate_size must be divisible by tp_size")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.tp_size = tp_size
        self.local_intermediate_size = intermediate_size // tp_size
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            intermediate_size * 2,
            tp_size=tp_size,
            bias=False,
        )
        self.act_fn = SiluAndMul(hidden_dim=self.local_intermediate_size)
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            tp_size=tp_size,
            bias=False,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"Expert input last dim must be {self.hidden_size}, got {x.shape[-1]}"
            )
        gate_up = self.gate_up_proj(x)
        gate, up = gate_up.chunk(2, dim=-1)
        activated = F.silu(gate) * up
        return self.down_proj(activated)

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        macro_op_list: list[MacroOp] = []
        macro_op_list.extend(self.gate_up_proj.macro_code_gen(graph_meta))
        macro_op_list.extend(self.act_fn.macro_code_gen(graph_meta))
        macro_op_list.extend(self.down_proj.macro_code_gen(graph_meta))
        return macro_op_list


class FusedMoE(HookModuleBase):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        top_k: int,
        ffn_ep_size: int = 1,
        ffn_tp_size: int = 1,
        shared_expert_intermediate_size: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_size <= 0:
            raise ValueError("hidden_size must be > 0")
        if intermediate_size <= 0:
            raise ValueError("intermediate_size must be > 0")
        if num_experts <= 0:
            raise ValueError("num_experts must be > 0")
        if top_k <= 0:
            raise ValueError("top_k must be > 0")
        if top_k > num_experts:
            raise ValueError("top_k must be <= num_experts")
        if ffn_ep_size <= 0:
            raise ValueError("ffn_ep_size must be > 0")
        if num_experts % ffn_ep_size != 0:
            raise ValueError("num_experts must be divisible by ffn_ep_size")
        if ffn_tp_size <= 0:
            raise ValueError("ffn_tp_size must be > 0")
        if intermediate_size % ffn_tp_size != 0:
            raise ValueError("intermediate_size must be divisible by ffn_tp_size")
        if shared_expert_intermediate_size is not None:
            raise NotImplementedError("shared expert is not implemented")

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_experts = num_experts
        self.top_k = top_k
        self.ffn_ep_size = ffn_ep_size
        self.ffn_tp_size = ffn_tp_size
        self.local_expert_count = num_experts // ffn_ep_size

        self.gate = RouterLinear(hidden_size, num_experts)
        self.router = TopKRouter(num_experts, top_k)
        self.experts = nn.ModuleList(
            [
                ExpertMLP(
                    hidden_size,
                    intermediate_size,
                    tp_size=ffn_tp_size,
                )
                for _ in range(self.local_expert_count)
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.ndim != 3:
            raise ValueError(f"FusedMoE expects a 3D tensor, got ndim={x.ndim}")
        if x.shape[-1] != self.hidden_size:
            raise ValueError(
                f"FusedMoE input last dim must be {self.hidden_size}, got {x.shape[-1]}"
            )

        # batch_size, seq_len, hidden_size = x.shape
        # flat_x = x.reshape(batch_size * seq_len, hidden_size)
        # logits = self.gate(flat_x)
        # weights, indices = self.router(logits)

        # output = flat_x.new_zeros(flat_x.shape)
        # for expert_id, expert in enumerate(self.experts):
        #     expert_output = expert(flat_x)
        #     token_weight = (
        #         indices.eq(expert_id).to(flat_x.dtype) * weights
        #     ).sum(dim=-1, keepdim=True)
        #     output = output + expert_output * token_weight
        # return output.reshape(batch_size, seq_len, hidden_size)

        return x+1 

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        parallel_config = graph_meta.inference_config.parallel_config

        assert isinstance(parallel_config, MoEParallelConfig)
        ffn_world_size = parallel_config.ffn_ep_size * parallel_config.ffn_tp_size
        attn_world_size = parallel_config.attn_dp_size * parallel_config.attn_tp_size
        if ffn_world_size != attn_world_size:
            raise ValueError(
                "ffn_ep_size * ffn_tp_size must equal attn_dp_size * attn_tp_size"
            )

        expert_batch_size = max(
            1,
            math.ceil(graph_meta.batch_size * self.top_k / self.num_experts),
        )
        expert_graph_meta = graph_meta.with_batch_size(expert_batch_size)
        macro_op_list: list[MacroOp] = []
        macro_op_list.extend(self.gate.macro_code_gen(graph_meta))
        macro_op_list.extend(self.router.macro_code_gen(graph_meta))

        if ffn_world_size != 1:
            op = self._build_pre_expert_communication(graph_meta)
            op.add_inputs(macro_op_list[-1])
            macro_op_list.append(op)

        for expert in self.experts:
            op_list = expert.macro_code_gen(expert_graph_meta)
            op_list[0].add_inputs(macro_op_list[-1])
            macro_op_list.extend(op_list)

        if ffn_world_size != 1:
            op = self._build_post_expert_communication(graph_meta)
            op.add_inputs(macro_op_list[-1])
            macro_op_list.append(op)

        macro_op_list.append(
            VectorOp(
                vector_op_type="moe_weighted_sum",
                vector_shape=[graph_meta.batch_size, self.hidden_size],
            ).with_inputs(macro_op_list[-1])
        )

        return macro_op_list

    # def _build_all_to_all_op(self, graph_meta: NxGraphMeta) -> All2AllOp:
    #     parallel_config = graph_meta.inference_config.parallel_config


    #     activation_bits = graph_meta.inference_config.activation_bits
    #     if activation_bits <= 0 or activation_bits % 8 != 0:
    #         raise ValueError("activation_bits must be a positive multiple of 8")

    #     total_payload_bytes = (
    #         graph_meta.batch_size
    #         * self.top_k
    #         * self.hidden_size
    #         * (activation_bits // 8)
    #     )
    #     peer_payload_bytes = math.ceil(total_payload_bytes / parallel_config.num_ranks)
    #     return All2AllOp(
    #         num_gpus=parallel_config.num_ranks,
    #         data_size=peer_payload_bytes,
    #         weight_bits=activation_bits,
    #     )


    def _build_pre_expert_communication(self,graph_meta:NxGraphMeta):
        parallel_config = graph_meta.inference_config.parallel_config
        assert isinstance(parallel_config, MoEParallelConfig)
        # 先 all to all 
        # 在 all gather -- 暂时不实现
        activation_bits = graph_meta.inference_config.activation_bits
        world_size = parallel_config.ffn_ep_size * parallel_config.ffn_tp_size
        if world_size <= 0:
            raise ValueError("ffn_ep_size * ffn_tp_size must be > 0")

        total_payload_bytes = (
            graph_meta.batch_size
            * self.top_k
            * self.hidden_size
            * (activation_bits // 8)
        )
        all_to_all_size_per_rank = (
            math.ceil(total_payload_bytes / world_size)
        )
        return All2AllOp(
            num_gpus=world_size,
            data_size=all_to_all_size_per_rank,
            weight_bits=activation_bits,
        )

    
    def _build_post_expert_communication(self,graph_meta:NxGraphMeta):
        
        # 先 reduce scatter 
        # 再 all to all -- 暂时不实现 


        parallel_config = graph_meta.inference_config.parallel_config
        assert isinstance(parallel_config, MoEParallelConfig)

        activation_bits = graph_meta.inference_config.activation_bits
        world_size = parallel_config.ffn_ep_size * parallel_config.ffn_tp_size
        if world_size <= 0:
            raise ValueError("ffn_ep_size * ffn_tp_size must be > 0")

        total_payload_bytes = (
            graph_meta.batch_size
            * self.top_k
            * self.hidden_size
            * (activation_bits // 8)
        )
        all_to_all_size_per_rank = math.ceil(total_payload_bytes / world_size)
        return All2AllOp(
            num_gpus=world_size,
            data_size=all_to_all_size_per_rank,
            weight_bits=activation_bits,
        )





class Qwen3MoEDecoderLayer(nn.Module):
    def __init__(
        self,
        config: object,
        parallel_config: MoEParallelConfig,
    ) -> None:
        super().__init__()
        moe_config = Qwen3MoEConfigView.from_config(config)
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
    "Qwen3MoEConfigView",
    "Qwen3MoEAttention",
    "RouterLinear",
    "TopKRouter",
    "ExpertMLP",
    "FusedMoE",
    "Qwen3MoEDecoderLayer",
]
