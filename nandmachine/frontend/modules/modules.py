from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from nandmachine.commands.macro import (
    All2AllOp,
    AllReduceOp,
    MacroOp,
    MatMulOp,
    SramPrefetch,
    SramPrefetchRelease,
    VectorOp,
)
from nandmachine.config.inference_config import DenseParallelConfig, MoEParallelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta
from nandmachine.kernels.attention import (
    GQAHBMKernel,
    GQANandKernel,
    MLAHBMKernel,
    MLANandKernel,
)
from nandmachine.kernels.lieanr import LinearHBMKernel, LinearNandKernel

if TYPE_CHECKING:
    from nandmachine.frontend.core.graph.base import NxGraphMeta


def divide(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"denominator must be > 0, got {denominator}")
    if numerator % denominator != 0:
        raise ValueError(
            f"{numerator} must be divisible by {denominator} for tensor parallel split"
        )
    return numerator // denominator


def ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"denominator must be > 0, got {denominator}")
    return (numerator + denominator - 1) // denominator


def get_kernel_backend(graph_meta: NxGraphMeta) -> str:
    backend = graph_meta.inference_config.memory_backend
    if backend not in {"nand", "hbm"}:
        raise ValueError(f"Unsupported memory_backend: {backend}")
    return backend




class HookModuleBase(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.module_info: dict[str, Any] = {
            "module_type": type(self).__name__,
        }

    def _record_module_info(self, **kwargs: Any) -> None:
        self.module_info.update(kwargs)

    def macro_code_gen(self, graph_meta: "NxGraphMeta")->list[MacroOp]:
        raise NotImplementedError


class RMSNorm(HookModuleBase):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))


    def macro_code_gen(self, graph_meta: "NxGraphMeta") -> list[MacroOp]:
        return [
            VectorOp(
                vector_op_type="rms_norm",
                vector_shape=[
                    graph_meta.batch_size,
                    self.hidden_size,
                ],
                weight_bits=graph_meta.inference_config.activation_bits,
            )
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearBase(HookModuleBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.weight = nn.Parameter(torch.empty(output_size, input_size))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((*x.shape[:-1], self.output_size))
    

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        m = graph_meta.batch_size
        k = self.input_size
        n = self.output_size
        weight_bits = graph_meta.inference_config.weight_bits
        input_bits = graph_meta.inference_config.activation_bits
        nand_config = graph_meta.nand_config
        backend = get_kernel_backend(graph_meta)

        if backend == "nand":
            return LinearNandKernel.lowering(
                m,
                k,
                n,
                weight_bits,
                input_bits,
                nand_config,
            )
        if backend == "hbm":
            return LinearHBMKernel.lowering(
                m,
                k,
                n,
                weight_bits,
                input_bits,
                nand_config,
            )
        raise AssertionError(f"Unhandled memory_backend: {backend}")


class ColumnParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size:int =1 ,
        bias: bool = False,
    ) -> None:
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {tp_size}")
        self.tp_size = tp_size
        self.tp_rank = 0
        self.tp_dim = 0
        self.global_input_size = input_size
        self.global_output_size = output_size
        super().__init__(input_size, divide(output_size, tp_size), bias)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        tp_size:int = 1,
        bias: bool = False,
    ) -> None:
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {tp_size}")
        self.tp_size = tp_size
        self.tp_rank = 0
        self.tp_dim = 1
        self.global_input_size = input_size
        self.global_output_size = output_size
        super().__init__(divide(input_size, tp_size), output_size, bias)


    def macro_code_gen(self, graph_meta: NxGraphMeta)->list[MacroOp]:
        macro_op_list = super().macro_code_gen(graph_meta)
        if self.tp_size == 1:
            return macro_op_list

        activation_bits = graph_meta.inference_config.activation_bits
        if activation_bits <= 0 or activation_bits % 8 != 0:
            raise ValueError(
                f"activation_bits must be a positive multiple of 8, got {activation_bits}"
            )

        matmul_ops = [
            macro_op for macro_op in macro_op_list if isinstance(macro_op, MatMulOp)
        ]
        if not matmul_ops:
            raise ValueError("RowParallelLinear expected MatMulOp before AllReduceOp")

        output_bytes = (
            graph_meta.batch_size
            * self.output_size
            * (activation_bits // 8)
        )
        macro_op_list.append(
            AllReduceOp(
                num_ranks=self.tp_size,
                data_size=output_bytes,
                weight_bits=activation_bits,
            ).with_inputs(*matmul_ops)
        )

        return macro_op_list


class RotaryEmbedding(HookModuleBase):
    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        assert rotary_dim == head_size

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        del positions
        
        return query, key
    

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        macro_op_list = [
            
        ]

        return macro_op_list


# TODO TP size 
class Attention(HookModuleBase):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
        tp_size: int,
        dp_size: int,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if head_dim <= 0:
            raise ValueError(f"head_dim must be > 0, got {head_dim}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {tp_size}")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be > 0, got {dp_size}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_kv_heads, got {num_heads} and {num_kv_heads}"
            )
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads must be divisible by tp_size, got {num_heads} and {tp_size}"
            )
        if num_kv_heads % tp_size != 0:
            raise ValueError(
                f"num_kv_heads must be divisible by tp_size, got {num_kv_heads} and {tp_size}"
            )
        

        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.local_num_heads = divide(num_heads, tp_size)
        self.local_num_kv_heads = divide(num_kv_heads, tp_size)



    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        # del k, v
        return q.new_zeros(q.shape)

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:

        # graph meta 需要添加什么参数呢？
        
        # prefill 和 decode 关键的区分点

        # 目前只考虑 decode 阶段的事情


        
        params = self.build_gqa_kernel_param(graph_meta)
        backend = get_kernel_backend(graph_meta)

        if backend == "nand":
            return GQANandKernel.lowering(*params)
        if backend == "hbm":
            return GQAHBMKernel.lowering(*params)
        raise AssertionError(f"Unhandled memory_backend: {backend}")





    def build_gqa_kernel_param(self,graph_meta:NxGraphMeta):
        model_config = graph_meta.model_config
        inference_config = graph_meta.inference_config

        if model_config.attention_type != 'gqa':
            raise ValueError(
                f"Attention macro codegen only supports gqa, got {model_config.attention_type}"
            )

        if model_config.num_attention_heads != self.num_heads:
            raise ValueError(
                "Attention global num_heads do not match model_config"
            )
        if model_config.num_key_value_heads != self.num_kv_heads:
            raise ValueError(
                "Attention global num_kv_heads do not match model_config"
            )
        expected_tp_size = divide(model_config.num_attention_heads, self.local_num_heads)
        expected_kv_tp_size = divide(
            model_config.num_key_value_heads,
            self.local_num_kv_heads,
        )
        if self.tp_size != expected_tp_size or self.tp_size != expected_kv_tp_size:
            raise ValueError(
                f"Attention tp_size must be {expected_tp_size}, got {self.tp_size}"
            )

        parallel_config = inference_config.parallel_config
        if isinstance(parallel_config, DenseParallelConfig):
            actual_dp_size = parallel_config.dp_size
        elif isinstance(parallel_config, MoEParallelConfig):
            actual_dp_size = parallel_config.attn_dp_size
        else:
            raise ValueError(
                f"Unsupported parallel_config type for Attention: {type(parallel_config).__name__}"
            )
        if actual_dp_size != self.dp_size:
            raise ValueError(
                f"Attention dp_size must be {actual_dp_size}, got {self.dp_size}"
            )
        local_batch_size = graph_meta.batch_size

        group_size = divide(self.local_num_heads, self.local_num_kv_heads)
        num_kv_heads = self.local_num_kv_heads
        head_dim = self.head_dim 
        block_bytes:int = inference_config.kv_block_size_bytes

        kv_cache_bits:int = inference_config.kv_cache_bits
        input_bits:int = inference_config.activation_bits
        nand_config = graph_meta.nand_config
        
        
        peak_sequence_length = (
            inference_config.input_sequence_length + inference_config.output_sequence_length
        )
        per_token_kv_value_count = num_kv_heads * head_dim * 2
        per_token_kv_bytes = ceil_div(per_token_kv_value_count * kv_cache_bits, 8)
        kv_block_size:int = ceil_div(block_bytes, per_token_kv_bytes)
        local_total_kv_value_count = (
            local_batch_size
            * peak_sequence_length
            * num_kv_heads
            * head_dim
            * 2
        )
        local_total_kv_bytes = ceil_div(local_total_kv_value_count * kv_cache_bits, 8)
        num_kv_blocks:int = (
            ceil_div(local_total_kv_bytes, block_bytes) if local_total_kv_bytes > 0 else 0
        )

        # TODO  引入 imbalanced 的 kv 的情况 

        kv_cache_state = graph_meta.kv_cache_state

        if kv_cache_state.is_imbalance:
            # 复写这一部分
            num_kv_blocks = kv_cache_state.num_kv_blocks
        


        return (
            group_size,
            num_kv_heads,
            head_dim,
            num_kv_blocks,

            kv_block_size,
            block_bytes,

            kv_cache_bits,
            input_bits,
            nand_config,
        )


class MLAAttention(HookModuleBase):
    def __init__(
        self,
        num_heads: int,
        q_lora_rank: int,
        kv_lora_rank: int,
        qk_nope_head_dim: int,
        qk_rope_head_dim: int,
        v_head_dim: int,
        tp_size: int,
        dp_size: int,
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if q_lora_rank <= 0:
            raise ValueError(f"q_lora_rank must be > 0, got {q_lora_rank}")
        if kv_lora_rank <= 0:
            raise ValueError(f"kv_lora_rank must be > 0, got {kv_lora_rank}")
        if qk_nope_head_dim <= 0:
            raise ValueError(f"qk_nope_head_dim must be > 0, got {qk_nope_head_dim}")
        if qk_rope_head_dim <= 0:
            raise ValueError(f"qk_rope_head_dim must be > 0, got {qk_rope_head_dim}")
        if v_head_dim <= 0:
            raise ValueError(f"v_head_dim must be > 0, got {v_head_dim}")
        if tp_size <= 0:
            raise ValueError(f"tp_size must be > 0, got {tp_size}")
        if dp_size <= 0:
            raise ValueError(f"dp_size must be > 0, got {dp_size}")
        if num_heads % tp_size != 0:
            raise ValueError(
                f"num_heads must be divisible by tp_size, got {num_heads} and {tp_size}"
            )

        self.num_heads = num_heads
        self.q_lora_rank = q_lora_rank
        self.kv_lora_rank = kv_lora_rank
        self.qk_nope_head_dim = qk_nope_head_dim
        self.qk_rope_head_dim = qk_rope_head_dim
        self.v_head_dim = v_head_dim
        self.tp_size = tp_size
        self.dp_size = dp_size
        self.local_num_heads = divide(num_heads, tp_size)

    def forward(
        self,
        q_nope: torch.Tensor,
        q_rope: torch.Tensor,
        c_kv_cache: torch.Tensor,
        k_rope_cache: torch.Tensor,
    ) -> torch.Tensor:
        del q_rope, c_kv_cache, k_rope_cache
        return q_nope.new_zeros((*q_nope.shape[:-1], self.v_head_dim))

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        (
            local_batch_size,
            local_num_kv_blocks,
            kv_block_size_tokens,
            block_bytes,
            kv_cache_bits,
            input_bits,
            nand_config,
        ) = self.build_mla_kernel_param(graph_meta)

        absorb_matmul = MatMulOp(
            (
                local_batch_size * self.local_num_heads,
                self.qk_nope_head_dim,
                self.kv_lora_rank,
            ),
            weight_bits=graph_meta.inference_config.weight_bits,
        )

        kernel_params = (
            self.local_num_heads,
            local_batch_size,
            local_num_kv_blocks,
            self.kv_lora_rank,
            self.qk_rope_head_dim,
            kv_block_size_tokens,
            block_bytes,
            kv_cache_bits,
            input_bits,
            nand_config,
        )
        backend = get_kernel_backend(graph_meta)
        if backend == "nand":
            kernel_macro_ops = MLANandKernel.lowering(*kernel_params)
        elif backend == "hbm":
            kernel_macro_ops = MLAHBMKernel.lowering(*kernel_params)
        else:
            raise AssertionError(f"Unhandled memory_backend: {backend}")

        if not kernel_macro_ops:
            raise ValueError("MLA kernel must return at least one macro op")
        kernel_macro_ops[0].add_inputs(absorb_matmul)
        tail_dependency = next(
            (
                macro_op
                for macro_op in reversed(kernel_macro_ops)
                if not isinstance(macro_op, SramPrefetchRelease)
            ),
            None,
        )
        if tail_dependency is None:
            raise ValueError("MLA kernel must return a non-release compute op")

        up_proj_matmul = MatMulOp(
            (
                local_batch_size * self.local_num_heads,
                self.kv_lora_rank,
                self.v_head_dim,
            ),
            weight_bits=graph_meta.inference_config.weight_bits,
        ).with_inputs(tail_dependency)

        return [absorb_matmul, *kernel_macro_ops, up_proj_matmul]

    def build_mla_kernel_param(self, graph_meta: NxGraphMeta):
        model_config = graph_meta.model_config
        inference_config = graph_meta.inference_config

        if model_config.attention_type != "mla":
            raise ValueError(
                f"MLAAttention only supports mla, got {model_config.attention_type}"
            )
        if model_config.num_attention_heads != self.num_heads:
            raise ValueError("MLAAttention num_heads do not match model_config")
        if model_config.kv_lora_rank != self.kv_lora_rank:
            raise ValueError("MLAAttention kv_lora_rank does not match model_config")
        if model_config.qk_nope_head_dim != self.qk_nope_head_dim:
            raise ValueError("MLAAttention qk_nope_head_dim does not match model_config")
        if model_config.qk_rope_head_dim != self.qk_rope_head_dim:
            raise ValueError("MLAAttention qk_rope_head_dim does not match model_config")
        if model_config.v_head_dim != self.v_head_dim:
            raise ValueError("MLAAttention v_head_dim does not match model_config")

        parallel_config = inference_config.parallel_config
        if not isinstance(parallel_config, MoEParallelConfig):
            raise ValueError(
                "MLAAttention requires MoEParallelConfig to derive attn_dp/attn_tp"
            )
        if parallel_config.attn_dp_size != self.dp_size:
            raise ValueError(
                f"MLAAttention dp_size must be {parallel_config.attn_dp_size}, got {self.dp_size}"
            )
        if parallel_config.attn_tp_size != self.tp_size:
            raise ValueError(
                f"MLAAttention tp_size must be {parallel_config.attn_tp_size}, got {self.tp_size}"
            )

        local_batch_size = graph_meta.batch_size
        peak_sequence_length = (
            inference_config.input_sequence_length + inference_config.output_sequence_length
        )
        per_token_kv_values = self.kv_lora_rank + self.qk_rope_head_dim
        per_token_kv_bytes = ceil_div(
            per_token_kv_values * inference_config.kv_cache_bits,
            8,
        )
        if per_token_kv_bytes <= 0:
            raise ValueError("MLA per-token KV cache bytes must be > 0")

        block_bytes = inference_config.kv_block_size_bytes
        kv_block_size_tokens = ceil_div(block_bytes, per_token_kv_bytes)
        local_total_kv_bytes = local_batch_size * peak_sequence_length * per_token_kv_bytes
        local_num_kv_blocks = (
            ceil_div(local_total_kv_bytes, block_bytes) if local_total_kv_bytes > 0 else 0
        )
        if local_num_kv_blocks <= 0:
            raise ValueError(
                "MLAAttention requires at least one local KV block, "
                f"got local_total_kv_bytes={local_total_kv_bytes}"
            )

        return (
            local_batch_size,
            local_num_kv_blocks,
            kv_block_size_tokens,
            block_bytes,
            inference_config.kv_cache_bits,
            inference_config.activation_bits,
            graph_meta.nand_config,
        )


class SiluAndMul(HookModuleBase):
    def __init__(self, hidden_dim: int = 0) -> None:
        super().__init__()

        self.hidden_dim = hidden_dim


    def forward(self,x:torch.Tensor):
        if self.hidden_dim == 0:
            self.hidden_dim = x.shape[-1] //2
        return x[:,self.hidden_dim]




    def macro_code_gen(self,graph_meta:NxGraphMeta):
        macro_op_list:list[MacroOp] = [
            VectorOp(
                vector_op_type='silu_mul',
                vector_shape=[graph_meta.batch_size,self.hidden_dim],
                weight_bits=graph_meta.inference_config.activation_bits,
            )
        ]

        return macro_op_list
    


class MergedColumnParallelLinear(ColumnParallelLinear):
    def __init__(self,
                 input_size:int,
                 output_size:int,
                 tp_size:int,
                 bias=False,
                 ) -> None:
        super().__init__(input_size,output_size,tp_size,bias)

        # 基本上复用 前一个，就是 weight load 的方式有变化，但是我们不关心这个 

    
    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        return super().macro_code_gen(graph_meta)


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
                weight_bits=graph_meta.inference_config.activation_bits,
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
        return x + 1

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
            math.ceil(graph_meta.global_batch_size * self.top_k / self.num_experts),
        )
        expert_graph_meta = graph_meta.with_batch_size(expert_batch_size)
        macro_op_list: list[MacroOp] = []
        macro_op_list.extend(self.gate.macro_code_gen(graph_meta))
        macro_op_list.extend(self.router.macro_code_gen(graph_meta))

        if ffn_world_size != 1:
            op = self._build_pre_expert_communication(graph_meta)
            op.add_inputs(macro_op_list[-1])
            macro_op_list.append(op)

        last_non_release_op = lambda ops: next(
            op for op in reversed(ops) if not isinstance(op, SramPrefetchRelease)
        )
        first_non_prefetch_op = lambda ops: next(
            op for op in ops if not isinstance(op, SramPrefetch)
        )

        # 依赖上一条指令，避免 all to all 接不起来
        dep_previous:bool = False

        first_expert:bool = True
        for expert in self.experts:
            op_list = expert.macro_code_gen(expert_graph_meta)
            if graph_meta.nand_config.enable_strict :
                if first_expert:
                    op_list[0].add_inputs(last_non_release_op(macro_op_list))
                    first_expert = False
                else:
                    first_non_prefetch_op(op_list).add_inputs(last_non_release_op(macro_op_list))
            else:
                if not dep_previous:
                    first_non_prefetch_op(op_list).add_inputs(last_non_release_op(macro_op_list))
                    dep_previous = True

            macro_op_list.extend(op_list)
            

    

        if ffn_world_size != 1:
            op = self._build_post_expert_communication(graph_meta)
            op.add_inputs(last_non_release_op(macro_op_list))
            macro_op_list.append(op)

        macro_op_list.append(
            VectorOp(
                vector_op_type="moe_weighted_sum",
                vector_shape=[graph_meta.batch_size, self.hidden_size],
                weight_bits=graph_meta.inference_config.activation_bits,
            ).with_inputs(last_non_release_op(macro_op_list))
        ) # 先不加上去了

        return macro_op_list

    def _build_pre_expert_communication(self, graph_meta: NxGraphMeta) -> All2AllOp:
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

    def _build_post_expert_communication(self, graph_meta: NxGraphMeta) -> All2AllOp:
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


class QKVParallelLinear(ColumnParallelLinear):
    def __init__(self,         
                 hidden_size: int,
                 head_size: int,
                 total_num_heads: int,
                 total_num_kv_heads: int | None = None,
                 tp_size:int = 1,
                 bias: bool = False,
                ) -> None:
            
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, tp_size,bias)



    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        return super().macro_code_gen(graph_meta)
