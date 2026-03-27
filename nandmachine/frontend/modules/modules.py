from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from nandmachine.commands.macro import AllReduceOp, MacroOp, MatMulOp, VectorOp
from nandmachine.frontend.core.graph.base import NxGraphMeta
from nandmachine.kernels.attention import GQANandKernel
from nandmachine.kernels.lieanr import LinearNandKernel

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
                    graph_meta.inference_config.batch_size,
                    self.hidden_size,
                ],
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
        m = graph_meta.inference_config.batch_size
        k = self.input_size
        n = self.output_size
        weight_bits = graph_meta.inference_config.weight_bits
        input_bits = graph_meta.inference_config.activation_bits
        nand_config = graph_meta.nand_config
        macro_op_list = LinearNandKernel.lowering(
            m,k,n,
            weight_bits,
            input_bits,
            nand_config
        )
        return macro_op_list


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
            graph_meta.inference_config.batch_size
            * self.output_size
            * (activation_bits // 8)
        )
        macro_op_list.append(
            AllReduceOp(
                num_ranks=self.tp_size,
                data_size=output_bytes,
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
    ) -> None:
        super().__init__()
        if num_heads <= 0:
            raise ValueError(f"num_heads must be > 0, got {num_heads}")
        if num_kv_heads <= 0:
            raise ValueError(f"num_kv_heads must be > 0, got {num_kv_heads}")
        if num_heads % num_kv_heads != 0:
            raise ValueError(
                f"num_heads must be divisible by num_kv_heads, got {num_heads} and {num_kv_heads}"
            )
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads



    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        del k, v
        return q.new_zeros(q.shape)

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:

        # graph meta 需要添加什么参数呢？
        
        # prefill 和 decode 关键的区分点

        # 目前只考虑 decode 阶段的事情


        
        params = self.build_gqa_kernel_param(graph_meta)
        macro_op_list = GQANandKernel.lowering(*params)

        return macro_op_list





    def build_gqa_kernel_param(self,graph_meta:NxGraphMeta):
        model_config = graph_meta.model_config
        inference_config = graph_meta.inference_config
        kv_cache_state = graph_meta.kv_cache_state

        if model_config.attention_type != 'gqa':
            raise ValueError(
                f"Attention macro codegen only supports gqa, got {model_config.attention_type}"
            )

        expected_tp_size = divide(model_config.num_key_value_heads, self.num_kv_heads)
        if model_config.num_attention_heads != self.num_heads * expected_tp_size:
            raise ValueError("Attention local heads do not match model_config tensor parallel split")

        actual_tp_size = getattr(inference_config.parallel_config, "tp_size", 1)
        if actual_tp_size != expected_tp_size:
            raise ValueError(
                f"inference_config.parallel_config.tp_size must be {expected_tp_size}, got {actual_tp_size}"
            )

        group_size = divide(self.num_heads, self.num_kv_heads)
        num_kv_heads = self.num_kv_heads
        head_dim = self.head_dim
        
        # TODO Fix this 
        num_kv_blocks:int = kv_cache_state.num_kv_blocks
        kv_block_size:int = kv_cache_state.kv_block_size_tokens
        block_bytes:int = inference_config.kv_block_size_bytes

        kv_cache_bits:int = inference_config.kv_cache_bits
        input_bits:int = inference_config.activation_bits
        nand_config = graph_meta.nand_config

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
                vector_shape=[graph_meta.inference_config.batch_size,self.hidden_dim]
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
