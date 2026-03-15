from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
import torch.nn as nn

from nandmachine.commands.macro import MacroOp
from nandmachine.frontend.core.graph.base import NxGraphMeta
from nandmachine.kernels.lieanr import LinearNandKernel

if TYPE_CHECKING:
    from nandmachine.frontend.core.graph.base import NxGraphMeta


def divide(numerator: int, denominator: int) -> int:
    assert numerator % denominator == 0
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
        raise NotImplemented


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


    def macro_code_gen(self, graph_meta: "NxGraphMeta"):
        pass

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class LinearBase(HookModuleBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ) -> None:
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = 0
        self.tp_size = 1

        self.input_size = input_size
        self.output_size = output_size

        self.weight = nn.Parameter(torch.empty(output_size, input_size))



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.new_zeros((*x.shape[:-1], self.weight.shape[0]))
    

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:
        m = graph_meta.inference_config.batch_size
        k = self.input_size
        n = self.output_size
        weight_bits = 16
        input_bits = 16
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
        bias: bool = False,
    ) -> None:
        tp_size = 1
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)


class RowParallelLinear(LinearBase):
    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ) -> None:
        tp_size = 1
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)


    def macro_code_gen(self, graph_meta: NxGraphMeta)->list[MacroOp]:
        macro_op_list = super().macro_code_gen(graph_meta)

        # TODO 最后加一个 all reduce 

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


class Attention(HookModuleBase):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scale: float,
        num_kv_heads: int,
    ) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = torch.tensor([])
        self.v_cache = torch.tensor([])


    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
    ) -> torch.Tensor:
        del k, v
        return q.new_zeros(q.shape)

    def macro_code_gen(self, graph_meta: NxGraphMeta) -> list[MacroOp]:


        



        return super().macro_code_gen(graph_meta)