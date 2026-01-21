from functools import lru_cache
import torch
import torch.nn as nn
import torch.nn.functional as F
# import torch.distributed as dist  # Original distributed import
from . import hook_dist as dist  # Fake distributed for single GPU testing


__all__ = [
    "divide",
    "SiluAndMul",
    "RMSNorm",
    "LinearBase",
    "ReplicatedLinear",
    "ColumnParallelLinear",
    "MergedColumnParallelLinear",
    "QKVParallelLinear",
    "RowParallelLinear",
    "apply_rotary_emb",
    "RotaryEmbedding",
    "get_rope",
    "Attention",
]


def divide(numerator, denominator):
    assert numerator % denominator == 0
    return numerator // denominator


class SiluAndMul(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x, y = x.chunk(2, -1)
        return F.silu(x) * y
    


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # orig_dtype = x.dtype
        # x = x.float()
        # var = x.pow(2).mean(dim=-1, keepdim=True)
        # x.mul_(torch.rsqrt(var + self.eps))
        # x = x.to(orig_dtype).mul_(self.weight)
        # x = x.mul_(self.weight)
        return x

    # def add_rms_forward(
    #     self,
    #     x: torch.Tensor,
    #     residual: torch.Tensor,
    # ) -> tuple[torch.Tensor, torch.Tensor]:
    #     orig_dtype = x.dtype
    #     x = x.float().add_(residual.float())
    #     residual = x.to(orig_dtype)
    #     var = x.pow(2).mean(dim=-1, keepdim=True)
    #     x.mul_(torch.rsqrt(var + self.eps))
    #     x = x.to(orig_dtype).mul_(self.weight)
    #     return x, residual

    # def forward(
    #     self,
    #     x: torch.Tensor,
    #     residual: torch.Tensor | None = None,
    # ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # if residual is None:
        #     return self.rms_forward(x)
        # else:
            # return self.add_rms_forward(x, residual)
    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor :
        return self.rms_forward(x)




class LinearBase(nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
        tp_dim: int | None = None,
    ):
        super().__init__()
        self.tp_dim = tp_dim
        self.tp_rank = dist.get_rank()
        self.tp_size = dist.get_world_size()
        self.weight = nn.Parameter(torch.empty(output_size, input_size))
        self.weight.weight_loader = self.weight_loader # type: ignore
        if bias:
            self.bias = nn.Parameter(torch.empty(output_size))
            self.bias.weight_loader = self.weight_loader # type: ignore
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class ReplicatedLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        super().__init__(input_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param.data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class ColumnParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(input_size, divide(output_size, tp_size), bias, 0)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # type: ignore
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)


class MergedColumnParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        input_size: int,
        output_sizes: list[int],
        bias: bool = False,
    ):
        self.output_sizes = output_sizes
        super().__init__(input_size, sum(output_sizes), bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: int):
        param_data = param.data
        shard_offset = sum(self.output_sizes[:loaded_shard_id]) // self.tp_size
        shard_size = self.output_sizes[loaded_shard_id] // self.tp_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # type: ignore
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # type: ignore
        param_data.copy_(loaded_weight)


class QKVParallelLinear(ColumnParallelLinear):

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: int | None = None,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        total_num_kv_heads = total_num_kv_heads or total_num_heads
        self.head_size = head_size
        self.num_heads = divide(total_num_heads, tp_size)
        self.num_kv_heads = divide(total_num_kv_heads, tp_size)
        output_size = (total_num_heads + 2 * total_num_kv_heads) * self.head_size
        super().__init__(hidden_size, output_size, bias)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor, loaded_shard_id: str):
        param_data = param.data
        assert loaded_shard_id in ["q", "k", "v"]
        if loaded_shard_id == "q":
            shard_size = self.num_heads * self.head_size
            shard_offset = 0
        elif loaded_shard_id == "k":
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size
        else:
            shard_size = self.num_kv_heads * self.head_size
            shard_offset = self.num_heads * self.head_size + self.num_kv_heads * self.head_size
        param_data = param_data.narrow(self.tp_dim, shard_offset, shard_size) # type: ignore
        loaded_weight = loaded_weight.chunk(self.tp_size, self.tp_dim)[self.tp_rank] # type: ignore
        param_data.copy_(loaded_weight)


class RowParallelLinear(LinearBase):

    def __init__(
        self,
        input_size: int,
        output_size: int,
        bias: bool = False,
    ):
        tp_size = dist.get_world_size()
        super().__init__(divide(input_size, tp_size), output_size, bias, 1)

    def weight_loader(self, param: nn.Parameter, loaded_weight: torch.Tensor):
        param_data = param.data
        shard_size = param_data.size(self.tp_dim)
        start_idx = self.tp_rank * shard_size
        loaded_weight = loaded_weight.narrow(self.tp_dim, start_idx, shard_size) # type: ignore
        param_data.copy_(loaded_weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.linear(x, self.weight, self.bias)
        dist.all_reduce(y)
        return y






def apply_rotary_emb(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class RotaryEmbedding(nn.Module):

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: float,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        assert rotary_dim == head_size
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1).unsqueeze_(1)
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos_sin = self.cos_sin_cache[positions] # type: ignore
        cos, sin = cos_sin.chunk(2, dim=-1)
        query = apply_rotary_emb(query, cos, sin)
        key = apply_rotary_emb(key, cos, sin)
        return query, key


@lru_cache(1)
def get_rope(
    head_size: int,
    rotary_dim: int,
    max_position: int,
    base: float,
    rope_scaling: dict | None = None,
):
    assert rope_scaling is None
    rotary_emb = RotaryEmbedding(head_size, rotary_dim, max_position, base)
    return rotary_emb






class Attention(nn.Module):

    def __init__(
        self,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        
        self.seq_len = 1024

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        # context = get_context()
        # k_cache, v_cache = self.k_cache, self.v_cache
        # if k_cache.numel() and v_cache.numel():
        #     store_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
        # if context.is_prefill:
        #     if context.block_tables is not None:    # prefix cache
        #         k, v = k_cache, v_cache
        #     o = flash_attn_varlen_func(q, k, v,
        #                                max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
        #                                max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
        #                                softmax_scale=self.scale, causal=True, block_table=context.block_tables)
        # else:    # decode
        #     o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
        #                                 cache_seqlens=context.context_lens, block_table=context.block_tables, 
        #                                 softmax_scale=self.scale, causal=True)
        # return o
        
        x = q + 1 

        return x 
        
