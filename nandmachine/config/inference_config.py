from dataclasses import dataclass
from typing import Literal


@dataclass
class ParallelConfig:
    num_ranks:int
 


@dataclass
class DenseParallelConfig(ParallelConfig):
    tp_size:int 
    dp_size:int 


@dataclass
class MoEParallelConfig(ParallelConfig):
    attn_dp_size:int 
    attn_tp_size:int

    ffn_tp_size:int
    ffn_ep_size:int 

    def __post_init__(self) -> None:
        if self.num_ranks <= 0:
            raise ValueError(f"num_ranks must be > 0, got {self.num_ranks}")
        if self.attn_dp_size <= 0:
            raise ValueError(f"attn_dp_size must be > 0, got {self.attn_dp_size}")
        if self.attn_tp_size <= 0:
            raise ValueError(f"attn_tp_size must be > 0, got {self.attn_tp_size}")
        if self.ffn_tp_size <= 0:
            raise ValueError(f"ffn_tp_size must be > 0, got {self.ffn_tp_size}")
        if self.ffn_ep_size <= 0:
            raise ValueError(f"ffn_ep_size must be > 0, got {self.ffn_ep_size}")

        attn_world_size = self.attn_dp_size * self.attn_tp_size
        ffn_world_size = self.ffn_ep_size * self.ffn_tp_size
        if self.num_ranks != attn_world_size or self.num_ranks != ffn_world_size:
            raise ValueError(
                "MoEParallelConfig must satisfy "
                "num_ranks == attn_dp_size * attn_tp_size == ffn_ep_size * ffn_tp_size, "
                f"got num_ranks={self.num_ranks}, attn_dp_size={self.attn_dp_size}, "
                f"attn_tp_size={self.attn_tp_size}, ffn_ep_size={self.ffn_ep_size}, "
                f"ffn_tp_size={self.ffn_tp_size}"
            )


@dataclass
class InferenceConfig:
    batch_size:int 

    input_sequence_length:int 
    output_sequence_length:int 

    weight_bits:int 
    activation_bits:int 
    kv_cache_bits:int 

    kv_block_size_bytes:int # 一个 block 有多少 bytes，不是 token 的个数 

    memory_backend: Literal["nand", "hbm"]

    parallel_config:ParallelConfig

    def __post_init__(self) -> None:
        supported_backends = {"nand", "hbm"}
        if self.memory_backend not in supported_backends:
            raise ValueError(
                f"Unsupported memory_backend={self.memory_backend}, "
                f"expected one of {sorted(supported_backends)}"
            )
