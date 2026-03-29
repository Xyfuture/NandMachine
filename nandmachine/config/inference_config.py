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
