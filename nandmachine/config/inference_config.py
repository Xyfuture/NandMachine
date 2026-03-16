from dataclasses import dataclass


@dataclass
class ParallelConfig:
    num_ranks:int
 


@dataclass
class DenseParallelConfig(ParallelConfig):
    tp_size:int 
    dp_size:int 


@dataclass
class MoEParallelConfig:
    attn_dp_size:int 

    ffn_tp_size:int
    ffn_ep_size:int 
    


@dataclass
class InferenceConfig:
    batch_size:int 

    kv_cache_num_pages_per_layer:int 

    parallel_config:ParallelConfig
