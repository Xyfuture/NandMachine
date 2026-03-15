from dataclasses import dataclass


@dataclass
class ParallelConfig:
    num_ranks:int = 1
    tp_size:int = 1
    ep_size:int = 1


@dataclass
class InferenceConfig:
    batch_size:int 

    kv_cache_num_pages_per_layer:int 

    parallel_config:ParallelConfig
