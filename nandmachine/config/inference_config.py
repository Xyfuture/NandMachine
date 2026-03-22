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

    input_sequence_length:int 
    output_sequence_length:int 

    weight_bits:int 
    activation_bits:int 
    kv_cache_bits:int 

    kv_block_size_bytes:int # 一个 block 有多少 bytes，不是 token 的个数 


    parallel_config:ParallelConfig
