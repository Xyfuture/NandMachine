


from dataclasses import dataclass


@dataclass
class KVCacheState:
    total_kv_cache_size_per_layer:int 
    num_nand_pages_per_layer:int 
    num_hyper_pages_per_layer:int 

    kv_block_size_tokens:int  # 一个 block 内部有多少个 token 的 kv cache， 注意不是容量的单位，不是 bytes，是 token 的个数 
    num_kv_blocks:int 
    

    # legacy 
    kv_cache_num_pages_per_layer:int