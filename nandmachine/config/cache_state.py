from __future__ import annotations

from dataclasses import dataclass


class InsufficientGPUMemoryError(RuntimeError):
    pass


@dataclass
class KVCacheState:
    total_kv_cache_size_per_layer: int   # 全局 batch size 下的 总KV Cache 容量  bytes

    kv_block_size_tokens: int # 没有TP的情况下，有 TP 的话，应该在 后面新算一个

    num_kv_blocks: int # 个 kv block size 强相关

    num_nand_pages_per_layer: int # 需要多少 page 来存储 所有的 KV Block
    num_hyper_pages_per_layer: int # page 组合成多少个 hyper page




@dataclass(frozen=True)
class BatchSizeCapacityResult:
    device_name: str
    batch_size: int
    num_ranks: int
    dp_size: int
    tp_size: int
    per_rank_capacity_bytes: int
    per_rank_weight_bytes: int
    per_rank_kv_cache_bytes: int
    per_rank_used_bytes: int
    per_rank_remaining_bytes: int
    total_capacity_bytes: int
    total_weight_bytes: int
    total_kv_cache_bytes: int
    total_used_bytes: int
    ffn_ep_size: int | None = None
    ffn_tp_size: int | None = None


__all__ = [
    "KVCacheState",
    "BatchSizeCapacityResult",
    "InsufficientGPUMemoryError",
]
