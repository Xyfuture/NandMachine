from __future__ import annotations

from dataclasses import dataclass


class InsufficientGPUMemoryError(RuntimeError):
    pass


@dataclass
class KVCacheState:
    total_kv_cache_size_per_layer: int   # 全局 batch size 下的 总KV Cache 容量 
    num_nand_pages_per_layer: int
    num_hyper_pages_per_layer: int

    kv_block_size_tokens: int
    num_kv_blocks: int

    # legacy
    kv_cache_num_pages_per_layer: int


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
