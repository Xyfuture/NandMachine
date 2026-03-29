from __future__ import annotations

from dataclasses import dataclass

from nandmachine.config.cache_state import KVCacheState
from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import InferenceConfig, ParallelConfig
from nandmachine.config.model_config import ModelConfigBase


@dataclass(frozen=True)
class _AttentionLayout:
    num_kv_heads: int
    head_dim: int


def _ceil_div(numerator: int, denominator: int) -> int:
    return (numerator + denominator - 1) // denominator


def _resolve_dp_size(parallel_config: ParallelConfig | None) -> int:
    if parallel_config is None:
        return 1

    if hasattr(parallel_config, "dp_size"):
        return parallel_config.dp_size
    if hasattr(parallel_config, "attn_dp_size"):
        return parallel_config.attn_dp_size
    return parallel_config.num_ranks


def _resolve_tp_size(parallel_config: ParallelConfig | None) -> int:
    if parallel_config is None:
        return 1
    if hasattr(parallel_config, "tp_size"):
        tp_size = parallel_config.tp_size
    elif hasattr(parallel_config, "attn_tp_size"):
        tp_size = parallel_config.attn_tp_size
    else:
        return 1
    if tp_size <= 0:
        raise ValueError(f"tp_size must be > 0, got {tp_size}")
    return tp_size


def _resolve_attention_layout(
    model_config: ModelConfigBase,
    parallel_config: ParallelConfig | None,
) -> _AttentionLayout:
    attention_type = model_config.attention_type.lower()
    head_dim = model_config.head_dim or (
        model_config.hidden_size // model_config.num_attention_heads
    )
    tp_size = _resolve_tp_size(parallel_config)

    match attention_type:
        case "mha":
            num_kv_heads = model_config.num_attention_heads
        case "gqa":
            num_kv_heads = model_config.num_key_value_heads
        case "mla":
            assert False, "MLA KV cache sizing is not implemented yet"
        case _:
            assert False, f"Unsupported attention_type: {model_config.attention_type}"

    if num_kv_heads % tp_size != 0:
        raise ValueError(
            f"num_kv_heads must be divisible by tp_size, got {num_kv_heads} and {tp_size}"
        )

    return _AttentionLayout(num_kv_heads=num_kv_heads // tp_size, head_dim=head_dim)


def calculate_kv_cache_state(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> KVCacheState:
    assert inference_config.batch_size >= 0
    assert inference_config.input_sequence_length >= 0
    assert inference_config.output_sequence_length >= 0
    assert inference_config.kv_cache_bits > 0
    assert inference_config.kv_block_size_bytes > 0

    layout = _resolve_attention_layout(
        model_config,
        inference_config.parallel_config,
    )
    dp_size = _resolve_dp_size(inference_config.parallel_config)
    assert dp_size > 0
    rank_batch = (inference_config.batch_size + dp_size - 1) // dp_size
    peak_sequence_length = (
        inference_config.input_sequence_length + inference_config.output_sequence_length
    )

    total_kv_values = (
        rank_batch
        * peak_sequence_length
        * layout.num_kv_heads
        * layout.head_dim
        * 2
    )
    total_bits = total_kv_values * inference_config.kv_cache_bits
    total_bytes = _ceil_div(total_bits, 8)

    per_token_kv_bits = (
        layout.num_kv_heads * layout.head_dim * 2 * inference_config.kv_cache_bits
    )
    per_token_kv_bytes = _ceil_div(per_token_kv_bits, 8)
    assert per_token_kv_bytes > 0
    kv_block_size_tokens = _ceil_div(
        inference_config.kv_block_size_bytes, per_token_kv_bytes
    )

    page_size_bytes = nand_config.page_size_bytes
    hyper_page_size_bytes = nand_config.num_plane * page_size_bytes
    num_nand_pages = _ceil_div(total_bytes, page_size_bytes) if total_bytes else 0
    num_hyper_pages = (
        _ceil_div(total_bytes, hyper_page_size_bytes)
        if total_bytes
        else 0
    )
    num_kv_blocks = (
        _ceil_div(total_bytes, inference_config.kv_block_size_bytes)
        if total_bytes
        else 0
    )

    return KVCacheState(
        total_kv_cache_size_per_layer=total_bytes,
        num_nand_pages_per_layer=num_nand_pages,
        num_hyper_pages_per_layer=num_hyper_pages,
        kv_block_size_tokens=kv_block_size_tokens,
        num_kv_blocks=num_kv_blocks,
        kv_cache_num_pages_per_layer=num_nand_pages,
    )


def build_kv_cache_state(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> KVCacheState:
    return calculate_kv_cache_state(nand_config, model_config, inference_config)


__all__ = ["build_kv_cache_state", "calculate_kv_cache_state"]
