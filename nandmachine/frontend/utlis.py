from __future__ import annotations

from dataclasses import dataclass
from random import Random

from nandmachine.config.cache_state import KVCacheState
from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import (
    InferenceConfig,
    resolve_local_batch_size_or_raise,
)
from nandmachine.config.model_config import ModelConfigBase


@dataclass(frozen=True)
class _AttentionLayout:
    per_token_kv_values: int


_IMBALANCED_KV_CACHE_MONTE_CARLO_TRIALS = 100
_IMBALANCED_KV_CACHE_RANDOM_SEED = 0


def _ceil_div(numerator: int, denominator: int) -> int:
    if denominator <= 0:
        raise ValueError(f"denominator must be positive, got {denominator}")
    return (numerator + denominator - 1) // denominator


def _bits_to_bytes(value_count: int, bits_per_value: int) -> int:
    if value_count < 0:
        raise ValueError(f"value_count must be non-negative, got {value_count}")
    if bits_per_value <= 0:
        raise ValueError(f"bits_per_value must be positive, got {bits_per_value}")
    return _ceil_div(value_count * bits_per_value, 8)


def _simulate_max_bin_load_mean(
    num_balls: int,
    num_bins: int,
    num_trials: int,
    seed: int,
) -> int:
    if num_balls < 0:
        raise ValueError(f"num_balls must be non-negative, got {num_balls}")
    if num_bins <= 0:
        raise ValueError(f"num_bins must be positive, got {num_bins}")
    if num_trials <= 0:
        raise ValueError(f"num_trials must be positive, got {num_trials}")
    if num_balls == 0:
        return 0

    rng = Random(seed)
    total_max_load = 0

    for _ in range(num_trials):
        bin_loads = [0] * num_bins
        max_load = 0
        for _ in range(num_balls):
            bin_index = rng.randrange(num_bins)
            load = bin_loads[bin_index] + 1
            bin_loads[bin_index] = load
            if load > max_load:
                max_load = load
        total_max_load += max_load

    return _ceil_div(total_max_load, num_trials)


def _resolve_attention_layout(model_config: ModelConfigBase) -> _AttentionLayout:
    attention_type = model_config.attention_type.lower()

    if attention_type == "mha":
        head_dim = model_config.head_dim or (
            model_config.hidden_size // model_config.num_attention_heads
        )
        per_token_kv_values = model_config.num_attention_heads * head_dim * 2
    elif attention_type == "gqa":
        head_dim = model_config.head_dim or (
            model_config.hidden_size // model_config.num_attention_heads
        )
        per_token_kv_values = model_config.num_key_value_heads * head_dim * 2
    elif attention_type == "mla":
        if not hasattr(model_config, "kv_lora_rank") or not hasattr(
            model_config,
            "qk_rope_head_dim",
        ):
            raise NotImplementedError("MLA KV cache sizing is not implemented yet")
        per_token_kv_values = (
            model_config.kv_lora_rank + model_config.qk_rope_head_dim
        )
    else:
        raise ValueError(f"Unsupported attention_type: {model_config.attention_type}")

    return _AttentionLayout(per_token_kv_values=per_token_kv_values)


def _resolve_attention_layout_legacy(model_config: ModelConfigBase) -> _AttentionLayout:
    try:
        return _resolve_attention_layout(model_config)
    except NotImplementedError as exc:
        raise AssertionError(str(exc)) from exc
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc


def calculate_kv_cache_state(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> KVCacheState:
    if inference_config.batch_size < 0:
        raise ValueError(f"batch_size must be non-negative, got {inference_config.batch_size}")
    if inference_config.input_sequence_length < 0:
        raise ValueError(
            f"input_sequence_length must be non-negative, got {inference_config.input_sequence_length}"
        )
    if inference_config.output_sequence_length < 0:
        raise ValueError(
            f"output_sequence_length must be non-negative, got {inference_config.output_sequence_length}"
        )
    if inference_config.kv_cache_bits <= 0:
        raise ValueError(f"kv_cache_bits must be positive, got {inference_config.kv_cache_bits}")
    if inference_config.kv_block_size_bytes <= 0:
        raise ValueError(
            f"kv_block_size_bytes must be positive, got {inference_config.kv_block_size_bytes}"
        )

    resolve_local_batch_size_or_raise(inference_config)

    layout = _resolve_attention_layout_legacy(model_config)
    peak_sequence_length = (
        inference_config.input_sequence_length + inference_config.output_sequence_length
    )

    total_kv_values = (
        inference_config.batch_size
        * peak_sequence_length
        * layout.per_token_kv_values
    )
    total_bytes = _bits_to_bytes(total_kv_values, inference_config.kv_cache_bits)

    per_token_kv_values = layout.per_token_kv_values
    per_token_kv_bytes = _bits_to_bytes(per_token_kv_values, inference_config.kv_cache_bits)
    if per_token_kv_bytes <= 0:
        raise ValueError("per-token KV cache size must be positive")
    kv_block_size_tokens = _ceil_div(
        inference_config.kv_block_size_bytes,
        per_token_kv_bytes,
    )

    page_size_bytes = nand_config.page_size_bytes
    hyper_page_size_bytes = (
        nand_config.num_channels * nand_config.num_plane * page_size_bytes
    )
    num_nand_pages = _ceil_div(total_bytes, page_size_bytes) if total_bytes else 0
    num_hyper_pages = _ceil_div(total_bytes, hyper_page_size_bytes) if total_bytes else 0
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
    )


def build_kv_cache_state(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> KVCacheState:
    return calculate_kv_cache_state(nand_config, model_config, inference_config)


def build_imbalanced_kv_cache_state(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> KVCacheState:
    balanced_state = build_kv_cache_state(nand_config, model_config, inference_config)
    bins = (
        nand_config.num_plane
        * nand_config.num_channels
        * nand_config.page_size_bytes
        // inference_config.kv_block_size_bytes
    )
    

    balls = balanced_state.num_kv_blocks // (inference_config.parallel_config.num_ranks)

    num_hyper_pages = _simulate_max_bin_load_mean(
        num_balls=balls,
        num_bins=bins,
        num_trials=_IMBALANCED_KV_CACHE_MONTE_CARLO_TRIALS,
        seed=_IMBALANCED_KV_CACHE_RANDOM_SEED,
    )
    num_kv_blocks = num_hyper_pages * (nand_config.num_channels*nand_config.num_plane * nand_config.page_size_bytes) // inference_config.kv_block_size_bytes

    return KVCacheState(
        total_kv_cache_size_per_layer= num_hyper_pages * (nand_config.num_channels*nand_config.num_plane * nand_config.page_size_bytes),
        num_nand_pages_per_layer=num_hyper_pages * (nand_config.num_channels*nand_config.num_plane),
        num_hyper_pages_per_layer=num_hyper_pages,
        kv_block_size_tokens=balanced_state.kv_block_size_tokens,
        num_kv_blocks=num_kv_blocks,
        is_imbalance=True
    )


__all__ = [
    "build_kv_cache_state",
    "build_imbalanced_kv_cache_state",
    "calculate_kv_cache_state",
]
