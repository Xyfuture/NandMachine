from __future__ import annotations

from dataclasses import dataclass, replace

from nandmachine.config.cache_state import (
    BatchSizeCapacityResult,
    InsufficientGPUMemoryError,
    KVCacheState,
)
from nandmachine.config.config import NandConfig
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
)
from nandmachine.config.inference_config import (
    DenseParallelConfig,
    InferenceConfig,
    MoEParallelConfig,
    ParallelConfig,
)
from nandmachine.config.model_config import (
    LlamaModelConfig,
    ModelConfigBase,
    Qwen3MoEModelConfig,
    Qwen3ModelConfig,
)


@dataclass(frozen=True)
class _AttentionLayout:
    num_kv_heads: int
    head_dim: int


@dataclass(frozen=True)
class _DenseParallelism:
    num_ranks: int
    dp_size: int
    tp_size: int


@dataclass(frozen=True)
class _DenseModelWeightSpec:
    qkv_output_size: int
    o_proj_input_size: int
    gate_up_output_size: int
    down_proj_input_size: int
    replicated_norm_params_per_layer: int


@dataclass(frozen=True)
class _MoEParallelism:
    num_ranks: int
    dp_size: int
    tp_size: int
    ffn_ep_size: int
    ffn_tp_size: int


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

    if attention_type == "mha":
        num_kv_heads = model_config.num_attention_heads
    elif attention_type == "gqa":
        num_kv_heads = model_config.num_key_value_heads
    elif attention_type == "mla":
        raise NotImplementedError("MLA KV cache sizing is not implemented yet")
    else:
        raise ValueError(f"Unsupported attention_type: {model_config.attention_type}")

    num_kv_heads = _require_divisible(num_kv_heads, tp_size, "num_kv_heads")

    return _AttentionLayout(num_kv_heads=num_kv_heads, head_dim=head_dim)


def _resolve_attention_layout_legacy(
    model_config: ModelConfigBase,
    parallel_config: ParallelConfig | None,
) -> _AttentionLayout:
    try:
        return _resolve_attention_layout(model_config, parallel_config)
    except NotImplementedError as exc:
        raise AssertionError(str(exc)) from exc
    except ValueError as exc:
        raise AssertionError(str(exc)) from exc


def _resolve_dense_parallelism(parallel_config: ParallelConfig | None) -> _DenseParallelism:
    if parallel_config is None:
        return _DenseParallelism(num_ranks=1, dp_size=1, tp_size=1)

    if isinstance(parallel_config, MoEParallelConfig):
        raise NotImplementedError(
            "MoEParallelConfig is not supported by dense GPU capacity calculator"
        )

    if isinstance(parallel_config, DenseParallelConfig):
        if parallel_config.num_ranks <= 0:
            raise ValueError(f"num_ranks must be positive, got {parallel_config.num_ranks}")
        if parallel_config.dp_size <= 0:
            raise ValueError(f"dp_size must be positive, got {parallel_config.dp_size}")
        if parallel_config.tp_size <= 0:
            raise ValueError(f"tp_size must be positive, got {parallel_config.tp_size}")
        if parallel_config.num_ranks != parallel_config.dp_size * parallel_config.tp_size:
            raise ValueError(
                "Dense parallel config must satisfy num_ranks == dp_size * tp_size, "
                f"got num_ranks={parallel_config.num_ranks}, "
                f"dp_size={parallel_config.dp_size}, tp_size={parallel_config.tp_size}"
            )
        return _DenseParallelism(
            num_ranks=parallel_config.num_ranks,
            dp_size=parallel_config.dp_size,
            tp_size=parallel_config.tp_size,
        )

    if not isinstance(parallel_config, ParallelConfig):
        raise TypeError(f"Unsupported parallel_config type: {type(parallel_config).__name__}")
    if parallel_config.num_ranks <= 0:
        raise ValueError(f"num_ranks must be positive, got {parallel_config.num_ranks}")

    return _DenseParallelism(
        num_ranks=parallel_config.num_ranks,
        dp_size=parallel_config.num_ranks,
        tp_size=1,
    )


def _require_dense_model_config(model_config: ModelConfigBase) -> Qwen3ModelConfig | LlamaModelConfig:
    if isinstance(model_config, (Qwen3ModelConfig, LlamaModelConfig)):
        return model_config
    raise NotImplementedError(
        "Dense GPU capacity calculator only supports Qwen3ModelConfig and LlamaModelConfig"
    )


def _resolve_moe_parallelism(parallel_config: ParallelConfig | None) -> _MoEParallelism:
    if not isinstance(parallel_config, MoEParallelConfig):
        raise ValueError("Qwen3MoE GPU capacity calculator requires MoEParallelConfig")

    if parallel_config.num_ranks <= 0:
        raise ValueError(f"num_ranks must be positive, got {parallel_config.num_ranks}")
    if parallel_config.attn_dp_size <= 0:
        raise ValueError(f"attn_dp_size must be positive, got {parallel_config.attn_dp_size}")
    if parallel_config.attn_tp_size <= 0:
        raise ValueError(f"attn_tp_size must be positive, got {parallel_config.attn_tp_size}")
    if parallel_config.ffn_ep_size <= 0:
        raise ValueError(f"ffn_ep_size must be positive, got {parallel_config.ffn_ep_size}")
    if parallel_config.ffn_tp_size <= 0:
        raise ValueError(f"ffn_tp_size must be positive, got {parallel_config.ffn_tp_size}")

    attn_world_size = parallel_config.attn_dp_size * parallel_config.attn_tp_size
    ffn_world_size = parallel_config.ffn_ep_size * parallel_config.ffn_tp_size
    if parallel_config.num_ranks != attn_world_size or parallel_config.num_ranks != ffn_world_size:
        raise ValueError(
            "MoE parallel config must satisfy "
            "num_ranks == attn_dp_size * attn_tp_size == ffn_ep_size * ffn_tp_size, "
            f"got num_ranks={parallel_config.num_ranks}, "
            f"attn_dp_size={parallel_config.attn_dp_size}, "
            f"attn_tp_size={parallel_config.attn_tp_size}, "
            f"ffn_ep_size={parallel_config.ffn_ep_size}, "
            f"ffn_tp_size={parallel_config.ffn_tp_size}"
        )

    return _MoEParallelism(
        num_ranks=parallel_config.num_ranks,
        dp_size=parallel_config.attn_dp_size,
        tp_size=parallel_config.attn_tp_size,
        ffn_ep_size=parallel_config.ffn_ep_size,
        ffn_tp_size=parallel_config.ffn_tp_size,
    )


def _require_supported_qwen3_moe_capacity_model(
    model_config: ModelConfigBase,
) -> Qwen3MoEModelConfig:
    if not isinstance(model_config, Qwen3MoEModelConfig):
        raise NotImplementedError(
            "MoE GPU capacity calculator only supports Qwen3MoEModelConfig"
        )
    if model_config.ffn_type != "moe":
        raise ValueError(f"Unsupported ffn_type for Qwen3MoE: {model_config.ffn_type}")
    if model_config.attention_type.lower() != "gqa":
        raise NotImplementedError(
            "Qwen3MoE GPU capacity calculator only supports attention_type == 'gqa'"
        )
    if model_config.shared_expert_intermediate_size is not None:
        raise NotImplementedError("shared expert is not implemented")
    if model_config.decoder_sparse_step != 1:
        raise NotImplementedError(
            "Qwen3MoE GPU capacity calculator only supports decoder_sparse_step == 1"
        )
    if model_config.mlp_only_layers:
        raise NotImplementedError(
            "Qwen3MoE GPU capacity calculator only supports empty mlp_only_layers"
        )
    return model_config


def _resolve_weight_spec(
    model_config: Qwen3ModelConfig | LlamaModelConfig,
) -> _DenseModelWeightSpec:
    if model_config.num_hidden_layers is None:
        raise ValueError("model_config.num_hidden_layers must be set")

    attention_type = model_config.attention_type.lower()
    if attention_type == "mha":
        kv_heads_for_qkv = model_config.num_attention_heads
    elif attention_type == "gqa":
        kv_heads_for_qkv = model_config.num_key_value_heads
    elif attention_type == "mla":
        raise NotImplementedError("MLA weight sizing is not implemented yet")
    else:
        raise ValueError(f"Unsupported attention_type: {model_config.attention_type}")

    head_dim = model_config.head_dim or (
        model_config.hidden_size // model_config.num_attention_heads
    )
    qkv_output_size = (
        model_config.num_attention_heads + 2 * kv_heads_for_qkv
    ) * head_dim
    o_proj_input_size = model_config.num_attention_heads * head_dim
    gate_up_output_size = model_config.intermediate_size * 2
    down_proj_input_size = model_config.intermediate_size

    replicated_norm_params_per_layer = model_config.hidden_size * 2
    if isinstance(model_config, Qwen3ModelConfig) and not model_config.attention_bias:
        replicated_norm_params_per_layer += head_dim * 2

    return _DenseModelWeightSpec(
        qkv_output_size=qkv_output_size,
        o_proj_input_size=o_proj_input_size,
        gate_up_output_size=gate_up_output_size,
        down_proj_input_size=down_proj_input_size,
        replicated_norm_params_per_layer=replicated_norm_params_per_layer,
    )


def _require_divisible(value: int, divisor: int, name: str) -> int:
    if divisor <= 0:
        raise ValueError(f"{name} divisor must be positive, got {divisor}")
    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by {divisor}")
    return value // divisor


def _calculate_per_rank_weight_bytes(
    model_config: Qwen3ModelConfig | LlamaModelConfig,
    inference_config: InferenceConfig,
    parallelism: _DenseParallelism,
) -> int:
    spec = _resolve_weight_spec(model_config)
    tp_size = parallelism.tp_size

    qkv_output_per_rank = _require_divisible(
        spec.qkv_output_size,
        tp_size,
        "qkv_output_size",
    )
    o_proj_input_per_rank = _require_divisible(
        spec.o_proj_input_size,
        tp_size,
        "o_proj_input_size",
    )
    gate_up_output_per_rank = _require_divisible(
        spec.gate_up_output_size,
        tp_size,
        "gate_up_output_size",
    )
    down_proj_input_per_rank = _require_divisible(
        spec.down_proj_input_size,
        tp_size,
        "down_proj_input_size",
    )

    weight_param_count_per_layer = (
        model_config.hidden_size * qkv_output_per_rank
        + model_config.hidden_size * o_proj_input_per_rank
        + model_config.hidden_size * gate_up_output_per_rank
        + model_config.hidden_size * down_proj_input_per_rank
        + spec.replicated_norm_params_per_layer
    )

    return _bits_to_bytes(
        weight_param_count_per_layer * model_config.num_hidden_layers,
        inference_config.weight_bits,
    )


def _calculate_per_rank_qwen3_moe_weight_bytes(
    model_config: Qwen3MoEModelConfig,
    inference_config: InferenceConfig,
    parallelism: _MoEParallelism,
) -> int:
    _require_divisible(
        model_config.num_attention_heads,
        parallelism.tp_size,
        "num_attention_heads",
    )
    _require_divisible(
        model_config.num_key_value_heads,
        parallelism.tp_size,
        "num_key_value_heads",
    )
    head_dim = model_config.head_dim or (
        model_config.hidden_size // model_config.num_attention_heads
    )
    attention_qkv_output_size = (
        model_config.num_attention_heads + 2 * model_config.num_key_value_heads
    ) * head_dim
    attention_o_proj_input_size = model_config.num_attention_heads * head_dim

    attention_qkv_output_per_rank = _require_divisible(
        attention_qkv_output_size,
        parallelism.tp_size,
        "attention_qkv_output_size",
    )
    attention_o_proj_input_per_rank = _require_divisible(
        attention_o_proj_input_size,
        parallelism.tp_size,
        "attention_o_proj_input_size",
    )
    local_expert_count = _require_divisible(
        model_config.num_experts,
        parallelism.ffn_ep_size,
        "num_experts",
    )
    local_moe_intermediate_size = _require_divisible(
        model_config.moe_intermediate_size,
        parallelism.ffn_tp_size,
        "moe_intermediate_size",
    )

    replicated_norm_params_per_layer = model_config.hidden_size * 2
    if not model_config.attention_bias:
        replicated_norm_params_per_layer += head_dim * 2

    router_param_count_per_layer = model_config.hidden_size * model_config.num_experts
    per_local_expert_param_count = (
        model_config.hidden_size * (local_moe_intermediate_size * 2)
        + model_config.hidden_size * local_moe_intermediate_size
    )

    weight_param_count_per_layer = (
        model_config.hidden_size * attention_qkv_output_per_rank
        + model_config.hidden_size * attention_o_proj_input_per_rank
        + replicated_norm_params_per_layer
        + router_param_count_per_layer
        + local_expert_count * per_local_expert_param_count
    )

    return _bits_to_bytes(
        weight_param_count_per_layer * model_config.num_hidden_layers,
        inference_config.weight_bits,
    )


def _calculate_per_rank_full_model_kv_cache_bytes(
    model_config: Qwen3ModelConfig | LlamaModelConfig,
    inference_config: InferenceConfig,
    parallelism: _DenseParallelism,
) -> int:
    if model_config.num_hidden_layers is None:
        raise ValueError("model_config.num_hidden_layers must be set")
    if inference_config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {inference_config.batch_size}")
    if inference_config.input_sequence_length < 0:
        raise ValueError(
            f"input_sequence_length must be non-negative, got {inference_config.input_sequence_length}"
        )
    if inference_config.output_sequence_length < 0:
        raise ValueError(
            f"output_sequence_length must be non-negative, got {inference_config.output_sequence_length}"
        )

    layout = _resolve_attention_layout(
        model_config,
        inference_config.parallel_config,
    )
    rank_batch = _ceil_div(inference_config.batch_size, parallelism.dp_size)
    peak_sequence_length = (
        inference_config.input_sequence_length + inference_config.output_sequence_length
    )

    per_layer_value_count = (
        rank_batch
        * peak_sequence_length
        * layout.num_kv_heads
        * layout.head_dim
        * 2
    )

    return _bits_to_bytes(
        per_layer_value_count * model_config.num_hidden_layers,
        inference_config.kv_cache_bits,
    )


def _calculate_per_rank_qwen3_moe_full_model_kv_cache_bytes(
    model_config: Qwen3MoEModelConfig,
    inference_config: InferenceConfig,
    parallelism: _MoEParallelism,
) -> int:
    if inference_config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {inference_config.batch_size}")
    if inference_config.input_sequence_length < 0:
        raise ValueError(
            f"input_sequence_length must be non-negative, got {inference_config.input_sequence_length}"
        )
    if inference_config.output_sequence_length < 0:
        raise ValueError(
            f"output_sequence_length must be non-negative, got {inference_config.output_sequence_length}"
        )

    layout = _resolve_attention_layout(
        model_config,
        inference_config.parallel_config,
    )
    rank_batch = _ceil_div(inference_config.batch_size, parallelism.dp_size)
    peak_sequence_length = (
        inference_config.input_sequence_length + inference_config.output_sequence_length
    )

    per_layer_value_count = (
        rank_batch
        * peak_sequence_length
        * layout.num_kv_heads
        * layout.head_dim
        * 2
    )

    return _bits_to_bytes(
        per_layer_value_count * model_config.num_hidden_layers,
        inference_config.kv_cache_bits,
    )


def _build_capacity_device_or_raise(device_name: str, memory_architecture: object):
    return build_device_for_hbm_hbf_architecture_or_raise(
        device_name,
        memory_architecture,
    )


def _build_capacity_result(
    device_name: str,
    memory_architecture: object,
    inference_config: InferenceConfig,
    parallelism: _DenseParallelism | _MoEParallelism,
    per_rank_weight_bytes: int,
    per_rank_kv_cache_bytes: int,
) -> BatchSizeCapacityResult:
    device = _build_capacity_device_or_raise(device_name, memory_architecture)
    per_rank_capacity_bytes = device.total_memory_capacity_bytes
    per_rank_used_bytes = per_rank_weight_bytes + per_rank_kv_cache_bytes
    per_rank_remaining_bytes = per_rank_capacity_bytes - per_rank_used_bytes
    total_capacity_bytes = per_rank_capacity_bytes * parallelism.num_ranks
    total_weight_bytes = per_rank_weight_bytes * parallelism.num_ranks
    total_kv_cache_bytes = per_rank_kv_cache_bytes * parallelism.num_ranks
    total_used_bytes = per_rank_used_bytes * parallelism.num_ranks

    return BatchSizeCapacityResult(
        device_name=device_name,
        batch_size=inference_config.batch_size,
        num_ranks=parallelism.num_ranks,
        dp_size=parallelism.dp_size,
        tp_size=parallelism.tp_size,
        ffn_ep_size=getattr(parallelism, "ffn_ep_size", None),
        ffn_tp_size=getattr(parallelism, "ffn_tp_size", None),
        per_rank_capacity_bytes=per_rank_capacity_bytes,
        per_rank_weight_bytes=per_rank_weight_bytes,
        per_rank_kv_cache_bytes=per_rank_kv_cache_bytes,
        per_rank_used_bytes=per_rank_used_bytes,
        per_rank_remaining_bytes=per_rank_remaining_bytes,
        total_capacity_bytes=total_capacity_bytes,
        total_weight_bytes=total_weight_bytes,
        total_kv_cache_bytes=total_kv_cache_bytes,
        total_used_bytes=total_used_bytes,
    )


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

    layout = _resolve_attention_layout_legacy(
        model_config,
        inference_config.parallel_config,
    )
    dp_size = _resolve_dp_size(inference_config.parallel_config)
    if dp_size <= 0:
        raise ValueError(f"dp_size must be positive, got {dp_size}")
    rank_batch = _ceil_div(inference_config.batch_size, dp_size)
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
    total_bytes = _bits_to_bytes(total_kv_values, inference_config.kv_cache_bits)

    per_token_kv_values = layout.num_kv_heads * layout.head_dim * 2
    per_token_kv_bytes = _bits_to_bytes(per_token_kv_values, inference_config.kv_cache_bits)
    if per_token_kv_bytes <= 0:
        raise ValueError("per-token KV cache size must be positive")
    kv_block_size_tokens = _ceil_div(
        inference_config.kv_block_size_bytes,
        per_token_kv_bytes,
    )

    page_size_bytes = nand_config.page_size_bytes
    hyper_page_size_bytes = nand_config.num_plane * page_size_bytes
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
        kv_cache_num_pages_per_layer=num_nand_pages,
    )


def build_kv_cache_state(
    nand_config: NandConfig,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> KVCacheState:
    return calculate_kv_cache_state(nand_config, model_config, inference_config)


def validate_batch_size_or_raise(
    device_name: str,
    memory_architecture: object,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> BatchSizeCapacityResult:
    if inference_config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {inference_config.batch_size}")

    if isinstance(model_config, (Qwen3ModelConfig, LlamaModelConfig)):
        dense_model_config = _require_dense_model_config(model_config)
        parallelism = _resolve_dense_parallelism(inference_config.parallel_config)
        per_rank_weight_bytes = _calculate_per_rank_weight_bytes(
            dense_model_config,
            inference_config,
            parallelism,
        )
        per_rank_kv_cache_bytes = _calculate_per_rank_full_model_kv_cache_bytes(
            dense_model_config,
            inference_config,
            parallelism,
        )
    elif isinstance(model_config, Qwen3MoEModelConfig):
        moe_model_config = _require_supported_qwen3_moe_capacity_model(model_config)
        parallelism = _resolve_moe_parallelism(inference_config.parallel_config)
        per_rank_weight_bytes = _calculate_per_rank_qwen3_moe_weight_bytes(
            moe_model_config,
            inference_config,
            parallelism,
        )
        per_rank_kv_cache_bytes = _calculate_per_rank_qwen3_moe_full_model_kv_cache_bytes(
            moe_model_config,
            inference_config,
            parallelism,
        )
    else:
        raise NotImplementedError(
            "GPU capacity calculator only supports dense Qwen3/Llama and Qwen3MoE"
        )

    result = _build_capacity_result(
        device_name,
        memory_architecture,
        inference_config,
        parallelism,
        per_rank_weight_bytes,
        per_rank_kv_cache_bytes,
    )

    if result.per_rank_used_bytes > result.per_rank_capacity_bytes:
        raise InsufficientGPUMemoryError(
            "Insufficient GPU memory for requested batch size: "
            f"device_name={device_name}, batch_size={inference_config.batch_size}, "
            f"per_rank_used_bytes={result.per_rank_used_bytes}, "
            f"per_rank_capacity_bytes={result.per_rank_capacity_bytes}"
        )

    return result


def calculate_max_batch_size(
    device_name: str,
    memory_architecture: object,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> BatchSizeCapacityResult:
    if (
        inference_config.input_sequence_length + inference_config.output_sequence_length
        <= 0
    ):
        raise ValueError(
            "Cannot calculate finite maximum batch size when peak sequence length is zero"
        )

    def _validate_candidate(batch_size: int) -> BatchSizeCapacityResult:
        candidate_config = replace(inference_config, batch_size=batch_size)
        return validate_batch_size_or_raise(
            device_name=device_name,
            memory_architecture=memory_architecture,
            model_config=model_config,
            inference_config=candidate_config,
        )

    try:
        best_result = _validate_candidate(1)
    except InsufficientGPUMemoryError as exc:
        raise InsufficientGPUMemoryError(
            f"Batch size 1 does not fit on device_name={device_name}"
        ) from exc

    lower_bound = 1
    upper_bound = 1

    while True:
        candidate_batch_size = upper_bound * 2
        try:
            best_result = _validate_candidate(candidate_batch_size)
        except InsufficientGPUMemoryError:
            break
        lower_bound = candidate_batch_size
        upper_bound = candidate_batch_size

    failed_upper_bound = upper_bound * 2
    left = lower_bound + 1
    right = failed_upper_bound - 1

    while left <= right:
        mid = (left + right) // 2
        try:
            candidate_result = _validate_candidate(mid)
        except InsufficientGPUMemoryError:
            right = mid - 1
            continue

        best_result = candidate_result
        left = mid + 1

    return best_result


__all__ = [
    "build_kv_cache_state",
    "calculate_kv_cache_state",
    "validate_batch_size_or_raise",
    "calculate_max_batch_size",
]
