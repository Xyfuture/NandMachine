from __future__ import annotations

from dataclasses import dataclass, replace

from nandmachine.config.cache_state import (
    BatchSizeCapacityResult,
    InsufficientGPUMemoryError,
)
from nandmachine.config.hbm_hbf_architecture import (
    build_device_for_hbm_hbf_architecture_or_raise,
)
from nandmachine.config.inference_config import (
    DenseParallelConfig,
    InferenceConfig,
    MoEParallelConfig,
    ParallelConfig,
    resolve_batch_partition_size_or_raise,
    resolve_local_batch_size_or_raise,
)
from nandmachine.config.model_config import (
    DeepseekV3ModelConfig,
    LlamaModelConfig,
    ModelConfigBase,
    Qwen3MoEModelConfig,
    Qwen3ModelConfig,
)


@dataclass(frozen=True)
class _AttentionLayout:
    per_token_kv_values: int


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


def _require_divisible(value: int, divisor: int, name: str) -> int:
    if divisor <= 0:
        raise ValueError(f"{name} divisor must be positive, got {divisor}")
    if value % divisor != 0:
        raise ValueError(f"{name}={value} must be divisible by {divisor}")
    return value // divisor


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


def _calculate_full_model_kv_cache_bytes(
    model_config: Qwen3ModelConfig | LlamaModelConfig,
    inference_config: InferenceConfig,
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

    layout = _resolve_attention_layout(model_config)
    peak_sequence_length = (
        inference_config.input_sequence_length + inference_config.output_sequence_length
    )

    per_layer_value_count = (
        inference_config.batch_size
        * peak_sequence_length
        * layout.per_token_kv_values
    )

    return _bits_to_bytes(
        per_layer_value_count * model_config.num_hidden_layers,
        inference_config.kv_cache_bits,
    )


def _calculate_qwen3_moe_full_model_kv_cache_bytes(
    model_config: Qwen3MoEModelConfig,
    inference_config: InferenceConfig,
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

    layout = _resolve_attention_layout(model_config)
    peak_sequence_length = (
        inference_config.input_sequence_length + inference_config.output_sequence_length
    )

    per_layer_value_count = (
        inference_config.batch_size
        * peak_sequence_length
        * layout.per_token_kv_values
    )

    return _bits_to_bytes(
        per_layer_value_count * model_config.num_hidden_layers,
        inference_config.kv_cache_bits,
    )


def _require_supported_deepseek_v3_capacity_model(
    model_config: ModelConfigBase,
) -> DeepseekV3ModelConfig:
    if not isinstance(model_config, DeepseekV3ModelConfig):
        raise NotImplementedError(
            "MoE GPU capacity calculator only supports DeepseekV3ModelConfig for MLA"
        )
    if model_config.attention_type.lower() != "mla":
        raise ValueError(
            "DeepseekV3 GPU capacity calculator requires attention_type == 'mla'"
        )
    if model_config.hidden_act != "silu":
        raise ValueError(f"Unsupported hidden_act: {model_config.hidden_act}")
    if model_config.num_nextn_predict_layers != 1:
        raise ValueError(
            "DeepseekV3 GPU capacity calculator only supports num_nextn_predict_layers == 1"
        )
    return model_config


def _calculate_per_rank_deepseek_v3_weight_bytes(
    model_config: DeepseekV3ModelConfig,
    inference_config: InferenceConfig,
    parallelism: _MoEParallelism,
) -> int:
    local_num_heads = _require_divisible(
        model_config.num_attention_heads,
        parallelism.tp_size,
        "num_attention_heads",
    )
    local_routed_expert_count = _require_divisible(
        model_config.n_routed_experts,
        parallelism.ffn_ep_size,
        "n_routed_experts",
    )
    local_moe_intermediate_size = _require_divisible(
        model_config.moe_intermediate_size,
        parallelism.ffn_tp_size,
        "moe_intermediate_size",
    )
    attention_param_count_per_layer = (
        model_config.hidden_size * model_config.q_lora_rank
        + model_config.q_lora_rank
        + model_config.q_lora_rank
        * (
            local_num_heads
            * (model_config.qk_nope_head_dim + model_config.qk_rope_head_dim)
        )
        + model_config.hidden_size
        * (model_config.kv_lora_rank + model_config.qk_rope_head_dim)
        + model_config.kv_lora_rank
        + local_num_heads
        * model_config.qk_nope_head_dim
        * model_config.kv_lora_rank
        + local_num_heads
        * model_config.kv_lora_rank
        * model_config.v_head_dim
        + model_config.hidden_size
        * (local_num_heads * model_config.v_head_dim)
        + model_config.hidden_size * 2
    )

    moe_router_param_count_per_layer = (
        model_config.hidden_size * model_config.n_routed_experts
    )
    moe_expert_param_count_per_layer = local_routed_expert_count * (
        model_config.hidden_size * (local_moe_intermediate_size * 2)
        + model_config.hidden_size * local_moe_intermediate_size
    )

    total_param_count = (
        attention_param_count_per_layer * model_config.num_hidden_layers
        + (moe_router_param_count_per_layer + moe_expert_param_count_per_layer)
        * model_config.num_hidden_layers
    )

    return _bits_to_bytes(total_param_count, inference_config.weight_bits)


def _calculate_deepseek_v3_full_model_kv_cache_bytes(
    model_config: DeepseekV3ModelConfig,
    inference_config: InferenceConfig,
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

    layout = _resolve_attention_layout(model_config)
    peak_sequence_length = (
        inference_config.input_sequence_length + inference_config.output_sequence_length
    )
    per_layer_value_count = (
        inference_config.batch_size
        * peak_sequence_length
        * layout.per_token_kv_values
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
    global_kv_cache_bytes: int,
) -> BatchSizeCapacityResult:
    device = _build_capacity_device_or_raise(device_name, memory_architecture)
    per_rank_capacity_bytes = device.total_memory_capacity_bytes
    total_capacity_bytes = per_rank_capacity_bytes * parallelism.num_ranks
    total_weight_bytes = per_rank_weight_bytes * parallelism.num_ranks
    total_kv_cache_bytes = global_kv_cache_bytes
    total_used_bytes = total_weight_bytes + total_kv_cache_bytes
    per_rank_kv_cache_bytes = _ceil_div(total_kv_cache_bytes, parallelism.num_ranks)
    per_rank_used_bytes = per_rank_weight_bytes + per_rank_kv_cache_bytes
    per_rank_remaining_bytes = per_rank_capacity_bytes - per_rank_used_bytes

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


def validate_batch_size_or_raise(
    device_name: str,
    memory_architecture: object,
    model_config: ModelConfigBase,
    inference_config: InferenceConfig,
) -> BatchSizeCapacityResult:
    if inference_config.batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {inference_config.batch_size}")
    resolve_local_batch_size_or_raise(inference_config)

    if isinstance(model_config, (Qwen3ModelConfig, LlamaModelConfig)):
        dense_model_config = _require_dense_model_config(model_config)
        parallelism = _resolve_dense_parallelism(inference_config.parallel_config)
        per_rank_weight_bytes = _calculate_per_rank_weight_bytes(
            dense_model_config,
            inference_config,
            parallelism,
        )
        global_kv_cache_bytes = _calculate_full_model_kv_cache_bytes(
            dense_model_config,
            inference_config,
        )
    elif isinstance(model_config, Qwen3MoEModelConfig):
        moe_model_config = _require_supported_qwen3_moe_capacity_model(model_config)
        parallelism = _resolve_moe_parallelism(inference_config.parallel_config)
        per_rank_weight_bytes = _calculate_per_rank_qwen3_moe_weight_bytes(
            moe_model_config,
            inference_config,
            parallelism,
        )
        global_kv_cache_bytes = _calculate_qwen3_moe_full_model_kv_cache_bytes(
            moe_model_config,
            inference_config,
        )
    elif isinstance(model_config, DeepseekV3ModelConfig):
        deepseek_model_config = _require_supported_deepseek_v3_capacity_model(model_config)
        parallelism = _resolve_moe_parallelism(inference_config.parallel_config)
        per_rank_weight_bytes = _calculate_per_rank_deepseek_v3_weight_bytes(
            deepseek_model_config,
            inference_config,
            parallelism,
        )
        global_kv_cache_bytes = _calculate_deepseek_v3_full_model_kv_cache_bytes(
            deepseek_model_config,
            inference_config,
        )
    else:
        raise NotImplementedError(
            "GPU capacity calculator only supports dense Qwen3/Llama, Qwen3MoE, and DeepseekV3"
        )

    result = _build_capacity_result(
        device_name,
        memory_architecture,
        inference_config,
        parallelism,
        per_rank_weight_bytes,
        global_kv_cache_bytes,
    )

    if result.total_used_bytes > result.total_capacity_bytes:
        raise InsufficientGPUMemoryError(
            "Insufficient GPU memory for requested batch size: "
            f"device_name={device_name}, batch_size={inference_config.batch_size}, "
            f"total_used_bytes={result.total_used_bytes}, "
            f"total_capacity_bytes={result.total_capacity_bytes}"
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
    batch_step = resolve_batch_partition_size_or_raise(inference_config.parallel_config)

    def _validate_candidate(batch_size: int) -> BatchSizeCapacityResult:
        candidate_config = replace(inference_config, batch_size=batch_size)
        return validate_batch_size_or_raise(
            device_name=device_name,
            memory_architecture=memory_architecture,
            model_config=model_config,
            inference_config=candidate_config,
        )

    try:
        best_result = _validate_candidate(batch_step)
    except InsufficientGPUMemoryError as exc:
        raise InsufficientGPUMemoryError(
            f"Batch size {batch_step} does not fit on device_name={device_name}"
        ) from exc

    lower_multiplier = 1
    upper_multiplier = 1

    while True:
        candidate_multiplier = upper_multiplier * 2
        candidate_batch_size = candidate_multiplier * batch_step
        try:
            best_result = _validate_candidate(candidate_batch_size)
        except InsufficientGPUMemoryError:
            break
        lower_multiplier = candidate_multiplier
        upper_multiplier = candidate_multiplier

    failed_upper_multiplier = upper_multiplier * 2
    left = lower_multiplier + 1
    right = failed_upper_multiplier - 1

    while left <= right:
        mid = (left + right) // 2
        candidate_batch_size = mid * batch_step
        try:
            candidate_result = _validate_candidate(candidate_batch_size)
        except InsufficientGPUMemoryError:
            right = mid - 1
            continue

        best_result = candidate_result
        left = mid + 1

    return best_result


__all__ = [
    "validate_batch_size_or_raise",
    "calculate_max_batch_size",
]
