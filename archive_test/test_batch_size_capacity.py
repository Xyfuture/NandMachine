import json
from pathlib import Path

import pytest

from nandmachine.config.cache_state import InsufficientGPUMemoryError
from nandmachine.config.inference_config import (
    DenseParallelConfig,
    InferenceConfig,
    MoEParallelConfig,
    ParallelConfig,
)
from nandmachine.config.model_config import (
    LlamaModelConfig,
    Qwen3MoEModelConfig,
    Qwen3ModelConfig,
)
from nandmachine.frontend.utlis import calculate_max_batch_size, validate_batch_size_or_raise


MODEL_CARD_DIR = Path(__file__).resolve().parents[1] / "model_cards"


def _make_inference_config(
    batch_size: int,
    *,
    input_sequence_length: int = 8,
    output_sequence_length: int = 4,
    weight_bits: int = 16,
    kv_cache_bits: int = 16,
    parallel_config: ParallelConfig,
) -> InferenceConfig:
    return InferenceConfig(
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        weight_bits=weight_bits,
        activation_bits=16,
        kv_cache_bits=kv_cache_bits,
        kv_block_size_bytes=64 * 1024,
        memory_backend="nand",
        parallel_config=parallel_config,
    )


def _a100_capacity_bytes() -> int:
    return 80 * 1024**3


def _make_qwen3_moe_model_config(**overrides) -> Qwen3MoEModelConfig:
    config = Qwen3MoEModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        moe_intermediate_size=256,
        num_experts=4,
        num_experts_per_tok=2,
        num_hidden_layers=2,
        decoder_sparse_step=1,
        mlp_only_layers=[],
        hidden_act="silu",
        ffn_type="moe",
        head_dim=128,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10000.0,
        shared_expert_intermediate_size=None,
    )
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


def _load_qwen3_8b_model_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig.from_dict(
        json.loads((MODEL_CARD_DIR / "qwen3-8B.json").read_text())
    )


def _load_qwen3_moe_235b_model_config() -> Qwen3MoEModelConfig:
    return Qwen3MoEModelConfig.from_config(
        type(
            "Qwen3Moe235BConfig",
            (),
            json.loads((MODEL_CARD_DIR / "qwen3-moe-235B.json").read_text()),
        )()
    )


def test_validate_batch_size_qwen3_single_rank_counts_qk_norm_weights():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
        attention_bias=False,
    )
    inference_config = _make_inference_config(
        4,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        1024 * ((8 + 2 * 2) * 128)
        + 1024 * (8 * 128)
        + 1024 * (2048 * 2)
        + 1024 * 2048
        + (2 * 1024 + 2 * 128)
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 2 * 2
    per_layer_kv_values = 4 * (8 + 4) * 2 * 128 * 2
    expected_per_rank_kv_bytes = per_layer_kv_values * 2 * 2

    assert result.device_name == "A100_80GB"
    assert result.batch_size == 4
    assert result.dp_size == 1
    assert result.tp_size == 1
    assert result.per_rank_capacity_bytes == _a100_capacity_bytes()
    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes
    assert result.per_rank_kv_cache_bytes == expected_per_rank_kv_bytes
    assert result.per_rank_used_bytes == (
        expected_per_rank_weight_bytes + expected_per_rank_kv_bytes
    )
    assert result.per_rank_remaining_bytes == (
        _a100_capacity_bytes() - result.per_rank_used_bytes
    )
    assert result.total_weight_bytes == result.per_rank_weight_bytes
    assert result.total_kv_cache_bytes == result.per_rank_kv_cache_bytes
    assert result.total_used_bytes == result.per_rank_used_bytes


def test_validate_batch_size_llama_single_rank_excludes_qwen_qk_norm_weights():
    model_config = LlamaModelConfig(
        hidden_size=1536,
        num_attention_heads=12,
        num_key_value_heads=3,
        max_position_embeddings=4096,
        intermediate_size=3072,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=3,
        attention_bias=False,
        mlp_bias=False,
    )
    inference_config = _make_inference_config(
        2,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        1536 * ((12 + 2 * 3) * 128)
        + 1536 * (12 * 128)
        + 1536 * (3072 * 2)
        + 1536 * 3072
        + (2 * 1536)
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 3 * 2

    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes


def test_validate_batch_size_mha_uses_attention_heads_for_qkv_weights():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
        attention_bias=False,
        attention_type="mha",
    )
    inference_config = _make_inference_config(
        1,
        input_sequence_length=1,
        output_sequence_length=1,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        1024 * ((8 + 2 * 8) * 128)
        + 1024 * (8 * 128)
        + 1024 * (2048 * 2)
        + 1024 * 2048
        + (2 * 1024 + 2 * 128)
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 2 * 2

    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes


def test_validate_batch_size_dense_parallel_splits_weight_and_kv_per_rank():
    model_config = LlamaModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
    )
    inference_config = _make_inference_config(
        10,
        parallel_config=DenseParallelConfig(num_ranks=4, dp_size=2, tp_size=2),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        1024 * (((8 + 2 * 2) * 128) // 2)
        + 1024 * ((8 * 128) // 2)
        + 1024 * ((2048 * 2) // 2)
        + 1024 * (2048 // 2)
        + (2 * 1024)
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 2 * 2
    expected_rank_batch = (10 + 2 - 1) // 2
    expected_local_kv_heads = 2 // 2
    per_layer_kv_values = (
        expected_rank_batch * (8 + 4) * expected_local_kv_heads * 128 * 2
    )
    expected_per_rank_kv_bytes = per_layer_kv_values * 2 * 2

    assert result.num_ranks == 4
    assert result.dp_size == 2
    assert result.tp_size == 2
    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes
    assert result.per_rank_kv_cache_bytes == expected_per_rank_kv_bytes
    assert result.total_weight_bytes == expected_per_rank_weight_bytes * 4
    assert result.total_kv_cache_bytes == expected_per_rank_kv_bytes * 4


def test_validate_batch_size_parallel_config_is_pure_dp():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
    )
    inference_config = _make_inference_config(
        10,
        parallel_config=ParallelConfig(num_ranks=4),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)

    expected_rank_batch = (10 + 4 - 1) // 4
    per_layer_kv_values = expected_rank_batch * (8 + 4) * 2 * 128 * 2
    expected_per_rank_kv_bytes = per_layer_kv_values * 2 * 2

    assert result.dp_size == 4
    assert result.tp_size == 1
    assert result.per_rank_kv_cache_bytes == expected_per_rank_kv_bytes


def test_validate_batch_size_raises_for_insufficient_gpu_memory():
    model_config = LlamaModelConfig(
        hidden_size=16384,
        num_attention_heads=128,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        intermediate_size=53248,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=126,
    )
    inference_config = _make_inference_config(
        1,
        input_sequence_length=1,
        output_sequence_length=1,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    with pytest.raises(InsufficientGPUMemoryError):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_calculate_max_batch_size_returns_exact_dense_limit():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=36,
        attention_bias=False,
    )
    inference_config = _make_inference_config(
        999,
        input_sequence_length=2048,
        output_sequence_length=512,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    result = calculate_max_batch_size("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        4096 * ((32 + 2 * 8) * 128)
        + 4096 * (32 * 128)
        + 4096 * (12288 * 2)
        + 4096 * 12288
        + (2 * 4096 + 2 * 128)
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 36 * 2
    per_batch_per_layer_kv_values = (2048 + 512) * 8 * 128 * 2
    per_batch_full_model_kv_bytes = per_batch_per_layer_kv_values * 36 * 2
    expected_max_batch_size = (
        _a100_capacity_bytes() - expected_per_rank_weight_bytes
    ) // per_batch_full_model_kv_bytes

    assert result.batch_size == expected_max_batch_size
    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes
    assert result.per_rank_used_bytes <= result.per_rank_capacity_bytes

    overflow_config = _make_inference_config(
        expected_max_batch_size + 1,
        input_sequence_length=2048,
        output_sequence_length=512,
        parallel_config=ParallelConfig(num_ranks=1),
    )
    with pytest.raises(InsufficientGPUMemoryError):
        validate_batch_size_or_raise("A100_80GB", model_config, overflow_config)


def test_calculate_max_batch_size_raises_when_batch_size_one_does_not_fit():
    model_config = LlamaModelConfig(
        hidden_size=16384,
        num_attention_heads=128,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        intermediate_size=53248,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=126,
    )
    inference_config = _make_inference_config(
        32,
        input_sequence_length=16,
        output_sequence_length=16,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    with pytest.raises(InsufficientGPUMemoryError):
        calculate_max_batch_size("A100_80GB", model_config, inference_config)


def test_validate_batch_size_rejects_invalid_dense_parallel_product():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
    )
    inference_config = _make_inference_config(
        2,
        parallel_config=DenseParallelConfig(num_ranks=4, dp_size=1, tp_size=2),
    )

    with pytest.raises(ValueError, match="num_ranks == dp_size \\* tp_size"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_validate_batch_size_rejects_gqa_heads_not_divisible_by_tp():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=3,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
        attention_type="gqa",
    )
    inference_config = _make_inference_config(
        2,
        parallel_config=DenseParallelConfig(num_ranks=2, dp_size=1, tp_size=2),
    )

    with pytest.raises(ValueError, match="num_kv_heads=3 must be divisible by 2"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_validate_batch_size_rejects_mha_heads_not_divisible_by_tp():
    model_config = Qwen3ModelConfig(
        hidden_size=896,
        num_attention_heads=7,
        num_key_value_heads=7,
        max_position_embeddings=4096,
        intermediate_size=1792,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
        attention_type="mha",
    )
    inference_config = _make_inference_config(
        2,
        parallel_config=DenseParallelConfig(num_ranks=2, dp_size=1, tp_size=2),
    )

    with pytest.raises(ValueError, match="num_kv_heads=7 must be divisible by 2"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_validate_batch_size_rejects_moe_parallel_config():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
    )
    inference_config = _make_inference_config(
        2,
        parallel_config=MoEParallelConfig(
            num_ranks=2,
            attn_dp_size=2,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=2,
        ),
    )

    with pytest.raises(NotImplementedError, match="MoEParallelConfig"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_validate_batch_size_qwen3_moe_single_rank_counts_router_and_expert_weights():
    model_config = _make_qwen3_moe_model_config()
    inference_config = _make_inference_config(
        4,
        parallel_config=MoEParallelConfig(
            num_ranks=1,
            attn_dp_size=1,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=1,
        ),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        1024 * ((8 + 2 * 2) * 128)
        + 1024 * (8 * 128)
        + (2 * 1024 + 2 * 128)
        + 1024 * 4
        + 4 * (1024 * (256 * 2) + 1024 * 256)
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 2 * 2
    per_layer_kv_values = 4 * (8 + 4) * 2 * 128 * 2
    expected_per_rank_kv_bytes = per_layer_kv_values * 2 * 2

    assert result.dp_size == 1
    assert result.tp_size == 1
    assert result.ffn_ep_size == 1
    assert result.ffn_tp_size == 1
    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes
    assert result.per_rank_kv_cache_bytes == expected_per_rank_kv_bytes


def test_validate_batch_size_qwen3_moe_multi_rank_splits_attention_and_experts():
    model_config = _make_qwen3_moe_model_config()
    inference_config = _make_inference_config(
        9,
        parallel_config=MoEParallelConfig(
            num_ranks=4,
            attn_dp_size=2,
            attn_tp_size=2,
            ffn_tp_size=2,
            ffn_ep_size=2,
        ),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        1024 * (((8 + 2 * 2) * 128) // 2)
        + 1024 * ((8 * 128) // 2)
        + (2 * 1024 + 2 * 128)
        + 1024 * 4
        + (4 // 2) * (1024 * ((256 * 2) // 2) + 1024 * (256 // 2))
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 2 * 2
    expected_rank_batch = (9 + 2 - 1) // 2
    expected_local_kv_heads = 2 // 2
    per_layer_kv_values = (
        expected_rank_batch * (8 + 4) * expected_local_kv_heads * 128 * 2
    )
    expected_per_rank_kv_bytes = per_layer_kv_values * 2 * 2

    assert result.num_ranks == 4
    assert result.dp_size == 2
    assert result.tp_size == 2
    assert result.ffn_ep_size == 2
    assert result.ffn_tp_size == 2
    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes
    assert result.per_rank_kv_cache_bytes == expected_per_rank_kv_bytes
    assert result.total_weight_bytes == expected_per_rank_weight_bytes * 4
    assert result.total_kv_cache_bytes == expected_per_rank_kv_bytes * 4


def test_calculate_max_batch_size_returns_exact_qwen3_moe_limit():
    model_config = _make_qwen3_moe_model_config(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=4,
        max_position_embeddings=40960,
        intermediate_size=4096,
        moe_intermediate_size=512,
        num_experts=8,
        num_experts_per_tok=2,
        num_hidden_layers=24,
    )
    inference_config = _make_inference_config(
        999,
        input_sequence_length=2048,
        output_sequence_length=512,
        parallel_config=MoEParallelConfig(
            num_ranks=1,
            attn_dp_size=1,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=1,
        ),
    )

    result = calculate_max_batch_size("A100_80GB", model_config, inference_config)

    per_layer_weight_params = (
        2048 * ((16 + 2 * 4) * 128)
        + 2048 * (16 * 128)
        + (2 * 2048 + 2 * 128)
        + 2048 * 8
        + 8 * (2048 * (512 * 2) + 2048 * 512)
    )
    expected_per_rank_weight_bytes = per_layer_weight_params * 24 * 2
    per_batch_per_layer_kv_values = (2048 + 512) * 4 * 128 * 2
    per_batch_full_model_kv_bytes = per_batch_per_layer_kv_values * 24 * 2
    expected_max_batch_size = (
        _a100_capacity_bytes() - expected_per_rank_weight_bytes
    ) // per_batch_full_model_kv_bytes

    assert result.batch_size == expected_max_batch_size
    assert result.per_rank_weight_bytes == expected_per_rank_weight_bytes
    assert result.per_rank_used_bytes <= result.per_rank_capacity_bytes


def test_validate_batch_size_rejects_qwen3_moe_shared_expert():
    model_config = _make_qwen3_moe_model_config(shared_expert_intermediate_size=128)
    inference_config = _make_inference_config(
        2,
        parallel_config=MoEParallelConfig(
            num_ranks=1,
            attn_dp_size=1,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=1,
        ),
    )

    with pytest.raises(NotImplementedError, match="shared expert"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_validate_batch_size_rejects_qwen3_moe_sparse_step_not_one():
    model_config = _make_qwen3_moe_model_config(decoder_sparse_step=2)
    inference_config = _make_inference_config(
        2,
        parallel_config=MoEParallelConfig(
            num_ranks=1,
            attn_dp_size=1,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=1,
        ),
    )

    with pytest.raises(NotImplementedError, match="decoder_sparse_step == 1"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_validate_batch_size_rejects_qwen3_moe_mlp_only_layers():
    model_config = _make_qwen3_moe_model_config(mlp_only_layers=[0])
    inference_config = _make_inference_config(
        2,
        parallel_config=MoEParallelConfig(
            num_ranks=1,
            attn_dp_size=1,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=1,
        ),
    )

    with pytest.raises(NotImplementedError, match="empty mlp_only_layers"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)


def test_qwen3_8b_model_card_capacity_matches_expected_scale():
    model_config = _load_qwen3_8b_model_config()
    inference_config = _make_inference_config(
        1,
        input_sequence_length=2048,
        output_sequence_length=512,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)
    max_result = calculate_max_batch_size("A100_80GB", model_config, inference_config)

    assert result.per_rank_weight_bytes == 13_892_143_104
    assert result.per_rank_kv_cache_bytes == 377_487_360
    assert max_result.batch_size == 190


def test_qwen3_moe_235b_model_card_capacity_matches_expected_scale():
    model_config = _load_qwen3_moe_235b_model_config()
    inference_config = _make_inference_config(
        1,
        input_sequence_length=2048,
        output_sequence_length=512,
        parallel_config=MoEParallelConfig(
            num_ranks=128,
            attn_dp_size=32,
            attn_tp_size=4,
            ffn_tp_size=4,
            ffn_ep_size=32,
        ),
    )

    result = validate_batch_size_or_raise("A100_80GB", model_config, inference_config)
    max_result = calculate_max_batch_size("A100_80GB", model_config, inference_config)

    assert result.per_rank_weight_bytes == 6_999_784_448
    assert result.per_rank_kv_cache_bytes == 123_207_680
    assert max_result.batch_size == 20_480


def test_validate_batch_size_rejects_mla_attention_type():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=4096,
        intermediate_size=2048,
        hidden_act="silu",
        head_dim=128,
        num_hidden_layers=2,
        attention_type="mla",
    )
    inference_config = _make_inference_config(
        2,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    with pytest.raises(NotImplementedError, match="MLA"):
        validate_batch_size_or_raise("A100_80GB", model_config, inference_config)
