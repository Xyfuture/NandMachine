import pytest

from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import DenseParallelConfig, InferenceConfig, ParallelConfig
from nandmachine.config.model_config import LlamaModelConfig, Qwen3ModelConfig
from nandmachine.frontend.utlis import calculate_kv_cache_state


def make_nand_config(page_size: int = 1, num_plane: int = 4) -> NandConfig:
    return NandConfig(
        num_channels=1,
        num_plane=num_plane,
        num_block=64,
        num_pages=256,
        tRead=1.0,
        tWrite=1.0,
        tErase=1.0,
        page_size=page_size,
        sram_threshold=1,
    )


def make_inference_config(
    batch_size: int,
    input_sequence_length: int,
    output_sequence_length: int,
    kv_cache_bits: int,
    parallel_config: ParallelConfig,
    kv_block_size_bytes: int = 64 * 1024,
) -> InferenceConfig:
    return InferenceConfig(
        batch_size=batch_size,
        input_sequence_length=input_sequence_length,
        output_sequence_length=output_sequence_length,
        weight_bits=16,
        activation_bits=16,
        kv_cache_bits=kv_cache_bits,
        kv_block_size_bytes=kv_block_size_bytes,
        parallel_config=parallel_config,
    )


def test_calculate_kv_cache_state_for_gqa_uses_peak_length_and_dp_rank_batch():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        attention_type="gqa",
    )
    inference_config = make_inference_config(
        batch_size=10,
        input_sequence_length=128,
        output_sequence_length=32,
        kv_cache_bits=16,
        parallel_config=DenseParallelConfig(num_ranks=4, tp_size=1, dp_size=4),
    )

    state = calculate_kv_cache_state(
        make_nand_config(page_size=1, num_plane=4),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 1_966_080
    assert state.num_nand_pages_per_layer == 1_920
    assert state.num_hyper_pages_per_layer == 480
    assert state.kv_block_size_tokens == 16
    assert state.num_kv_blocks == 30
    assert state.kv_cache_num_pages_per_layer == state.num_nand_pages_per_layer


def test_calculate_kv_cache_state_for_mha_uses_attention_heads_as_kv_heads():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        attention_type="mha",
    )
    inference_config = make_inference_config(
        batch_size=2,
        input_sequence_length=16,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
    )

    state = calculate_kv_cache_state(
        make_nand_config(page_size=4, num_plane=2),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 262_144
    assert state.num_nand_pages_per_layer == 64
    assert state.num_hyper_pages_per_layer == 32
    assert state.kv_block_size_tokens == 8
    assert state.num_kv_blocks == 4


def test_calculate_kv_cache_state_derives_head_dim_from_hidden_size():
    model_config = Qwen3ModelConfig(
        hidden_size=2048,
        num_attention_heads=16,
        num_key_value_heads=16,
        max_position_embeddings=40960,
        intermediate_size=8192,
        hidden_act="silu",
        head_dim=None,
        attention_type="mha",
    )
    inference_config = make_inference_config(
        batch_size=1,
        input_sequence_length=10,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
    )

    state = calculate_kv_cache_state(
        make_nand_config(page_size=4, num_plane=2),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 40_960
    assert state.num_nand_pages_per_layer == 10
    assert state.num_hyper_pages_per_layer == 5
    assert state.kv_block_size_tokens == 16
    assert state.num_kv_blocks == 1


def test_calculate_kv_cache_state_scales_with_kv_precision():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        attention_type="gqa",
    )
    parallel_config = DenseParallelConfig(num_ranks=4, tp_size=1, dp_size=4)

    state_8bit = calculate_kv_cache_state(
        make_nand_config(page_size=1, num_plane=4),
        model_config,
        make_inference_config(
            batch_size=8,
            input_sequence_length=64,
            output_sequence_length=16,
            kv_cache_bits=8,
            parallel_config=parallel_config,
        ),
    )
    state_16bit = calculate_kv_cache_state(
        make_nand_config(page_size=1, num_plane=4),
        model_config,
        make_inference_config(
            batch_size=8,
            input_sequence_length=64,
            output_sequence_length=16,
            kv_cache_bits=16,
            parallel_config=parallel_config,
        ),
    )

    assert state_16bit.total_kv_cache_size_per_layer == 2 * state_8bit.total_kv_cache_size_per_layer
    assert state_16bit.num_nand_pages_per_layer == 2 * state_8bit.num_nand_pages_per_layer
    assert state_16bit.num_hyper_pages_per_layer == 2 * state_8bit.num_hyper_pages_per_layer
    assert state_8bit.kv_block_size_tokens == 32
    assert state_16bit.kv_block_size_tokens == 16
    assert state_16bit.num_kv_blocks == 2 * state_8bit.num_kv_blocks


def test_calculate_kv_cache_state_uses_num_ranks_when_only_device_count_is_available():
    model_config = Qwen3ModelConfig(
        hidden_size=1024,
        num_attention_heads=8,
        num_key_value_heads=8,
        max_position_embeddings=8192,
        intermediate_size=4096,
        hidden_act="silu",
        head_dim=128,
        attention_type="mha",
    )
    inference_config = make_inference_config(
        batch_size=9,
        input_sequence_length=8,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=ParallelConfig(num_ranks=4),
    )

    state = calculate_kv_cache_state(
        make_nand_config(page_size=1, num_plane=2),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 49_152
    assert state.num_nand_pages_per_layer == 48
    assert state.num_hyper_pages_per_layer == 24
    assert state.kv_block_size_tokens == 32
    assert state.num_kv_blocks == 1


def test_calculate_kv_cache_state_supports_llama_gqa_config():
    model_config = LlamaModelConfig(
        hidden_size=16384,
        num_attention_heads=128,
        num_key_value_heads=8,
        max_position_embeddings=131072,
        intermediate_size=53248,
        hidden_act="silu",
        head_dim=128,
    )
    inference_config = make_inference_config(
        batch_size=2,
        input_sequence_length=16,
        output_sequence_length=4,
        kv_cache_bits=16,
        parallel_config=ParallelConfig(num_ranks=1),
    )

    state = calculate_kv_cache_state(
        make_nand_config(page_size=4, num_plane=2),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 163_840
    assert state.num_nand_pages_per_layer == 40
    assert state.num_hyper_pages_per_layer == 20
    assert state.kv_block_size_tokens == 16
    assert state.num_kv_blocks == 3


def test_calculate_kv_cache_state_rounds_up_kv_block_size_tokens():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        attention_type="gqa",
    )
    inference_config = make_inference_config(
        batch_size=1,
        input_sequence_length=3,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
        kv_block_size_bytes=5_000,
    )

    state = calculate_kv_cache_state(
        make_nand_config(page_size=4, num_plane=2),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 6_144
    assert state.kv_block_size_tokens == 3
    assert state.num_kv_blocks == 2


def test_calculate_kv_cache_state_keeps_block_shape_when_total_cache_is_zero():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        attention_type="gqa",
    )
    inference_config = make_inference_config(
        batch_size=2,
        input_sequence_length=0,
        output_sequence_length=0,
        kv_cache_bits=16,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
        kv_block_size_bytes=5_000,
    )

    state = calculate_kv_cache_state(
        make_nand_config(page_size=4, num_plane=2),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 0
    assert state.num_nand_pages_per_layer == 0
    assert state.num_hyper_pages_per_layer == 0
    assert state.kv_block_size_tokens == 2
    assert state.num_kv_blocks == 0


def test_qwen3_model_config_from_dict_reads_num_hidden_layers():
    config = Qwen3ModelConfig.from_dict(
        {
            "hidden_size": 4096,
            "num_attention_heads": 32,
            "num_key_value_heads": 8,
            "max_position_embeddings": 40960,
            "intermediate_size": 12288,
            "hidden_act": "silu",
            "num_hidden_layers": 36,
        }
    )

    assert config.num_hidden_layers == 36


def test_calculate_kv_cache_state_rejects_mla_attention_type():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        attention_type="mla",
    )
    inference_config = make_inference_config(
        batch_size=2,
        input_sequence_length=16,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
    )

    with pytest.raises(AssertionError, match="MLA KV cache sizing is not implemented yet"):
        calculate_kv_cache_state(
            make_nand_config(page_size=4, num_plane=2),
            model_config,
            inference_config,
        )


def test_calculate_kv_cache_state_rejects_unknown_attention_type():
    model_config = Qwen3ModelConfig(
        hidden_size=4096,
        num_attention_heads=32,
        num_key_value_heads=8,
        max_position_embeddings=40960,
        intermediate_size=12288,
        hidden_act="silu",
        head_dim=128,
        attention_type="unknown",
    )
    inference_config = make_inference_config(
        batch_size=2,
        input_sequence_length=16,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
    )

    with pytest.raises(AssertionError, match="Unsupported attention_type"):
        calculate_kv_cache_state(
            make_nand_config(page_size=4, num_plane=2),
            model_config,
            inference_config,
        )
