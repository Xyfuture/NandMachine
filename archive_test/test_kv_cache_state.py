import pytest

from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import (
    DenseParallelConfig,
    InferenceConfig,
    MoEParallelConfig,
    ParallelConfig,
)
from nandmachine.config.model_config import LlamaModelConfig, Qwen3ModelConfig
from nandmachine.frontend.utlis import (
    build_imbalanced_kv_cache_state,
    build_kv_cache_state,
    calculate_kv_cache_state,
)


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
        memory_backend="nand",
        parallel_config=parallel_config,
    )


def test_calculate_kv_cache_state_for_gqa_uses_peak_length_and_global_batch():
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

    assert state.total_kv_cache_size_per_layer == 6_553_600
    assert state.num_nand_pages_per_layer == 6_400
    assert state.num_hyper_pages_per_layer == 1_600
    assert state.kv_block_size_tokens == 16
    assert state.num_kv_blocks == 100


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


def test_calculate_kv_cache_state_ignores_tp_when_sizing_global_kv_cache():
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

    state = calculate_kv_cache_state(
        make_nand_config(page_size=1, num_plane=4),
        model_config,
        make_inference_config(
            batch_size=8,
            input_sequence_length=64,
            output_sequence_length=16,
            kv_cache_bits=16,
            parallel_config=DenseParallelConfig(num_ranks=4, tp_size=4, dp_size=1),
        ),
    )

    assert state.total_kv_cache_size_per_layer == 2_621_440
    assert state.num_nand_pages_per_layer == 2_560
    assert state.num_hyper_pages_per_layer == 640
    assert state.kv_block_size_tokens == 16
    assert state.num_kv_blocks == 40


def test_calculate_kv_cache_state_ignores_moe_parallelism_when_sizing_global_kv_cache():
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

    state = calculate_kv_cache_state(
        make_nand_config(page_size=1, num_plane=4),
        model_config,
        make_inference_config(
            batch_size=8,
            input_sequence_length=64,
            output_sequence_length=16,
            kv_cache_bits=16,
            parallel_config=MoEParallelConfig(
                num_ranks=4,
                attn_dp_size=1,
                attn_tp_size=4,
                ffn_tp_size=2,
                ffn_ep_size=2,
            ),
        ),
    )

    assert state.total_kv_cache_size_per_layer == 2_621_440
    assert state.num_nand_pages_per_layer == 2_560
    assert state.num_hyper_pages_per_layer == 640
    assert state.kv_block_size_tokens == 16
    assert state.num_kv_blocks == 40


def test_calculate_kv_cache_state_ignores_num_ranks_when_sizing_global_kv_cache():
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

    assert state.total_kv_cache_size_per_layer == 147_456
    assert state.num_nand_pages_per_layer == 144
    assert state.num_hyper_pages_per_layer == 72
    assert state.kv_block_size_tokens == 32
    assert state.num_kv_blocks == 3


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


def test_build_imbalanced_kv_cache_state_uses_balanced_hyper_page_size_with_channels():
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
    nand_config = NandConfig(
        num_channels=2,
        num_plane=4,
        num_block=64,
        num_pages=256,
        tRead=1.0,
        tWrite=1.0,
        tErase=1.0,
        page_size=8,
        sram_threshold=1,
    )
    inference_config = make_inference_config(
        batch_size=10,
        input_sequence_length=128,
        output_sequence_length=32,
        kv_cache_bits=16,
        parallel_config=DenseParallelConfig(num_ranks=4, tp_size=1, dp_size=4),
    )

    balanced_state = build_kv_cache_state(
        nand_config,
        model_config,
        inference_config,
    )
    imbalanced_state = build_imbalanced_kv_cache_state(
        nand_config,
        model_config,
        inference_config,
    )

    assert imbalanced_state.total_kv_cache_size_per_layer == balanced_state.total_kv_cache_size_per_layer
    assert imbalanced_state.num_nand_pages_per_layer == balanced_state.num_nand_pages_per_layer
    assert imbalanced_state.kv_block_size_tokens == balanced_state.kv_block_size_tokens
    assert imbalanced_state.num_kv_blocks == balanced_state.num_kv_blocks
    assert balanced_state.num_hyper_pages_per_layer == 100
    assert imbalanced_state.num_hyper_pages_per_layer == 100


def test_calculate_kv_cache_state_uses_num_channels_in_hyper_page_count():
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
        input_sequence_length=16,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
        kv_block_size_bytes=8 * 1024,
    )

    state_one_channel = calculate_kv_cache_state(
        NandConfig(
            num_channels=1,
            num_plane=1,
            num_block=4,
            num_pages=8,
            tRead=1.0,
            tWrite=1.0,
            tErase=1.0,
            page_size=8,
            sram_threshold=1,
        ),
        model_config,
        inference_config,
    )
    state_two_channels = calculate_kv_cache_state(
        NandConfig(
            num_channels=2,
            num_plane=1,
            num_block=4,
            num_pages=8,
            tRead=1.0,
            tWrite=1.0,
            tErase=1.0,
            page_size=8,
            sram_threshold=1,
        ),
        model_config,
        inference_config,
    )

    assert state_one_channel.total_kv_cache_size_per_layer == state_two_channels.total_kv_cache_size_per_layer
    assert state_one_channel.num_nand_pages_per_layer == state_two_channels.num_nand_pages_per_layer
    assert state_one_channel.num_hyper_pages_per_layer == 4
    assert state_two_channels.num_hyper_pages_per_layer == 2


def test_build_imbalanced_kv_cache_state_uses_num_channels_in_bin_count():
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
        input_sequence_length=16,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
        kv_block_size_bytes=8 * 1024,
    )

    state_one_channel = build_imbalanced_kv_cache_state(
        NandConfig(
            num_channels=1,
            num_plane=1,
            num_block=4,
            num_pages=8,
            tRead=1.0,
            tWrite=1.0,
            tErase=1.0,
            page_size=8,
            sram_threshold=1,
        ),
        model_config,
        inference_config,
    )
    state_two_channels = build_imbalanced_kv_cache_state(
        NandConfig(
            num_channels=2,
            num_plane=1,
            num_block=4,
            num_pages=8,
            tRead=1.0,
            tWrite=1.0,
            tErase=1.0,
            page_size=8,
            sram_threshold=1,
        ),
        model_config,
        inference_config,
    )

    assert state_one_channel.num_kv_blocks == 4
    assert state_one_channel.num_hyper_pages_per_layer == 4
    assert state_two_channels.num_hyper_pages_per_layer == 3


def test_build_imbalanced_kv_cache_state_returns_zero_when_no_kv_blocks():
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

    state = build_imbalanced_kv_cache_state(
        make_nand_config(page_size=4, num_plane=2),
        model_config,
        inference_config,
    )

    assert state.total_kv_cache_size_per_layer == 0
    assert state.num_kv_blocks == 0
    assert state.num_hyper_pages_per_layer == 0


def test_build_imbalanced_kv_cache_state_rejects_non_positive_bin_count():
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
    nand_config = NandConfig(
        num_channels=1,
        num_plane=1,
        num_block=1,
        num_pages=1,
        tRead=1.0,
        tWrite=1.0,
        tErase=1.0,
        page_size=1,
        sram_threshold=1,
    )
    inference_config = make_inference_config(
        batch_size=1,
        input_sequence_length=1,
        output_sequence_length=0,
        kv_cache_bits=8,
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
        kv_block_size_bytes=2 * 1024,
    )

    with pytest.raises(ValueError, match="num_bins must be positive, got 0"):
        build_imbalanced_kv_cache_state(
            nand_config,
            model_config,
            inference_config,
        )


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
