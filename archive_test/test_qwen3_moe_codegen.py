import pytest


torch = pytest.importorskip("torch")

from nandmachine.commands.macro import All2AllOp, MatMulOp, VectorOp
from nandmachine.config.cache_state import KVCacheState
from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import InferenceConfig, MoEParallelConfig
from nandmachine.config.model_config import ModelConfigBase, Qwen3MoEModelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta
from nandmachine.frontend.modules.modules import FusedMoE
from nandmachine.frontend.network.qwen3_moe import (
    Qwen3MoEAttention,
    Qwen3MoEDecoderLayer,
)


def _build_graph_meta(batch_size: int, parallel_config: MoEParallelConfig) -> NxGraphMeta:
    return NxGraphMeta(
        nand_config=NandConfig(
            num_channels=1,
            num_plane=1,
            num_block=4,
            num_pages=16,
            tRead=1.0,
            tWrite=2.0,
            tErase=3.0,
            page_size=4,
            sram_threshold=1024,
        ),
        model_config=ModelConfigBase(attention_type="gqa"),
        inference_config=InferenceConfig(
            batch_size=batch_size,
            input_sequence_length=8,
            output_sequence_length=4,
            weight_bits=16,
            activation_bits=16,
            kv_cache_bits=16,
            kv_block_size_bytes=1024,
            memory_backend="nand",
            parallel_config=parallel_config,
        ),
        kv_cache_state=KVCacheState(
            total_kv_cache_size_per_layer=1024,
            kv_block_size_tokens=16,
            num_kv_blocks=4,
            num_nand_pages_per_layer=8,
            num_hyper_pages_per_layer=1,
        ),
    )


def test_fused_moe_codegen_uses_expert_batch_size_for_expert_ops():
    graph_meta = _build_graph_meta(
        batch_size=5,
        parallel_config=MoEParallelConfig(
            num_ranks=1,
            attn_dp_size=1,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=1,
        ),
    )
    module = FusedMoE(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
    )

    macro_op_list = module.macro_code_gen(graph_meta)
    vector_ops = [op for op in macro_op_list if isinstance(op, VectorOp)]
    matmul_ops = [op for op in macro_op_list if isinstance(op, MatMulOp)]

    assert any(
        op.vector_op_type == "moe_topk_router" and op.vector_shape == [5, 4, 2]
        for op in vector_ops
    )
    assert any(
        op.vector_op_type == "moe_weighted_sum" and op.vector_shape == [5, 16]
        for op in vector_ops
    )
    assert any(
        op.vector_op_type == "silu_mul" and op.vector_shape == [3, 32]
        for op in vector_ops
    )
    assert any(op.dim == (3, 16, 64) for op in matmul_ops)
    assert any(op.dim == (3, 32, 16) for op in matmul_ops)


def test_fused_moe_codegen_clamps_expert_batch_size_to_one():
    graph_meta = _build_graph_meta(
        batch_size=1,
        parallel_config=MoEParallelConfig(
            num_ranks=1,
            attn_dp_size=1,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=1,
        ),
    )
    module = FusedMoE(
        hidden_size=8,
        intermediate_size=16,
        num_experts=8,
        top_k=1,
    )

    macro_op_list = module.macro_code_gen(graph_meta)
    vector_ops = [op for op in macro_op_list if isinstance(op, VectorOp)]
    matmul_ops = [op for op in macro_op_list if isinstance(op, MatMulOp)]

    assert any(
        op.vector_op_type == "moe_topk_router" and op.vector_shape == [1, 8, 1]
        for op in vector_ops
    )
    assert any(
        op.vector_op_type == "silu_mul" and op.vector_shape == [1, 16]
        for op in vector_ops
    )
    assert any(op.dim == (1, 8, 32) for op in matmul_ops)
    assert any(op.dim == (1, 16, 8) for op in matmul_ops)


def test_fused_moe_codegen_uses_local_batch_for_all_to_all_and_expert_shapes():
    graph_meta = _build_graph_meta(
        batch_size=6,
        parallel_config=MoEParallelConfig(
            num_ranks=2,
            attn_dp_size=2,
            attn_tp_size=1,
            ffn_tp_size=1,
            ffn_ep_size=2,
        ),
    )
    module = FusedMoE(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
        ffn_ep_size=2,
    )

    macro_op_list = module.macro_code_gen(graph_meta)
    all_to_all_ops = [op for op in macro_op_list if isinstance(op, All2AllOp)]
    silu_ops = [
        op for op in macro_op_list
        if isinstance(op, VectorOp) and op.vector_op_type == "silu_mul"
    ]

    assert len(all_to_all_ops) == 2
    assert all(op.num_gpus == 2 for op in all_to_all_ops)
    assert all(op.data_size == 96 for op in all_to_all_ops)
    assert len(module.experts) == 2
    assert len(silu_ops) == 2
    assert all(op.vector_shape == [2, 32] for op in silu_ops)


def test_fused_moe_codegen_uses_ffn_tp_size_for_expert_ops():
    graph_meta = _build_graph_meta(
        batch_size=12,
        parallel_config=MoEParallelConfig(
            num_ranks=4,
            attn_dp_size=4,
            attn_tp_size=1,
            ffn_tp_size=2,
            ffn_ep_size=2,
        ),
    )
    module = FusedMoE(
        hidden_size=16,
        intermediate_size=32,
        num_experts=4,
        top_k=2,
        ffn_ep_size=2,
        ffn_tp_size=2,
    )

    macro_op_list = module.macro_code_gen(graph_meta)
    vector_ops = [op for op in macro_op_list if isinstance(op, VectorOp)]
    matmul_ops = [op for op in macro_op_list if isinstance(op, MatMulOp)]
    all_to_all_ops = [op for op in macro_op_list if isinstance(op, All2AllOp)]

    assert len(all_to_all_ops) == 2
    assert all(op.num_gpus == 4 for op in all_to_all_ops)
    assert all(op.data_size == 48 for op in all_to_all_ops)
    assert len(module.experts) == 2
    assert any(
        op.vector_op_type == "silu_mul" and op.vector_shape == [2, 16]
        for op in vector_ops
    )
    assert any(op.dim == (2, 16, 32) for op in matmul_ops)
    assert any(op.dim == (2, 16, 16) for op in matmul_ops)


def test_moe_parallel_config_rejects_invalid_world_size():
    with pytest.raises(
        ValueError,
        match="num_ranks == attn_dp_size \\* attn_tp_size == ffn_ep_size \\* ffn_tp_size",
    ):
        MoEParallelConfig(
            num_ranks=8,
            attn_dp_size=8,
            attn_tp_size=1,
            ffn_tp_size=2,
            ffn_ep_size=2,
        )


def test_qwen3_moe_decoder_layer_uses_parallel_config():
    config = type(
        "MockQwen3MoeConfig",
        (),
        {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 64,
            "num_attention_heads": 8,
            "num_key_value_heads": 4,
            "max_position_embeddings": 128,
            "intermediate_size": 96,
            "moe_intermediate_size": 32,
            "num_experts": 8,
            "num_experts_per_tok": 2,
            "num_hidden_layers": 12,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "hidden_act": "silu",
            "ffn_type": "moe",
            "head_dim": 8,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 10000.0,
            "shared_expert_intermediate_size": None,
        },
    )()
    parallel_config = MoEParallelConfig(
        num_ranks=4,
        attn_dp_size=4,
        attn_tp_size=1,
        ffn_tp_size=2,
        ffn_ep_size=2,
    )

    layer = Qwen3MoEDecoderLayer(config, parallel_config)

    assert isinstance(layer.self_attn, Qwen3MoEAttention)
    assert layer.self_attn.tp_size == 1
    assert layer.self_attn.dp_size == 4
    assert layer.self_attn.attn.tp_size == 1
    assert layer.self_attn.attn.dp_size == 4
    assert layer.mlp.ffn_tp_size == 2


def test_qwen3_moe_model_config_from_config_reads_full_model_fields():
    config = type(
        "MockQwen3MoeConfig",
        (),
        {
            "model_type": "qwen3_moe",
            "architectures": ["Qwen3MoeForCausalLM"],
            "hidden_size": 4096,
            "num_attention_heads": 64,
            "num_key_value_heads": 4,
            "max_position_embeddings": 40960,
            "intermediate_size": 12288,
            "moe_intermediate_size": 1536,
            "num_experts": 128,
            "num_experts_per_tok": 8,
            "num_hidden_layers": 94,
            "decoder_sparse_step": 1,
            "mlp_only_layers": [],
            "hidden_act": "silu",
            "ffn_type": "moe",
            "head_dim": 128,
            "rms_norm_eps": 1e-6,
            "attention_bias": False,
            "rope_theta": 1_000_000.0,
            "shared_expert_intermediate_size": None,
        },
    )()

    model_config = Qwen3MoEModelConfig.from_config(config)

    assert model_config.intermediate_size == 12288
    assert model_config.moe_intermediate_size == 1536
    assert model_config.num_hidden_layers == 94
    assert model_config.decoder_sparse_step == 1
    assert model_config.mlp_only_layers == []
