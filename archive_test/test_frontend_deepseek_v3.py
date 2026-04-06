import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from torch.fx import GraphModule

from nandmachine.commands.macro import (
    All2AllOp,
    AllReduceOp,
    FlashMLAOp,
    MatMulOp,
    SramPrefetch,
    SramPrefetchRelease,
    VectorOp,
)
from nandmachine.config.config import NandConfig
from nandmachine.config.hardware_config import get_device_or_raise
from nandmachine.config.inference_config import InferenceConfig, MoEParallelConfig
from nandmachine.config.model_config import DeepseekV3ModelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta, NxTracer
from nandmachine.frontend.core.passes.cod_gen import CodeGenPass
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.modules.modules import MLAAttention
from nandmachine.frontend.network.deepseek_v3 import (
    DeepseekV3Attention,
    DeepseekV3DecoderLayer,
    DeepseekV3MoE,
)
from nandmachine.frontend.utlis import build_kv_cache_state
from nandmachine.frontend.validator import validate_batch_size_or_raise
from nandmachine.kernels.attention import MLAHBMKernel, MLANandKernel
from nandmachine.simulator.entry_point import run_macro_ops


MODEL_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "model_cards" / "deepseek-v3.json"
)


def _build_supported_config() -> DeepseekV3ModelConfig:
    return DeepseekV3ModelConfig(
        hidden_size=32,
        num_attention_heads=4,
        max_position_embeddings=128,
        intermediate_size=48,
        moe_intermediate_size=24,
        num_hidden_layers=6,
        num_experts_per_tok=2,
        n_routed_experts=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=4,
        rms_norm_eps=1e-6,
        attention_bias=False,
        rope_theta=10000.0,
        hidden_act="silu",
        num_nextn_predict_layers=1,
        attention_type="mla",
    )


def _build_parallel_config() -> MoEParallelConfig:
    return MoEParallelConfig(
        num_ranks=4,
        attn_dp_size=4,
        attn_tp_size=1,
        ffn_tp_size=2,
        ffn_ep_size=2,
    )


def _build_nand_config() -> NandConfig:
    return NandConfig(
        num_channels=1,
        num_plane=1,
        num_block=8,
        num_pages=32,
        tRead=4.0,
        tWrite=8.0,
        tErase=16.0,
        page_size=1,
        sram_threshold=64,
    )


def _build_inference_config(memory_backend: str = "hbm") -> InferenceConfig:
    return InferenceConfig(
        batch_size=8,
        input_sequence_length=3,
        output_sequence_length=2,
        weight_bits=16,
        activation_bits=16,
        kv_cache_bits=16,
        kv_block_size_bytes=24,
        memory_backend=memory_backend,
        parallel_config=_build_parallel_config(),
    )


def _build_graph_meta(memory_backend: str = "hbm") -> NxGraphMeta:
    nand_config = _build_nand_config()
    model_config = _build_supported_config()
    inference_config = _build_inference_config(memory_backend)
    return NxGraphMeta(
        nand_config=nand_config,
        model_config=model_config,
        inference_config=inference_config,
        kv_cache_state=build_kv_cache_state(
            nand_config,
            model_config,
            inference_config,
        ),
    )


def _generate_macro_ops(memory_backend: str = "hbm"):
    graph_meta = _build_graph_meta(memory_backend)
    with torch.device("meta"):
        model = DeepseekV3DecoderLayer(
            layer_idx=3,
            config=graph_meta.model_config,
            parallel_config=graph_meta.inference_config.parallel_config,
        )
        graph = NxTracer().trace(model)
        graph_module = GraphModule(model, graph)

    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: graph_meta}
    CodeGenPass().transform(graph_module)
    return graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]


def _normalize_macro_op_list(macro_op_list):
    normalized = []
    for op in macro_op_list:
        deps = tuple(macro_op_list.index(dep) for dep in op.input_ops)
        if isinstance(op, MatMulOp):
            normalized.append(
                ("MatMulOp", deps, op.dim, op.weight_bits)
            )
        elif isinstance(op, FlashMLAOp):
            normalized.append(
                (
                    "FlashMLAOp",
                    deps,
                    op.qk_latent_bmm_shape,
                    op.qk_rope_bmm_shape,
                    op.sv_latent_bmm_shape,
                    op.softmax_shape,
                    op.weight_bits,
                )
            )
        elif isinstance(op, VectorOp):
            normalized.append(
                (
                    "VectorOp",
                    deps,
                    op.vector_op_type,
                    tuple(op.vector_shape),
                    op.weight_bits,
                )
            )
        elif isinstance(op, AllReduceOp):
            normalized.append(
                ("AllReduceOp", deps, op.num_ranks, op.data_size, op.weight_bits)
            )
        elif isinstance(op, All2AllOp):
            normalized.append(
                ("All2AllOp", deps, op.num_gpus, op.data_size, op.weight_bits)
            )
        elif isinstance(op, SramPrefetch):
            normalized.append(("SramPrefetch", deps, op.num_prefetch_pages))
        elif isinstance(op, SramPrefetchRelease):
            normalized.append(("SramPrefetchRelease", deps))
        else:
            raise TypeError(f"Unsupported macro op type: {type(op).__name__}")
    return normalized


EXPECTED_NAND_MACRO_SIGNATURE = [
    ("VectorOp", (), "rms_norm", (8, 32), 16),
    ("SramPrefetch", (), 1),
    ("MatMulOp", (1, 0), (8, 32, 8), 16),
    ("SramPrefetchRelease", (2,)),
    ("VectorOp", (2,), "rms_norm", (8, 8), 16),
    ("SramPrefetch", (), 1),
    ("MatMulOp", (5, 4), (8, 8, 32), 16),
    ("SramPrefetchRelease", (6,)),
    ("SramPrefetch", (), 1),
    ("MatMulOp", (8, 6), (8, 32, 6), 16),
    ("SramPrefetchRelease", (9,)),
    ("VectorOp", (9,), "rms_norm", (8, 4), 16),
    ("MatMulOp", (11,), (8, 6, 4), 16),
    ("SramPrefetch", (12,), 1),
    ("FlashMLAOp", (13,), (5, 4, 4, 2), (5, 4, 2, 2), (5, 4, 2, 4), (5, 8), 16),
    ("SramPrefetchRelease", (14,)),
    ("MatMulOp", (14,), (8, 4, 4), 16),
    ("SramPrefetch", (), 1),
    ("MatMulOp", (17, 16), (8, 16, 32), 16),
    ("SramPrefetchRelease", (18,)),
    ("VectorOp", (18,), "rms_norm", (8, 32), 16),
    ("SramPrefetch", (), 1),
    ("MatMulOp", (21, 20), (8, 32, 4), 16),
    ("SramPrefetchRelease", (22,)),
    ("VectorOp", (), "moe_topk_router", (8, 4, 2), 16),
    ("All2AllOp", (24,), 4, 256, 16),
    ("SramPrefetch", (), 2),
    ("MatMulOp", (26, 25), (4, 32, 24), 16),
    ("SramPrefetchRelease", (27,)),
    ("VectorOp", (), "silu_mul", (4, 12), 16),
    ("SramPrefetch", (), 1),
    ("MatMulOp", (30,), (4, 12, 32), 16),
    ("SramPrefetchRelease", (31,)),
    ("AllReduceOp", (31,), 2, 256, 16),
    ("SramPrefetch", (), 2),
    ("MatMulOp", (34,), (4, 32, 24), 16),
    ("SramPrefetchRelease", (35,)),
    ("VectorOp", (), "silu_mul", (4, 12), 16),
    ("SramPrefetch", (), 1),
    ("MatMulOp", (38,), (4, 12, 32), 16),
    ("SramPrefetchRelease", (39,)),
    ("AllReduceOp", (39,), 2, 256, 16),
    ("All2AllOp", (41,), 4, 256, 16),
    ("VectorOp", (42,), "moe_weighted_sum", (8, 32), 16),
]


def test_deepseek_v3_model_config_from_dict_reads_supported_fields():
    raw_config = json.loads(MODEL_CARD_PATH.read_text())

    config = DeepseekV3ModelConfig.from_dict(raw_config)

    assert config.attention_type == "mla"
    assert config.q_lora_rank == 1536
    assert config.kv_lora_rank == 512
    assert config.qk_nope_head_dim == 128
    assert config.qk_rope_head_dim == 64
    assert config.v_head_dim == 128
    assert not hasattr(config, "first_k_dense_replace")
    assert not hasattr(config, "n_shared_experts")


def test_build_kv_cache_state_for_mla_uses_compressed_cache_formula():
    state = build_kv_cache_state(
        _build_nand_config(),
        _build_supported_config(),
        _build_inference_config(),
    )

    assert state.total_kv_cache_size_per_layer == 480
    assert state.num_nand_pages_per_layer == 1
    assert state.num_hyper_pages_per_layer == 1
    assert state.kv_block_size_tokens == 2
    assert state.num_kv_blocks == 20


def test_mla_attention_codegen_uses_local_batch_and_local_blocks():
    graph_meta = _build_graph_meta("hbm")
    module = MLAAttention(
        num_heads=4,
        q_lora_rank=8,
        kv_lora_rank=4,
        qk_nope_head_dim=6,
        qk_rope_head_dim=2,
        v_head_dim=4,
        tp_size=1,
        dp_size=4,
    )

    macro_op_list = module.macro_code_gen(graph_meta)

    assert len(macro_op_list) == 3
    assert isinstance(macro_op_list[0], MatMulOp)
    assert macro_op_list[0].dim == (8, 6, 4)
    assert isinstance(macro_op_list[1], FlashMLAOp)
    assert macro_op_list[1].qk_latent_bmm_shape == (5, 4, 4, 2)
    assert macro_op_list[1].qk_rope_bmm_shape == (5, 4, 2, 2)
    assert macro_op_list[1].sv_latent_bmm_shape == (5, 4, 2, 4)
    assert macro_op_list[1].softmax_shape == (5, 8)
    assert isinstance(macro_op_list[2], MatMulOp)
    assert macro_op_list[2].dim == (8, 4, 4)


def test_validate_batch_size_supports_deepseek_v3_mla_capacity_path():
    result = validate_batch_size_or_raise(
        "A100_80GB",
        {"mode": "hbm_only"},
        _build_supported_config(),
        _build_inference_config(),
    )

    assert result.batch_size == 8
    assert result.dp_size == 4
    assert result.tp_size == 1
    assert result.ffn_ep_size == 2
    assert result.ffn_tp_size == 2
    assert result.per_rank_weight_bytes > 0
    assert result.per_rank_kv_cache_bytes > 0


def test_mla_kernels_lower_to_expected_macro_op_sequences():
    hbm_macro_ops = MLAHBMKernel.lowering(
        2,
        4,
        10,
        4,
        2,
        2,
        24,
        16,
        16,
        _build_nand_config(),
    )
    nand_macro_ops = MLANandKernel.lowering(
        2,
        4,
        10,
        4,
        2,
        2,
        24,
        16,
        16,
        _build_nand_config(),
    )

    assert len(hbm_macro_ops) == 1
    assert isinstance(hbm_macro_ops[0], FlashMLAOp)
    assert hbm_macro_ops[0].qk_latent_bmm_shape == (10, 2, 4, 2)
    assert hbm_macro_ops[0].qk_rope_bmm_shape == (10, 2, 2, 2)
    assert hbm_macro_ops[0].sv_latent_bmm_shape == (10, 2, 2, 4)
    assert hbm_macro_ops[0].softmax_shape == (10, 4)
    assert isinstance(nand_macro_ops[0], SramPrefetch)
    assert isinstance(nand_macro_ops[1], FlashMLAOp)
    assert isinstance(nand_macro_ops[2], SramPrefetchRelease)


def test_deepseek_v3_decoder_layer_uses_moe_module_on_meta():
    config = _build_supported_config()
    parallel_config = _build_parallel_config()
    with torch.device("meta"):
        moe_layer = DeepseekV3DecoderLayer(3, config, parallel_config)
        hidden_states = torch.empty((2, 1, config.hidden_size), device="meta")
        moe_output = moe_layer(hidden_states)

    assert isinstance(moe_layer.self_attn, DeepseekV3Attention)
    assert isinstance(moe_layer.mlp, DeepseekV3MoE)
    assert moe_output.shape == hidden_states.shape


def test_deepseek_v3_pipeline_generates_and_runs_flashmla_macro_ops():
    macro_op_list = _generate_macro_ops("hbm")

    assert any(isinstance(op, FlashMLAOp) for op in macro_op_list)
    flash_mla_index = next(
        index for index, op in enumerate(macro_op_list) if isinstance(op, FlashMLAOp)
    )
    assert isinstance(macro_op_list[flash_mla_index - 1], MatMulOp)
    assert isinstance(macro_op_list[flash_mla_index + 1], MatMulOp)


def test_deepseek_v3_nand_macro_op_list_matches_snapshot():
    macro_op_list = _generate_macro_ops("nand")

    assert _normalize_macro_op_list(macro_op_list) == EXPECTED_NAND_MACRO_SIGNATURE


def test_deepseek_v3_hbm_codegen_keeps_single_flashmla_op():
    macro_op_list = _generate_macro_ops("hbm")

    flash_mla_ops = [op for op in macro_op_list if isinstance(op, FlashMLAOp)]

    assert len(flash_mla_ops) == 1
    assert flash_mla_ops[0].qk_latent_bmm_shape == (5, 4, 4, 2)
    assert flash_mla_ops[0].softmax_shape == (5, 8)


def test_run_macro_ops_executes_flashmla_sequence():
    absorb = MatMulOp((2, 6, 4), weight_bits=16)
    flash_mla = FlashMLAOp(
        qk_latent_bmm_shape=(2, 1, 4, 2),
        qk_rope_bmm_shape=(2, 1, 2, 2),
        sv_latent_bmm_shape=(2, 1, 2, 4),
        softmax_shape=(2, 2),
        weight_bits=16,
    ).with_inputs(absorb)
    up_proj = MatMulOp((2, 4, 4), weight_bits=16).with_inputs(flash_mla)

    result = run_macro_ops(
        _build_nand_config(),
        [absorb, flash_mla, up_proj],
        hbm_bandwidth_bytes_per_sec=get_device_or_raise("A100_80GB").io_module.bandwidth,
    )

    assert result.cycle > 0
    assert result.time_ns > 0
