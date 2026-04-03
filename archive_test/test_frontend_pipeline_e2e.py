import json
from pathlib import Path

import pytest


torch = pytest.importorskip("torch")

from torch.fx import GraphModule

from nandmachine.commands.macro import FlashAttnOp, MacroOp, MatMulOp, VectorOp
from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import DenseParallelConfig, InferenceConfig
from nandmachine.config.model_config import LlamaModelConfig, Qwen3ModelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta, NxTracer
from nandmachine.frontend.core.passes.cod_gen import CodeGenPass
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.network.llama import LlamaDecoderLayer
from nandmachine.frontend.network.qwen3 import Qwen3DecoderLayer
from nandmachine.frontend.utlis import build_kv_cache_state
from nandmachine.simulator.entry_point import run_macro_ops


MODEL_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "model_cards" / "qwen3-8B.json"
)
LLAMA_MODEL_CARD_PATH = (
    Path(__file__).resolve().parents[1] / "model_cards" / "llama-405B.json"
)


def _load_qwen3_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig.from_dict(json.loads(MODEL_CARD_PATH.read_text()))


def _load_llama_config() -> LlamaModelConfig:
    return LlamaModelConfig.from_dict(json.loads(LLAMA_MODEL_CARD_PATH.read_text()))


def _build_nand_config() -> NandConfig:
    return NandConfig(
        num_channels=1,
        num_plane=1,
        num_block=4,
        num_pages=16,
        tRead=1.0,
        tWrite=2.0,
        tErase=3.0,
        page_size=4,
        sram_threshold=1,
    )


def _build_inference_config() -> InferenceConfig:
    return InferenceConfig(
        batch_size=2,
        input_sequence_length=8,
        output_sequence_length=4,
        weight_bits=16,
        activation_bits=16,
        kv_cache_bits=16,
        kv_block_size_bytes=1024,
        memory_backend="nand",
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
    )


def _build_small_qwen3_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        intermediate_size=32,
        hidden_act="silu",
        head_dim=4,
    )


def _build_small_llama_config() -> LlamaModelConfig:
    return LlamaModelConfig(
        hidden_size=16,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        intermediate_size=32,
        hidden_act="silu",
        head_dim=4,
    )


def _build_sim_nand_config() -> NandConfig:
    return NandConfig(
        num_channels=1,
        num_plane=2,
        num_block=8,
        num_pages=32,
        tRead=4.0,
        tWrite=8.0,
        tErase=16.0,
        page_size=16,
        sram_threshold=256,
    )


def _build_sim_inference_config() -> InferenceConfig:
    return InferenceConfig(
        batch_size=2,
        input_sequence_length=4,
        output_sequence_length=2,
        weight_bits=16,
        activation_bits=16,
        kv_cache_bits=16,
        kv_block_size_bytes=64,
        memory_backend="nand",
        parallel_config=DenseParallelConfig(num_ranks=1, tp_size=1, dp_size=1),
    )


def _build_hbm_only_architecture() -> dict[str, object]:
    return {"mode": "hbm_only"}


def _generate_macro_ops(
    model_config: Qwen3ModelConfig | LlamaModelConfig,
    nand_config: NandConfig,
    inference_config: InferenceConfig,
    model_cls: type[Qwen3DecoderLayer] | type[LlamaDecoderLayer],
) -> list[MacroOp]:
    kv_cache_state = build_kv_cache_state(
        nand_config,
        model_config,
        inference_config,
    )

    with torch.device("meta"):
        model = model_cls(model_config, tp_size=1)
        graph = NxTracer().trace(model)
        graph_module = GraphModule(model, graph)

    NormalizePass().transform(graph_module)

    graph_module.graph.meta = {
        CodeGenPass.GRAPH_META_KEY: NxGraphMeta(
            nand_config=nand_config,
            model_config=model_config,
            inference_config=inference_config,
            kv_cache_state=kv_cache_state,
        )
    }

    CodeGenPass().transform(graph_module)
    return graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]


def test_frontend_pipeline_generates_macro_ops_for_qwen3_decoder_layer():
    macro_op_list = _generate_macro_ops(
        _load_qwen3_config(),
        _build_nand_config(),
        _build_inference_config(),
        Qwen3DecoderLayer,
    )
    vector_ops = [op for op in macro_op_list if isinstance(op, VectorOp)]

    assert isinstance(macro_op_list, list)
    assert macro_op_list
    assert all(isinstance(op, MacroOp) for op in macro_op_list)
    assert any(isinstance(op, MatMulOp) for op in macro_op_list)
    assert any(isinstance(op, FlashAttnOp) for op in macro_op_list)
    assert any(op.vector_op_type == "rms_norm" for op in vector_ops)
    assert any(op.vector_op_type == "silu_mul" for op in vector_ops)
    assert all(all(dim > 0 for dim in op.vector_shape) for op in vector_ops)


def test_frontend_pipeline_macro_ops_run_on_xpu_simulator():
    model_config = _build_small_qwen3_config()
    nand_config = _build_sim_nand_config()
    inference_config = _build_sim_inference_config()

    macro_op_list = _generate_macro_ops(
        model_config,
        nand_config,
        inference_config,
        Qwen3DecoderLayer,
    )
    result = run_macro_ops(
        nand_config,
        macro_op_list,
        hbf_sram_intermediate_buffer=True,
        memory_architecture=_build_hbm_only_architecture(),
    )

    assert result.cycle > 0
    assert result.time_ns > 0
    assert any(isinstance(op, VectorOp) for op in macro_op_list)
    assert any(isinstance(op, MatMulOp) for op in macro_op_list)
    assert any(isinstance(op, FlashAttnOp) for op in macro_op_list)


def test_frontend_pipeline_generates_macro_ops_for_llama_decoder_layer():
    macro_op_list = _generate_macro_ops(
        _load_llama_config(),
        _build_nand_config(),
        _build_inference_config(),
        LlamaDecoderLayer,
    )
    vector_ops = [op for op in macro_op_list if isinstance(op, VectorOp)]

    assert isinstance(macro_op_list, list)
    assert macro_op_list
    assert all(isinstance(op, MacroOp) for op in macro_op_list)
    assert any(isinstance(op, MatMulOp) for op in macro_op_list)
    assert any(isinstance(op, FlashAttnOp) for op in macro_op_list)
    assert any(op.vector_op_type == "rms_norm" for op in vector_ops)
    assert any(op.vector_op_type == "silu_mul" for op in vector_ops)
    assert all(all(dim > 0 for dim in op.vector_shape) for op in vector_ops)


def test_frontend_pipeline_llama_macro_ops_run_on_xpu_simulator():
    model_config = _build_small_llama_config()
    nand_config = _build_sim_nand_config()
    inference_config = _build_sim_inference_config()

    macro_op_list = _generate_macro_ops(
        model_config,
        nand_config,
        inference_config,
        LlamaDecoderLayer,
    )
    result = run_macro_ops(
        nand_config,
        macro_op_list,
        hbf_sram_intermediate_buffer=True,
        memory_architecture=_build_hbm_only_architecture(),
    )

    assert result.cycle > 0
    assert result.time_ns > 0
    assert any(isinstance(op, VectorOp) for op in macro_op_list)
    assert any(isinstance(op, MatMulOp) for op in macro_op_list)
    assert any(isinstance(op, FlashAttnOp) for op in macro_op_list)
