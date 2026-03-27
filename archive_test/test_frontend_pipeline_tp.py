import pytest


torch = pytest.importorskip("torch")

from torch.fx import GraphModule

from nandmachine.commands.macro import AllReduceOp, FlashAttnOp, MacroOp, MatMulOp, VectorOp
from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import DenseParallelConfig, InferenceConfig
from nandmachine.config.model_config import Qwen3ModelConfig
from nandmachine.frontend.core.graph.base import NxGraphMeta, NxTracer
from nandmachine.frontend.core.passes.cod_gen import CodeGenPass
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.network.qwen3 import Qwen3DecoderLayer
from nandmachine.frontend.utlis import build_kv_cache_state


def _build_nand_config() -> NandConfig:
    return NandConfig(
        num_channels=1,
        num_plane=2,
        num_block=8,
        num_pages=32,
        tRead=4.0,
        tWrite=8.0,
        tErase=16.0,
        page_size=16,
        sram_threshold=1,
    )


def _build_model_config() -> Qwen3ModelConfig:
    return Qwen3ModelConfig(
        hidden_size=32,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=64,
        intermediate_size=64,
        hidden_act="silu",
        head_dim=4,
    )


def _build_inference_config(tp_size: int) -> InferenceConfig:
    return InferenceConfig(
        batch_size=2,
        input_sequence_length=4,
        output_sequence_length=2,
        weight_bits=16,
        activation_bits=16,
        kv_cache_bits=16,
        kv_block_size_bytes=64,
        parallel_config=DenseParallelConfig(
            num_ranks=tp_size,
            tp_size=tp_size,
            dp_size=1,
        ),
    )


def _generate_macro_ops(tp_size: int) -> list[MacroOp]:
    model_config = _build_model_config()
    nand_config = _build_nand_config()
    inference_config = _build_inference_config(tp_size)
    kv_cache_state = build_kv_cache_state(
        nand_config,
        model_config,
        inference_config,
    )

    with torch.device("meta"):
        model = Qwen3DecoderLayer(model_config, tp_size=tp_size)
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


def test_frontend_pipeline_generates_tp_macro_ops_for_qwen3_decoder_layer():
    tp_size = 4
    macro_op_list = _generate_macro_ops(tp_size)
    all_reduce_ops = [
        op for op in macro_op_list if isinstance(op, AllReduceOp)
    ]
    vector_ops = [op for op in macro_op_list if isinstance(op, VectorOp)]

    assert macro_op_list
    assert all(isinstance(op, MacroOp) for op in macro_op_list)
    assert any(isinstance(op, MatMulOp) for op in macro_op_list)
    assert any(isinstance(op, FlashAttnOp) for op in macro_op_list)
    assert any(op.vector_op_type == "rms_norm" for op in vector_ops)
    assert any(op.vector_op_type == "silu_mul" for op in vector_ops)

    assert len(all_reduce_ops) == 2
    assert all(op.num_ranks == tp_size for op in all_reduce_ops)
    assert all(op.data_size == 128 for op in all_reduce_ops)
    assert all(op.input_ops for op in all_reduce_ops)
    assert all(
        all(isinstance(input_op, MatMulOp) for input_op in op.input_ops)
        for op in all_reduce_ops
    )
