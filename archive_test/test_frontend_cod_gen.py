import logging

import pytest


torch = pytest.importorskip("torch")

from torch.fx import GraphModule

from nandmachine.commands.macro import AllReduceOp, MatMulOp, VectorOp
from nandmachine.config.cache_state import KVCacheState
from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import DenseParallelConfig, InferenceConfig, ParallelConfig
from nandmachine.config.model_config import ModelConfigBase
from nandmachine.frontend.core.graph.base import NxGraphMeta, NxTracer
from nandmachine.frontend.core.passes.cod_gen import CodeGenPass
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.modules.modules import (
    ColumnParallelLinear,
    HookModuleBase,
    RMSNorm,
    RowParallelLinear,
)


def _trace_graph_module(model: torch.nn.Module) -> GraphModule:
    graph = NxTracer().trace(model)
    return GraphModule(model, graph)


def _build_graph_meta(
    parallel_config: ParallelConfig | None = None,
) -> NxGraphMeta:
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
            sram_threshold=1,
        ),
        model_config=ModelConfigBase(attention_type="gqa"),
        inference_config=InferenceConfig(
            batch_size=2,
            input_sequence_length=8,
            output_sequence_length=4,
            weight_bits=16,
            activation_bits=16,
            kv_cache_bits=16,
            kv_block_size_bytes=1024,
            parallel_config=parallel_config or ParallelConfig(num_ranks=1),
        ),
        kv_cache_state=KVCacheState(
            total_kv_cache_size_per_layer=1024,
            num_nand_pages_per_layer=8,
            num_hyper_pages_per_layer=1,
            kv_block_size_tokens=16,
            num_kv_blocks=4,
            kv_cache_num_pages_per_layer=8,
        ),
    )


class CountingHook(HookModuleBase):
    def __init__(self, label: str, num_ops: int) -> None:
        super().__init__()
        self.label = label
        self.num_ops = num_ops
        self.seen_graph_meta = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def macro_code_gen(self, graph_meta: NxGraphMeta):
        self.seen_graph_meta = graph_meta
        return [
            VectorOp(
                vector_op_type=f"{self.label}_{index}",
                vector_shape=[graph_meta.batch_size, index + 1],
            )
            for index in range(self.num_ops)
        ]


class InvalidItemHook(HookModuleBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def macro_code_gen(self, graph_meta: NxGraphMeta):
        del graph_meta
        return ["bad-op"]


class InvalidContainerHook(HookModuleBase):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x

    def macro_code_gen(self, graph_meta: NxGraphMeta):
        del graph_meta
        return (
            VectorOp(vector_op_type="tuple_op", vector_shape=[1, 1]),
        )


def test_codegen_pass_collects_macro_ops_in_topological_order(caplog: pytest.LogCaptureFixture):
    class TwoHookModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = CountingHook("first", 1)
            self.second = CountingHook("second", 2)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = self.first(x)
            x = x + y
            return self.second(x)

    graph_module = _trace_graph_module(TwoHookModel())
    NormalizePass().transform(graph_module)

    graph_meta = _build_graph_meta()
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: graph_meta}

    with caplog.at_level(logging.INFO, logger="nandmachine.frontend.core.passes.cod_gen"):
        CodeGenPass().transform(graph_module)

    macro_op_list = graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]

    assert [macro_op.vector_op_type for macro_op in macro_op_list] == [
        "first_0",
        "second_0",
        "second_1",
    ]
    assert graph_module.get_submodule("first").seen_graph_meta is graph_meta
    assert graph_module.get_submodule("second").seen_graph_meta is graph_meta
    assert "Processed node=first generated_macro_ops=1" in caplog.text
    assert "Processed node=second generated_macro_ops=2" in caplog.text


def test_codegen_pass_skips_non_hook_nodes_and_handles_empty_macro_lists():
    class MixedModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = CountingHook("first", 1)
            self.empty = CountingHook("empty", 0)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            x = self.first(x)
            x = x + y
            return self.empty(x)

    graph_module = _trace_graph_module(MixedModel())
    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: _build_graph_meta()}

    CodeGenPass().transform(graph_module)

    macro_op_list = graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]

    assert [macro_op.vector_op_type for macro_op in macro_op_list] == ["first_0"]


def test_codegen_pass_runs_rms_norm_hook_module():
    class RmsNormModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(hidden_size=16)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.norm(x)

    graph_module = _trace_graph_module(RmsNormModel())
    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: _build_graph_meta()}

    CodeGenPass().transform(graph_module)

    macro_op_list = graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]

    assert len(macro_op_list) == 1
    assert isinstance(macro_op_list[0], VectorOp)
    assert macro_op_list[0].vector_op_type == "rms_norm"
    assert macro_op_list[0].vector_shape == [2, 16]


def test_nx_graph_meta_batch_size_override():
    graph_meta = _build_graph_meta()

    assert graph_meta.batch_size == 2

    graph_meta.batch_size = 5

    assert graph_meta.batch_size == 5
    assert graph_meta.inference_config.batch_size == 2


def test_nx_graph_meta_with_batch_size_returns_overridden_copy():
    graph_meta = _build_graph_meta()

    updated_graph_meta = graph_meta.with_batch_size(7)

    assert updated_graph_meta is not graph_meta
    assert updated_graph_meta.batch_size == 7
    assert graph_meta.batch_size == 2


def test_nx_graph_meta_batch_size_rejects_non_positive_values():
    graph_meta = _build_graph_meta()

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        graph_meta.batch_size = 0

    with pytest.raises(ValueError, match="batch_size must be > 0"):
        graph_meta.with_batch_size(0)


def test_codegen_pass_handles_linear_hooks_with_small_sram_threshold():
    class LinearModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.proj = ColumnParallelLinear(input_size=4096, output_size=2048)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.proj(x)

    with torch.device("meta"):
        graph_module = _trace_graph_module(LinearModel())

    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: _build_graph_meta()}

    CodeGenPass().transform(graph_module)

    macro_op_list = graph_module.graph.meta[CodeGenPass.MACRO_OP_LIST_META_KEY]

    assert macro_op_list
    assert any(isinstance(macro_op, MatMulOp) for macro_op in macro_op_list)


def test_row_parallel_linear_codegen_appends_all_reduce_for_tp():
    graph_meta = _build_graph_meta(
        DenseParallelConfig(num_ranks=2, tp_size=2, dp_size=1)
    )
    module = RowParallelLinear(input_size=16, output_size=12, tp_size=2)

    macro_op_list = module.macro_code_gen(graph_meta)
    all_reduce_ops = [
        macro_op for macro_op in macro_op_list if isinstance(macro_op, AllReduceOp)
    ]

    assert len(all_reduce_ops) == 1
    assert all_reduce_ops[0].num_ranks == 2
    assert all_reduce_ops[0].data_size == 48
    assert all_reduce_ops[0].input_ops
    assert all(isinstance(op, MatMulOp) for op in all_reduce_ops[0].input_ops)


def test_codegen_pass_raises_when_graph_meta_is_missing():
    class SingleHookModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.first = CountingHook("first", 1)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.first(x)

    graph_module = _trace_graph_module(SingleHookModel())
    NormalizePass().transform(graph_module)

    with pytest.raises(KeyError, match="graph.meta\\['graph_meta'\\]"):
        CodeGenPass().transform(graph_module)


def test_codegen_pass_raises_for_non_list_macro_codegen_results():
    class InvalidContainerModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bad = InvalidContainerHook()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bad(x)

    graph_module = _trace_graph_module(InvalidContainerModel())
    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: _build_graph_meta()}

    with pytest.raises(TypeError, match="must return list\\[MacroOp\\]"):
        CodeGenPass().transform(graph_module)


def test_codegen_pass_raises_for_non_macro_op_items():
    class InvalidItemModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.bad = InvalidItemHook()

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return self.bad(x)

    graph_module = _trace_graph_module(InvalidItemModel())
    NormalizePass().transform(graph_module)
    graph_module.graph.meta = {CodeGenPass.GRAPH_META_KEY: _build_graph_meta()}

    with pytest.raises(TypeError, match="returned non-MacroOp item"):
        CodeGenPass().transform(graph_module)
