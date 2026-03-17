import pytest


torch = pytest.importorskip("torch")

from torch.fx import GraphModule

from nandmachine.frontend.core.graph.base import NxTracer
from nandmachine.frontend.core.passes.normalize import NormalizePass
from nandmachine.frontend.modules.modules import RMSNorm, RotaryEmbedding


def _trace_graph_module(model: torch.nn.Module) -> GraphModule:
    graph = NxTracer().trace(model)
    return GraphModule(model, graph)


def _target_name(target: object) -> str:
    if isinstance(target, str):
        return target
    name = getattr(target, "__name__", None)
    if isinstance(name, str):
        return name
    return str(target)


def test_normalize_pass_removes_view_chain_and_keeps_basic_add():
    class ViewAddModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(hidden_size=16)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = self.norm(x)
            z = z.view(z.shape)
            return z + y

    graph_module = _trace_graph_module(ViewAddModel())

    NormalizePass().transform(graph_module)

    nodes = [(node.op, _target_name(node.target)) for node in graph_module.graph.nodes]
    norm_node = next(node for node in graph_module.graph.nodes if node.op == "call_module")
    add_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and _target_name(node.target) == "add"
    )

    assert ("call_module", "norm") in nodes
    assert ("call_function", "add") in nodes
    assert ("call_method", "view") not in nodes
    assert ("call_function", "getattr") not in nodes
    assert add_node.args[0] is norm_node
    assert add_node.args[1].name == "y"
    assert norm_node.meta["hook_module"] is graph_module.get_submodule("norm")


def test_normalize_pass_keeps_call_method_add_and_matmul():
    class ArithmeticModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(hidden_size=4)

        def forward(
            self,
            x: torch.Tensor,
            bias: torch.Tensor,
            weight: torch.Tensor,
        ) -> torch.Tensor:
            z = self.norm(x)
            z = z.add(bias)
            return torch.matmul(z, weight)

    graph_module = _trace_graph_module(ArithmeticModel())

    NormalizePass().transform(graph_module)

    nodes = [(node.op, _target_name(node.target)) for node in graph_module.graph.nodes]
    norm_node = next(node for node in graph_module.graph.nodes if node.op == "call_module")
    add_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_method" and _target_name(node.target) == "add"
    )
    matmul_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and _target_name(node.target) == "matmul"
    )

    assert ("call_module", "norm") in nodes
    assert ("call_method", "add") in nodes
    assert ("call_function", "matmul") in nodes
    assert add_node.args[0] is norm_node
    assert add_node.args[1].name == "bias"
    assert matmul_node.args[0] is add_node
    assert matmul_node.args[1].name == "weight"


def test_normalize_pass_rewires_tuple_getitem_to_hook_node():
    class RotaryModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.rotary = RotaryEmbedding(
                head_size=8,
                rotary_dim=8,
                max_position_embeddings=64,
                base=10000.0,
            )

        def forward(
            self,
            positions: torch.Tensor,
            q: torch.Tensor,
            k: torch.Tensor,
        ) -> torch.Tensor:
            q_out, k_out = self.rotary(positions, q, k)
            return q_out + k_out

    graph_module = _trace_graph_module(RotaryModel())

    NormalizePass().transform(graph_module)

    nodes = [(node.op, _target_name(node.target)) for node in graph_module.graph.nodes]
    rotary_node = next(node for node in graph_module.graph.nodes if node.op == "call_module")
    add_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and _target_name(node.target) == "add"
    )

    assert ("call_module", "rotary") in nodes
    assert ("call_function", "getitem") not in nodes
    assert add_node.args[0] is rotary_node
    assert add_node.args[1] is rotary_node


def test_normalize_pass_removes_where_chain_to_first_input():
    class WhereModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.norm = RMSNorm(hidden_size=4)

        def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
            z = self.norm(x)
            bad = torch.where(z > 0, z, y)
            return bad + y

    graph_module = _trace_graph_module(WhereModel())

    NormalizePass().transform(graph_module)

    nodes = [(node.op, _target_name(node.target)) for node in graph_module.graph.nodes]
    norm_node = next(node for node in graph_module.graph.nodes if node.op == "call_module")
    add_node = next(
        node
        for node in graph_module.graph.nodes
        if node.op == "call_function" and _target_name(node.target) == "add"
    )

    assert ("call_function", "gt") not in nodes
    assert ("call_function", "where") not in nodes
    assert add_node.args[0] is norm_node
    assert add_node.args[1].name == "y"
