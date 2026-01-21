"""NxGraph/NxNode extensions built on torch.fx."""

from __future__ import annotations

from typing import ClassVar, Iterable, Optional, Set, cast

import torch
import torch.fx as fx
from torch.fx import node as fx_node



def _get_fx_legal_ops() -> Set[str]:

    return cast(Set[str], fx_node._legal_ops)


_FX_LEGAL_OPS = _get_fx_legal_ops()


class NxNode(fx.Node):
    """torch.fx.Node with extensible op types."""

    extra_ops: ClassVar[Set[str]] = set()
    auto_register_ops: ClassVar[bool] = True

    @classmethod
    def register_op_type(cls, op: str) -> None:
        cls._validate_op(op)
        cls.extra_ops.add(op)
        cls._sync_legal_ops()

    @classmethod
    def register_op_types(cls, ops: Iterable[str]) -> None:
        for op in ops:
            cls._validate_op(op)
        cls.extra_ops.update(ops)
        cls._sync_legal_ops()

    @classmethod
    def _validate_op(cls, op: str) -> None:
        if not isinstance(op, str) or not op:
            raise ValueError("op must be a non-empty string")

    @classmethod
    def _sync_legal_ops(cls) -> None:
        if _FX_LEGAL_OPS is not None:
            _FX_LEGAL_OPS.update(cls.extra_ops)

    @classmethod
    def _ensure_legal_op(cls, op: str) -> None:
        cls._validate_op(op)
        if op in _FX_LEGAL_OPS:
            return
        if not cls.auto_register_ops:
            raise ValueError(f"Unsupported op type: {op!r}")
        cls.extra_ops.add(op)
        _FX_LEGAL_OPS.add(op)

    def __init__(self, *args, **kwargs):
        if "op" in kwargs:
            op = kwargs["op"]
        elif len(args) >= 3:
            op = args[2]
        else:
            raise TypeError("op is required to construct NxNode")
        self._ensure_legal_op(op)
        super().__init__(*args, **kwargs)


class NxGraph(fx.Graph):
    """torch.fx.Graph that produces NxNode instances."""

    node_cls: ClassVar[type[NxNode]] = NxNode

    def create_node(self, *args, **kwargs):
        if "op" in kwargs:
            op = kwargs["op"]
        elif args:
            op = args[0]
        else:
            raise TypeError("op is required to create a node")
        NxNode._ensure_legal_op(op)
        return super().create_node(*args, **kwargs)


class NxTracer(fx.Tracer):
    """torch.fx.Tracer that builds NxGraph and supports custom leaf modules."""

    def __init__(
        self,
        *,
        leaf_modules: Iterable[type[torch.nn.Module]] = (),
        leaf_module_names: Iterable[str] = (),
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self._leaf_module_types: Set[type[torch.nn.Module]] = set(leaf_modules)
        self._leaf_module_names: Set[str] = set(leaf_module_names)

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        if type(m) in self._leaf_module_types:
            return True
        if module_qualified_name in self._leaf_module_names:
            return True
        return super().is_leaf_module(m, module_qualified_name)

    def trace(self, root, concrete_args=None) -> fx.Graph:
        graph = super().trace(root, concrete_args=concrete_args)
        if not isinstance(graph, NxGraph):
            tracer_cls = getattr(self, "__class__", None)
            nx_graph = NxGraph(tracer_cls=tracer_cls)
            nx_graph.graph_copy(graph, {})  # Preserve node metadata and ordering.
            nx_graph._co_fields = dict(getattr(graph, "_co_fields", {}))
            nx_graph._tracer_extras = getattr(graph, "_tracer_extras", None)
            graph = nx_graph
            self.graph = nx_graph
        return graph


__all__ = ["NxGraph", "NxNode", "NxTracer"]
