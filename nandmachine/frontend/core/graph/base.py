"""NxGraph/NxNode extensions built on torch.fx."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Iterable, Optional, Set, Type, cast

import torch
import torch.fx as fx
from torch.fx import node as fx_node
from torch.fx.node import Argument

from nandmachine.commands.macro import MacroOp
from nandmachine.config.config import NandConfig
from nandmachine.config.inference_config import InferenceConfig
from nandmachine.config.model_config import ModelConfigBase



def _get_fx_legal_ops() -> Set[str]:

    return cast(Set[str], fx_node._legal_ops)


_FX_LEGAL_OPS = _get_fx_legal_ops()
_HOOK_MODULE_BASE_MODULE = "nandmachine.frontend.modules.modules"
_HOOK_MODULE_BASE_NAME = "HookModuleBase"


def _is_frontend_hook_module(module: torch.nn.Module) -> bool:
    return any(
        base.__name__ == _HOOK_MODULE_BASE_NAME
        and base.__module__ == _HOOK_MODULE_BASE_MODULE
        for base in type(module).__mro__
    )

@dataclass 
class NxGraphMeta:

    nand_config:NandConfig

    model_config:ModelConfigBase

    inference_config:InferenceConfig 






class NxGraph(fx.Graph):
    """torch.fx.Graph that produces NxNode instances."""


    def call_command(
        self,
        command: Type[MacroOp],
        args: Optional[tuple["Argument", ...]] = None,
        kwargs: Optional[dict[str, "Argument"]] = None,
        name: Optional[str] = None 
    ):
    
        return self.create_node(
            "call_command", command, args,kwargs,name=name
        )
        


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
        if self._leaf_module_types and isinstance(m, tuple(self._leaf_module_types)):
            return True
        if _is_frontend_hook_module(m):
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
