"""Normalize FX graphs by pruning non-core nodes."""

from __future__ import annotations

import torch.fx as fx
from torch.fx import GraphModule

from nandmachine.frontend.core.passes.base import GraphPass
from nandmachine.frontend.modules.modules import HookModuleBase


class NormalizePass(GraphPass):
    """Keep Hook modules and core arithmetic while pruning everything else."""

    BASIC_OP_NAMES = frozenset({"add", "sub", "mul", "div", "matmul"})
    HOOK_MODULE_META_KEY = "hook_module"

    def transform(self, graph_module: GraphModule) -> fx.Graph:
        """Prune non-core nodes from a GraphModule in place."""
        graph = graph_module.graph

        for node in reversed(list(graph.nodes)):
            if self._should_keep(graph_module, node):
                continue
            self._remove_node(graph, node)

        graph.lint()
        graph_module.recompile()
        return graph

    def _should_keep(self, graph_module: GraphModule, node: fx.Node) -> bool:
        if node.op in {"placeholder", "output", "get_attr"}:
            return True

        if node.op == "call_module":
            return self._is_hook_module_node(graph_module, node)

        if node.op in {"call_function", "call_method"}:
            return self._matches_target(node, self.BASIC_OP_NAMES)

        return False

    def _is_hook_module_node(self, graph_module: GraphModule, node: fx.Node) -> bool:
        module = graph_module.get_submodule(str(node.target))
        if not isinstance(module, HookModuleBase):
            return False

        node.meta[self.HOOK_MODULE_META_KEY] = module
        # print(node.meta)
        return True

    def _remove_node(self, graph: fx.Graph, node: fx.Node) -> None:
        input_nodes = list(node.all_input_nodes)
        if input_nodes and node.users:
            node.replace_all_uses_with(input_nodes[0])
        graph.erase_node(node)

    def _matches_target(self, node: fx.Node, expected_names: set[str] | frozenset[str]) -> bool:
        return any(token in expected_names for token in self._target_tokens(node.target))

    def _target_tokens(self, target: object) -> set[str]:
        candidates: list[str] = []

        if isinstance(target, str):
            candidates.append(target.lower())
        else:
            for attr_name in ("__name__", "__qualname__"):
                attr_value = getattr(target, attr_name, None)
                if isinstance(attr_value, str):
                    candidates.append(attr_value.lower())

            overload_packet = getattr(target, "overloadpacket", None)
            overload_name = getattr(overload_packet, "__name__", None)
            if isinstance(overload_name, str):
                candidates.append(overload_name.lower())

            candidates.append(str(target).lower())

        tokens: set[str] = set()
        for candidate in candidates:
            normalized = candidate.replace(".", "_").replace("-", "_")
            tokens.update(part for part in normalized.split("_") if part)

        return tokens


__all__ = ["NormalizePass"]
