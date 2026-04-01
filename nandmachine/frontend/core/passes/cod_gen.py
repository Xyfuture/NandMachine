"""Generate macro op lists from normalized hook-module nodes."""

from __future__ import annotations

import logging

import torch.fx as fx
from torch.fx import GraphModule

from nandmachine.commands.macro import MacroOp
from nandmachine.frontend.core.graph.base import NxGraphMeta
from nandmachine.frontend.core.passes.base import GraphPass


logger = logging.getLogger(__name__)


class CodeGenPass(GraphPass):
    """Collect macro ops emitted by hook modules in graph order."""

    GRAPH_META_KEY = "graph_meta"
    HOOK_MODULE_META_KEY = "hook_module"
    MACRO_OP_LIST_META_KEY = "macro_op_list"

    def transform(self, graph_module: GraphModule) -> fx.Graph:
        graph = graph_module.graph
        graph_meta_dict = self._get_graph_meta_dict(graph)
        nx_graph_meta = self._get_nx_graph_meta(graph_meta_dict)

        macro_op_list: list[MacroOp] = []
        for node in graph.nodes:
            hook_module = node.meta.get(self.HOOK_MODULE_META_KEY)
            if hook_module is None:
                continue

            node_macro_ops = hook_module.macro_code_gen(nx_graph_meta)
            validated_ops = self._validate_macro_ops(node, node_macro_ops)
            macro_op_list.extend(validated_ops)

            logger.info(
                "Processed node=%s generated_macro_ops=%d",
                node.name,
                len(validated_ops),
            )
            node.meta['marco_op_list'] = validated_ops
 
        graph_meta_dict[self.MACRO_OP_LIST_META_KEY] = macro_op_list
        return graph

    def _get_graph_meta_dict(self, graph: fx.Graph) -> dict[str, object]:
        if not hasattr(graph, "meta"):
            graph.meta = {}  # type: ignore[attr-defined]

        if not isinstance(graph.meta, dict):  # type: ignore[attr-defined]
            raise TypeError("graph.meta must be a dict before running CodeGenPass")

        return graph.meta  # type: ignore[return-value, attr-defined]

    def _get_nx_graph_meta(self, graph_meta_dict: dict[str, object]) -> NxGraphMeta:
        nx_graph_meta = graph_meta_dict.get(self.GRAPH_META_KEY)
        if nx_graph_meta is None:
            raise KeyError(
                f"graph.meta['{self.GRAPH_META_KEY}'] must be set to an NxGraphMeta instance"
            )

        if not isinstance(nx_graph_meta, NxGraphMeta):
            raise TypeError(
                f"graph.meta['{self.GRAPH_META_KEY}'] must be an NxGraphMeta instance"
            )

        return nx_graph_meta

    def _validate_macro_ops(
        self,
        node: fx.Node,
        node_macro_ops: object,
    ) -> list[MacroOp]:
        if not isinstance(node_macro_ops, list):
            raise TypeError(
                f"Node '{node.name}' macro_code_gen must return list[MacroOp], "
                f"got {type(node_macro_ops).__name__}"
            )

        invalid_macro_op = next(
            (macro_op for macro_op in node_macro_ops if not isinstance(macro_op, MacroOp)),
            None,
        )
        if invalid_macro_op is not None:
            raise TypeError(
                f"Node '{node.name}' macro_code_gen returned non-MacroOp item "
                f"of type {type(invalid_macro_op).__name__}"
            )

        return node_macro_ops


__all__ = ["CodeGenPass"]
