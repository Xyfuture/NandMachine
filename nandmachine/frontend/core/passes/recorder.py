# Record key metadata for each module node

import math
from typing import Any

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import GraphModule

from nandmachine.frontend.core.passes.base import GraphPass
from nandmachine.frontend.network.torch_kernels import (
    Attention, LinearBase, RowParallelLinear
)


class RecorderPass(GraphPass):
    """
    A graph pass that records key metadata for nodes after FakeTensorProp.

    Records:
    - input_shapes / output_shapes for all nodes
    - module_type for call_module nodes
    - linear_info for LinearBase subclasses
    - nand_store_pages for LinearBase subclasses
    """

    BYTES_PER_ELEMENT = 2   # fp16
    PAGE_SIZE_BYTES = 4096  # 4KB per page

    def __init__(self):
        super().__init__()
        self._linear_types = (LinearBase,)
        self._attention_types = (Attention,)

    def transform(self, graph_module: GraphModule) -> fx.Graph:
        """
        Transform the graph by recording metadata for each node.

        Args:
            graph_module: The GraphModule to process

        Returns:
            The modified graph with metadata recorded in node.meta
        """
        graph = graph_module.graph

        for node in graph.nodes:
            # Record shapes for all nodes
            self._record_shapes(node)

            # For call_module nodes, record additional info
            if node.op == 'call_module':
                module = graph_module.get_submodule(node.target)
                self._record_module_type(node, module)

                # Record linear-specific info
                if isinstance(module, self._linear_types):
                    self._record_linear_info(node, module)
                    self._record_nand_pages(node, module)

        return graph

    def _record_shapes(self, node: fx.Node) -> None:
        """Record input and output shapes for a node."""
        # Extract output shapes from node.meta['val']
        if 'val' in node.meta:
            node.meta['output_shapes'] = self._extract_shapes(node.meta['val'])

        # Extract input shapes from node.args
        input_shapes = []
        for arg in node.args:
            if isinstance(arg, fx.Node) and 'val' in arg.meta:
                input_shapes.append(self._extract_shapes(arg.meta['val']))
            elif isinstance(arg, (tuple, list)):
                # Handle tuple/list of nodes
                nested_shapes = []
                for item in arg:
                    if isinstance(item, fx.Node) and 'val' in item.meta:
                        nested_shapes.append(self._extract_shapes(item.meta['val']))
                if nested_shapes:
                    input_shapes.append(nested_shapes)

        if input_shapes:
            node.meta['input_shapes'] = input_shapes

    def _extract_shapes(self, val: Any) -> Any:
        """
        Extract shapes from a value.

        Args:
            val: A tensor, tuple/list of tensors, or None

        Returns:
            Shape tuple, list of shapes, or None
        """
        if val is None:
            return None

        if isinstance(val, torch.Tensor):
            return tuple(val.shape)

        if isinstance(val, (tuple, list)):
            return [self._extract_shapes(v) for v in val]

        return None

    def _record_module_type(self, node: fx.Node, module: nn.Module) -> None:
        """Record the module type name."""
        node.meta['module_type'] = type(module).__name__

    def _record_linear_info(self, node: fx.Node, module: LinearBase) -> None:
        """Record linear-specific information."""
        linear_info = {
            'weight_shape': tuple(module.weight.shape),
            'require_all_reduce': isinstance(module, RowParallelLinear),
            'has_bias': module.bias is not None,
        }

        # Record tensor parallel info if available
        if module.tp_dim is not None:
            linear_info['tp_info'] = {
                'tp_dim': module.tp_dim,
                'tp_rank': module.tp_rank,
                'tp_size': module.tp_size,
            }

        node.meta['linear_info'] = linear_info

    def _record_nand_pages(self, node: fx.Node, module: LinearBase) -> None:
        """Calculate and record the number of NAND storage pages needed."""
        # Calculate weight storage
        weight_elements = module.weight.numel()
        total_bytes = weight_elements * self.BYTES_PER_ELEMENT

        # Add bias storage if present
        if module.bias is not None:
            bias_elements = module.bias.numel()
            total_bytes += bias_elements * self.BYTES_PER_ELEMENT

        # Calculate number of pages
        num_pages = math.ceil(total_bytes / self.PAGE_SIZE_BYTES)

        node.meta['nand_store_pages'] = num_pages

        # Also record detailed storage info for debugging
        node.meta['storage_info'] = {
            'weight_elements': weight_elements,
            'bias_elements': module.bias.numel() if module.bias is not None else 0,
            'total_bytes': total_bytes,
            'bytes_per_element': self.BYTES_PER_ELEMENT,
            'page_size_bytes': self.PAGE_SIZE_BYTES,
        }
