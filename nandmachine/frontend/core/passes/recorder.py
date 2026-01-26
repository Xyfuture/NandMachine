# 记录各个module 的关键信息 

import math
from typing import Any

import torch
import torch.nn as nn
import torch.fx as fx
from torch.fx import GraphModule


from nandmachine.frontend.core.passes.base import GraphPass
from nandmachine.frontend.network.torch_kernels import Attention, LinearBase, RowParallelLinear


# 所有的东西在这里面做好 ， 负载内容的记载
"""
负责内容
在 node.meta 中记录关键信息

- 记录 输入/输出的 shape --> 多个输入/输出情况的特判
- module_type -> 从 graph module 中获取，最后记录下来
- linear_info 针对 linear 类型单独的记录
    - weight_shape 
    - require_all_reduce ? --> 有无更好的方式记录这个信息
- attention_info 针对 attention 类型单独的记录
    - pass -> 先不记录了， 后面想好了再重新记录
- nand_store_pages -> 记录这个 node 需要存储多少 nand pages, 不需要存储数据的可以跳过
    - 目前对于 linear 需要进行存储


"""



def _shape_from_tensor_meta(meta: Any):
    if meta is None:
        return None
    if hasattr(meta, "shape"):
        try:
            return tuple(meta.shape)
        except Exception:
            return None
    return None


def _extract_shape(val: Any):
    if val is None:
        return None
    if isinstance(val, torch.Tensor):
        return tuple(val.shape)
    if isinstance(val, (list, tuple)):
        return type(val)(_extract_shape(v) for v in val)
    if isinstance(val, dict):
        return {k: _extract_shape(v) for k, v in val.items()}
    return _shape_from_tensor_meta(val)


def _node_output_shape(node: fx.Node):
    tensor_meta = node.meta.get("tensor_meta")
    shape = _shape_from_tensor_meta(tensor_meta)
    if shape is not None:
        return shape
    if "val" in node.meta:
        return _extract_shape(node.meta["val"])
    return None


def _map_arg_shapes(arg: Any):
    if isinstance(arg, fx.Node):
        return _node_output_shape(arg)
    if isinstance(arg, (list, tuple)):
        return type(arg)(_map_arg_shapes(v) for v in arg)
    if isinstance(arg, dict):
        return {k: _map_arg_shapes(v) for k, v in arg.items()}
    return _extract_shape(arg)


class RecorderPass(GraphPass):
    def __init__(self) -> None:
        super().__init__()

        self._to_record_type = {
            LinearBase,
            nn.Linear,
            Attention
        }
    

    def transform(self, gm: GraphModule):
        graph = gm.graph

        # Build module dictionary for fast lookup
        module_dict = dict(gm.named_modules())

        for node in graph.nodes:
            # Only process module call nodes
            if node.op == "call_module":
                module_name = node.target

                # Get the actual module object
                if module_name in module_dict:
                    target_module = module_dict[module_name]

                    # Check if module type should be recorded
                    if type(target_module) in self._to_record_type:
                        # Record the target object in node metadata
                        node.meta["target_obj"] = target_module
                        node.meta["module_type"] = type(target_module).__name__

                        input_shapes = _map_arg_shapes(node.args)
                        if node.kwargs:
                            input_shapes = {
                                "args": input_shapes,
                                "kwargs": _map_arg_shapes(node.kwargs),
                            }
                        node.meta["input_shapes"] = input_shapes
                        node.meta["output_shapes"] = _node_output_shape(node)

                        if isinstance(target_module, (LinearBase, nn.Linear)):
                            linear_info = {}
                            if hasattr(target_module, "weight") and target_module.weight is not None:
                                linear_info["weight_shape"] = tuple(target_module.weight.shape)
                                weight_numel = target_module.weight.numel()
                                weight_bytes = weight_numel * 2  # fp16 weights
                                pages = math.ceil(weight_bytes / 4096)
                                if pages > 0:
                                    node.meta["nand_store_pages"] = pages
                            linear_info["require_all_reduce"] = isinstance(target_module, RowParallelLinear)
                            node.meta["linear_info"] = linear_info


    
