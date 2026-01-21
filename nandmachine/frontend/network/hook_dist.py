"""
Fake torch.distributed implementation for single GPU testing.

This module provides a monkey-patched version of torch.distributed that allows
code to run in single GPU environments without requiring actual distributed setup.
"""

import torch
from typing import Optional


class FakeDistributed:
    """Fake torch.distributed implementation for single GPU environment"""

    def __init__(self, world_size: int = 1):
        """
        Initialize fake distributed environment.

        Args:
            world_size: Simulated number of GPUs (default: 1)
        """
        self._world_size = world_size
        self._rank = 0

    def get_rank(self) -> int:
        """Always return rank 0 (simulating single process)"""
        return self._rank

    def get_world_size(self) -> int:
        """Return configured world size"""
        return self._world_size

    def all_reduce(self, tensor: torch.Tensor, op=None, group=None, async_op=False):
        """
        Fake all_reduce operation.

        - For regular tensors: return tensor as-is (no actual reduction)
        - For meta tensors: preserve shape and dtype

        Args:
            tensor: Input tensor to reduce
            op: Reduction operation (ignored in fake implementation)
            group: Process group (ignored in fake implementation)
            async_op: Whether to perform async operation (ignored)

        Returns:
            The input tensor unchanged
        """
        # # Support meta device for shape inference
        # if tensor.device.type == 'meta':
        #     return tensor

        # For real tensors, just return as-is (simulating single GPU)
        return tensor

    def set_world_size(self, world_size: int):
        """
        Update world size configuration.

        Args:
            world_size: New world size value
        """
        self._world_size = world_size


# Global fake distributed instance
_fake_dist = FakeDistributed(world_size=1)


# Export functions matching torch.distributed API
def get_rank() -> int:
    """Get current process rank (always 0 in fake implementation)"""
    return _fake_dist.get_rank()


def get_world_size() -> int:
    """Get total number of processes (configurable in fake implementation)"""
    return _fake_dist.get_world_size()


def all_reduce(tensor: torch.Tensor, op=None, group=None, async_op=False):
    """
    Perform all-reduce operation (no-op in fake implementation).

    Args:
        tensor: Input tensor
        op: Reduction operation (ignored)
        group: Process group (ignored)
        async_op: Async flag (ignored)

    Returns:
        Input tensor unchanged
    """
    return _fake_dist.all_reduce(tensor, op, group, async_op)


def set_world_size(world_size: int):
    """
    Helper function to configure world size.

    Args:
        world_size: Number of simulated GPUs
    """
    _fake_dist.set_world_size(world_size)
