"""Runtime patch for the upstream ScaleSim bug in total cycle aggregation.

This module monkey-patches
`double_buffered_scratchpad.service_memory_requests` at runtime to fix a
TypeError caused by converting a non-scalar NumPy array to `int` when
computing `self.total_cycles`.

Why this exists:
- We must keep using the public ScaleSim package without editing site-packages.
- LUT miss paths in matmul/flash-attention still need to execute ScaleSim.
"""

import inspect
import textwrap

import numpy as np
from scalesim.memory.double_buffered_scratchpad_mem import double_buffered_scratchpad

_PATCH_APPLIED = False


def apply_scalesim_total_cycles_patch() -> None:
    global _PATCH_APPLIED
    if _PATCH_APPLIED:
        return

    target = double_buffered_scratchpad.service_memory_requests
    source = inspect.getsource(target)
    buggy_line = "self.total_cycles = int(max(ofmap_serviced_cycles))"
    fixed_line = "self.total_cycles = int(np.asarray(ofmap_serviced_cycles).reshape(-1).max())"
    if buggy_line not in source:
        raise RuntimeError(
            "Unsupported scalesim service_memory_requests implementation: expected buggy line not found"
        )

    patched_source = source.replace(buggy_line, fixed_line, 1)
    namespace = dict(target.__globals__)
    namespace["np"] = np
    exec(textwrap.dedent(patched_source), namespace)
    patched = namespace.get(target.__name__)
    if patched is None:
        raise RuntimeError(
            "Failed to build patched scalesim service_memory_requests function"
        )

    setattr(double_buffered_scratchpad, target.__name__, patched)
    _PATCH_APPLIED = True
